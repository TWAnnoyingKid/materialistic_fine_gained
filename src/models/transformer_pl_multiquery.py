import os
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import pytorch_lightning as pl
import src.models.model_utils as model_utils
import src.loss_functions.loss_utils as loss_utils
import src.utils.utils as utils
from omegaconf import DictConfig


class Transformer_pl_MultiQuery(pl.LightningModule):
    def __init__(self, conf: DictConfig, args):
        super().__init__()
        print("============================================================")
        print("Initializing Multi-Query Training Model")
        print("============================================================")

        self.use_chroma = args.use_chroma
        self.use_depth = args.use_depth
        self.use_normal = args.use_normal
        self.lr = args.lr
        self.print_freq = args.print_every

        self.num_queries_per_image = getattr(args, 'num_queries', 10)
        self.max_batch_queries = getattr(args, 'max_batch_queries', 4)

        print(f"Multi-query config: num_queries_per_image={self.num_queries_per_image}, max_batch_queries={self.max_batch_queries}")

        self.net = model_utils.create_model(conf)
        self.criterion = loss_utils.create_loss_function(conf)

    def forward(self, x, reference_locations):
        return self.net(x, reference_locations)

    def _prob_to_logit(self, scores, eps: float = 1e-6):
        # no longer needed; model now outputs logits directly
        return scores

    def training_step(self, batch, batch_idx):
        image = batch['images']
        mat_labels = batch["mat_labels"]
        reference_locations = batch["reference_locations"]

        B, C, H, W = image.shape

        # 先為每張圖採樣多個查詢點與對應標籤（不重複編碼）
        queries_per_image = []
        labels_per_image = []
        for b in range(B):
            q_list = []
            l_list = []
            unique_mats = torch.unique(mat_labels[b])
            unique_mats = unique_mats[unique_mats != 0]
            if len(unique_mats) == 0:
                queries_per_image.append(q_list)
                labels_per_image.append(l_list)
                continue
            for _ in range(self.num_queries_per_image):
                mat_idx = torch.randint(len(unique_mats), (1,), device=mat_labels.device).item()
                selected_mat = unique_mats[mat_idx]
                mat_mask = (mat_labels[b] == selected_mat)
                valid_coords = torch.nonzero(mat_mask, as_tuple=False)
                if valid_coords.numel() == 0:
                    continue
                coord_idx = torch.randint(valid_coords.shape[0], (1,), device=mat_labels.device).item()
                query_h, query_w = valid_coords[coord_idx]
                q_list.append(torch.stack([query_h, query_w]))
                l_list.append((mat_labels[b] == selected_mat).float())
            queries_per_image.append(q_list)
            labels_per_image.append(l_list)

        any_query = any(len(q_list) > 0 for q_list in queries_per_image)
        if not any_query:
            scores, *_ = self.net(image, reference_locations)
            loss = self.criterion(scores, mat_labels)
            return loss

        total_loss = 0.0
        num_batches = 0
        total_queries = 0

        # 對每張圖：先 encode 一次，再將聚合特徵複製至查詢批次維度，僅跑 attention/head
        for b in range(B):
            if len(queries_per_image[b]) == 0:
                continue
            enc = self.net.encode_image(image[b:b+1])
            agg = enc["agg"]
            out_size = enc["output_size"]
            q_tensor = torch.stack(queries_per_image[b])
            l_tensor = torch.stack(labels_per_image[b])
            total_queries += q_tensor.shape[0]
            for i in range(0, q_tensor.shape[0], self.max_batch_queries):
                end_i = min(i + self.max_batch_queries, q_tensor.shape[0])
                q_batch = q_tensor[i:end_i]
                l_batch = l_tensor[i:end_i]
                agg_batched = tuple(feat.repeat(q_batch.shape[0], 1, 1, 1) for feat in agg)
                scores, *_ = self.net.forward_with_features(agg_batched, q_batch, out_size)
                loss = self.criterion(scores, l_batch)
                total_loss = total_loss + loss
                num_batches += 1

        avg_loss = total_loss / max(1, num_batches)

        if batch_idx % self.print_freq == 0:
            self.log("train_loss", avg_loss.item(), on_step=True, on_epoch=False)
            self.log("train_num_queries", float(total_queries), on_step=True, on_epoch=False)
            try:
                with torch.no_grad():
                    viz_scores, path1, *others = self.net(image[[0]], reference_locations[[0]])
                viz_data = {
                    "images": image[[0]],
                    "mat_labels": (mat_labels[[0]] == mat_labels[0, reference_locations[0, 0], reference_locations[0, 1]]).float(),
                    "reference_locations": reference_locations[[0]],
                    "scores": viz_scores,
                    "path1": path1,
                    "context_embeddings_1": others[0],
                    "layer_1": others[4],
                }
                viz_np, *_ = utils.get_classification_visualization(viz_data)
                self.logger.experiment.add_image("train_images", viz_np, self.global_step)
            except Exception as e:
                print(f"Visualization failed: {e}")

        return avg_loss

    def validation_step(self, batch, batch_idx):
        image = batch['images']
        mat_labels = batch["mat_labels"]
        reference_locations = batch["reference_locations"]
        scores, *_ = self.net(image, reference_locations)
        loss = self.criterion(self._prob_to_logit(scores), mat_labels)
        self.log("val_loss", loss.item(), on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def load_checkpoint(self, checkpoint_path, map_location=None):
        files = os.listdir(checkpoint_path)
        if map_location is None:
            map_location = torch.device('cuda:0')
        files = [f for f in files if ".ckpt" in f] + [f for f in files if "model.pth" in f]
        if len(files) == 0:
            print("No checkpoint found")
            return
        latest_file = files[-1]
        if '.ckpt' in latest_file:
            checkpoint = torch.load(os.path.join(checkpoint_path, latest_file), map_location=map_location)
            self.load_state_dict(checkpoint['state_dict'])
        else:
            state_dict = torch.load(os.path.join(checkpoint_path, latest_file), map_location=map_location)
            for key in list(state_dict.keys()):
                state_dict['net.' + key.replace("module.", '')] = state_dict.pop(key)
            self.load_state_dict(state_dict)


