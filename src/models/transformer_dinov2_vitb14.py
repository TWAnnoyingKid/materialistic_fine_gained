import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

activations = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index
        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features))

    def forward(self, x, global_token=None):
        if global_token is None:
            global_token = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], global_token), -1)
        return self.project(features)


class Unflatten(nn.Module):
    def __init__(self):
        super(Unflatten, self).__init__()

    def forward(self, x):
        size = int(x.shape[1] ** 0.5)
        spatial_transform = nn.Sequential(
            Transpose(1, 2), nn.Unflatten(2, torch.Size([size, size]))
        )
        x = spatial_transform(x)
        return x


def init_weights(self, init_type='xavier_uniform', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':
                m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    self.apply(init_func)
    for n, m in self.named_children():
        m.apply(init_weights)


def local_pixel_coord(x, y, s):
    patch_idx_x = torch.div(x, s, rounding_mode="trunc")
    patch_idx_y = torch.div(y, s, rounding_mode="trunc")

    patch_center_x = patch_idx_x * s + (s - 1.)/2.
    patch_center_y = patch_idx_y * s + (s - 1.)/2.

    out_x = (x - patch_center_x) / ((s - 1) / 2 + 1e-4)
    out_y = (y - patch_center_y) / ((s - 1) / 2 + 1e-4)
    return out_x, out_y


class ReferenceEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ReferenceEmbedding, self).__init__()
        self.reference_feature_extractor = nn.Sequential(
            nn.Linear(in_channels + 2, in_channels * 2),
            nn.ReLU(),
            nn.Linear(in_channels * 2, out_channels)
        )
        self.stride = stride

    def forward(self, embeddings, reference_locations):
        reference_locations_patch = torch.div(reference_locations, self.stride, rounding_mode="trunc")
        # clamp to valid spatial range to avoid out-of-bounds
        Hf, Wf = embeddings.shape[2], embeddings.shape[3]
        ref_h = reference_locations_patch[:, 0].clamp(0, Hf - 1).type(torch.long)
        ref_w = reference_locations_patch[:, 1].clamp(0, Wf - 1).type(torch.long)
        reference_embeddings = embeddings[range(embeddings.shape[0]), :, ref_h, ref_w]

        local_coordinate = local_pixel_coord(reference_locations[:, 0].type(torch.long), reference_locations[:, 1].type(torch.long), self.stride)
        local_coordinate = torch.stack(local_coordinate, dim=1).to(embeddings.device)

        reference_embeddings = torch.cat([reference_embeddings, local_coordinate], dim=1)
        reference_embeddings = self.reference_feature_extractor(reference_embeddings)
        return reference_embeddings


class CrossAttentionWithReferenceEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=16, stride=1):
        super(CrossAttentionWithReferenceEmbedding, self).__init__()
        self.num_heads = num_heads
        head_dim = in_channels // num_heads
        self.scale = head_dim ** -0.5

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_channels = out_channels // num_heads
        assert self.head_channels * num_heads == out_channels, "out_channels must be divisible by num_heads"

        self.Unflatten = Unflatten()
        self.LayerNorm = nn.LayerNorm(in_channels)
        self.get_reference_embedding = ReferenceEmbedding(in_channels, in_channels, stride=stride)
        self.query = nn.Linear(in_channels, out_channels)
        self.key = nn.Linear(in_channels, out_channels)
        self.value = nn.Linear(in_channels, out_channels)
        self.out = nn.Linear(out_channels, out_channels)
        self.out_norm = nn.LayerNorm(out_channels)

    def forward(self, embeddings, reference_locations, reference_embeddings=None, stride_override=None):
        B, N, C = embeddings.shape
        embeddings = self.LayerNorm(embeddings)
        if reference_embeddings is None:
            # allow dynamic stride override at runtime for accurate coordinate mapping
            if stride_override is not None:
                self.get_reference_embedding.stride = int(stride_override)
            reference_embeddings = self.get_reference_embedding(self.Unflatten(embeddings), reference_locations)

        q = self.query(reference_embeddings).reshape(B, 1, self.num_heads, self.head_channels).permute(0, 2, 1, 3)
        k = self.key(embeddings).reshape(B, N, self.num_heads, self.head_channels).permute(0, 2, 1, 3)
        v = self.value(embeddings).reshape(B, N, self.num_heads, self.head_channels).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.sigmoid(attn)

        x = (attn.transpose(-1, -2) * v).permute(0, 2, 1, 3).reshape(B, N, self.out_channels)
        x = self.out(x)
        x = self.out_norm(x)
        return x, attn, q, k, v, reference_embeddings


class ResidualConvUnit_custom(nn.Module):
    def __init__(self, features, activation):
        super().__init__()
        self.groups = 1
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False, groups=self.groups)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock_custom(nn.Module):
    def __init__(self, features, activation, align_corners=True, upsample=True, upsample_scale=2.0):
        super(FeatureFusionBlock_custom, self).__init__()
        self.align_corners = align_corners
        out_features = features
        self.groups = 1
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.resConfUnit1 = ResidualConvUnit_custom(features, activation)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation)
        self.upsample = upsample
        self.upsample_scale = upsample_scale
        self.layer_norm_1 = nn.LayerNorm(out_features)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        if self.upsample:
            output = nn.functional.interpolate(output, scale_factor=self.upsample_scale, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


class Transformer_DINOv2_ViTB14(nn.Module):
    def __init__(self):
        super().__init__()

        # Load DINOv2 ViT-B/14
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
        self.dino_model.eval()

        # Register hooks to capture intermediate block outputs (tokens incl. CLS)
        hooks = [2, 5, 8, 11]
        self.dino_model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
        self.dino_model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
        self.dino_model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
        self.dino_model.blocks[hooks[3]].register_forward_hook(get_activation("4"))
        self.activations = activations

        vit_features = 768
        attn_out_feats = 256
        out_channels = 256

        # token->spatial projection shared across blocks
        self.project_readout = ProjectReadout(vit_features, 1)
        self.unflatten = Unflatten()
        self.linear_proj = nn.Conv2d(in_channels=vit_features, out_channels=attn_out_feats, kernel_size=1, stride=1, padding=0)

        # Aggregation reduction after concat(I1, I2_tiled)
        self.reduce_1 = nn.Conv2d(in_channels=attn_out_feats * 2, out_channels=attn_out_feats, kernel_size=1)
        self.reduce_2 = nn.Conv2d(in_channels=attn_out_feats * 2, out_channels=attn_out_feats, kernel_size=1)
        self.reduce_3 = nn.Conv2d(in_channels=attn_out_feats * 2, out_channels=attn_out_feats, kernel_size=1)
        self.reduce_4 = nn.Conv2d(in_channels=attn_out_feats * 2, out_channels=attn_out_feats, kernel_size=1)

        # Cross-attention blocks; strides map from input(1024) to feature maps (512/256/128/128)
        self.cross_attention_1 = CrossAttentionWithReferenceEmbedding(attn_out_feats, out_channels, stride=2)
        self.cross_attention_2 = CrossAttentionWithReferenceEmbedding(attn_out_feats, out_channels, stride=4)
        self.cross_attention_3 = CrossAttentionWithReferenceEmbedding(attn_out_feats, out_channels, stride=8)
        self.cross_attention_4 = CrossAttentionWithReferenceEmbedding(attn_out_feats, out_channels, stride=8)

        # Feature fusion head (DPT-style): upsample x2 for path1-3; path4 no upsample
        self.fusion_1 = FeatureFusionBlock_custom(out_channels, nn.ReLU(), align_corners=True, upsample=True, upsample_scale=2.0)
        self.fusion_2 = FeatureFusionBlock_custom(out_channels, nn.ReLU(), align_corners=True, upsample=True, upsample_scale=2.0)
        self.fusion_3 = FeatureFusionBlock_custom(out_channels, nn.ReLU(), align_corners=True, upsample=True, upsample_scale=2.0)
        self.fusion_4 = FeatureFusionBlock_custom(out_channels, nn.ReLU(), align_corners=True, upsample=False)

        # Per-pixel MLP head
        self.out_conv = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.linear_proj.apply(init_weights)
        self.reduce_1.apply(init_weights)
        self.reduce_2.apply(init_weights)
        self.reduce_3.apply(init_weights)
        self.reduce_4.apply(init_weights)
        self.cross_attention_1.apply(init_weights)
        self.cross_attention_2.apply(init_weights)
        self.cross_attention_3.apply(init_weights)
        self.cross_attention_4.apply(init_weights)
        self.fusion_1.apply(init_weights)
        self.fusion_2.apply(init_weights)
        self.fusion_3.apply(init_weights)
        self.fusion_4.apply(init_weights)
        self.out_conv.apply(init_weights)

    def _split_tiles(self, x, r: int = 518, mult: int = 14):
        B, C, H, W = x.shape
        pad_h = (((H + mult - 1) // mult) * mult) - H
        pad_w = (((W + mult - 1) // mult) * mult) - W
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        th, tw = x_pad.shape[-2] // 2, x_pad.shape[-1] // 2
        tiles = [
            x_pad[..., 0:th, 0:tw],
            x_pad[..., 0:th, tw:tw*2],
            x_pad[..., th:th*2, 0:tw],
            x_pad[..., th:th*2, tw:tw*2],
        ]
        tiles = [F.interpolate(t, size=(r, r), mode='bilinear', align_corners=True) for t in tiles]
        return tiles

    def encode_image(self, x):
        """Encode image once (I1 full + I2 tiles), aggregate multi-scale features.
        Returns a dict with:
          - 'agg': tuple(agg_1, agg_2, agg_3, agg_4)
          - 'output_size': (H, W)
          - 'layers': (layer_1, layer_2, layer_3, layer_4) from I1 tokens
        """
        B, C, H, W = x.shape
        r = 518
        # I1 full
        i1 = F.interpolate(x, size=(r, r), mode='bilinear', align_corners=True)
        t_i1 = self._encode_tokens(i1)
        # I2 tiles
        tiles = self._split_tiles(x, r=r)
        t_tiles = [self._encode_tokens(t) for t in tiles]

        # project tokens to spatial per block
        sp_i1_1 = self._tokens_to_spatial(t_i1["1"])  # (B,256,37,37)
        sp_i1_2 = self._tokens_to_spatial(t_i1["2"])  # (B,256,37,37)
        sp_i1_3 = self._tokens_to_spatial(t_i1["3"])  # (B,256,37,37)
        sp_i1_4 = self._tokens_to_spatial(t_i1["4"])  # (B,256,37,37)

        sp_t_1 = [self._tokens_to_spatial(t_["1"]) for t_ in t_tiles]
        sp_t_2 = [self._tokens_to_spatial(t_["2"]) for t_ in t_tiles]
        sp_t_3 = [self._tokens_to_spatial(t_["3"]) for t_ in t_tiles]
        sp_t_4 = [self._tokens_to_spatial(t_["4"]) for t_ in t_tiles]

        sp_i2_1 = self._tile_concat_2x2(*sp_t_1)
        sp_i2_2 = self._tile_concat_2x2(*sp_t_2)
        sp_i2_3 = self._tile_concat_2x2(*sp_t_3)
        sp_i2_4 = self._tile_concat_2x2(*sp_t_4)

        # target sizes unified to 74x74 for all blocks
        target_1 = target_2 = target_3 = target_4 = (74, 74)

        i1_1 = F.interpolate(sp_i1_1, size=target_1, mode='bilinear', align_corners=True)
        i2_1 = F.interpolate(sp_i2_1, size=target_1, mode='bilinear', align_corners=True)
        agg_1 = self.reduce_1(torch.cat([i1_1, i2_1], dim=1))

        i1_2 = F.interpolate(sp_i1_2, size=target_2, mode='bilinear', align_corners=True)
        i2_2 = F.interpolate(sp_i2_2, size=target_2, mode='bilinear', align_corners=True)
        agg_2 = self.reduce_2(torch.cat([i1_2, i2_2], dim=1))

        i1_3 = F.interpolate(sp_i1_3, size=target_3, mode='bilinear', align_corners=True)
        i2_3 = F.interpolate(sp_i2_3, size=target_3, mode='bilinear', align_corners=True)
        agg_3 = self.reduce_3(torch.cat([i1_3, i2_3], dim=1))

        i1_4 = F.interpolate(sp_i1_4, size=target_4, mode='bilinear', align_corners=True)
        i2_4 = F.interpolate(sp_i2_4, size=target_4, mode='bilinear', align_corners=True)
        agg_4 = self.reduce_4(torch.cat([i1_4, i2_4], dim=1))

        return {
            "agg": (agg_1, agg_2, agg_3, agg_4),
            "output_size": (H, W),
            "layers": (t_i1["1"], t_i1["2"], t_i1["3"], t_i1["4"]),
        }

    def forward_with_features(self, agg_features, reference_locations, out_size):
        """Use precomputed aggregated features and run cross-attention + fusion + head.
        agg_features: tuple(agg_1..agg_4) with shape (B, 256, H_i, W_i)
        reference_locations: (B, 2) or (Q, 2) matching batch of agg_features
        out_size: (H, W) to upsample predictions back
        Returns: scores(sigmoid), path1..path4
        """
        agg_1, agg_2, agg_3, agg_4 = agg_features
        B = agg_1.shape[0]
        target_1 = (agg_1.shape[2], agg_1.shape[3])
        target_2 = (agg_2.shape[2], agg_2.shape[3])
        target_3 = (agg_3.shape[2], agg_3.shape[3])
        target_4 = (agg_4.shape[2], agg_4.shape[3])

        # Spatial Processing (FG-style): s={4,2,1,1}
        sp1 = F.interpolate(agg_1, scale_factor=4.0, mode='bilinear', align_corners=True)   # 74 -> 296
        sp2 = F.interpolate(agg_2, scale_factor=2.0, mode='bilinear', align_corners=True)   # 74 -> 148
        sp3 = agg_3  # 74
        sp4 = agg_4  # 74

        emb_1 = sp1.permute(0, 2, 3, 1).reshape(B, -1, 256)
        emb_2 = sp2.permute(0, 2, 3, 1).reshape(B, -1, 256)
        emb_3 = sp3.permute(0, 2, 3, 1).reshape(B, -1, 256)
        emb_4 = sp4.permute(0, 2, 3, 1).reshape(B, -1, 256)

        # compute dynamic strides from output/image size and feature map size
        H, W = out_size
        # Cross-sim strides derived from SP sizes; if可控，建議將 out_size pad 為 1184
        stride_1 = max(1, H // sp1.shape[2])
        stride_2 = max(1, H // sp2.shape[2])
        stride_3 = max(1, H // sp3.shape[2])
        stride_4 = max(1, H // sp4.shape[2])

        ca1, *_ = self.cross_attention_1(emb_1, reference_locations, stride_override=stride_1)
        ca2, *_ = self.cross_attention_2(emb_2, reference_locations, stride_override=stride_2)
        ca3, *_ = self.cross_attention_3(emb_3, reference_locations, stride_override=stride_3)
        ca4, *_ = self.cross_attention_4(emb_4, reference_locations, stride_override=stride_4)

        ca1 = ca1.permute(0, 2, 1).reshape(B, -1, sp1.shape[2], sp1.shape[3])
        ca2 = ca2.permute(0, 2, 1).reshape(B, -1, sp2.shape[2], sp2.shape[3])
        ca3 = ca3.permute(0, 2, 1).reshape(B, -1, sp3.shape[2], sp3.shape[3])
        ca4 = ca4.permute(0, 2, 1).reshape(B, -1, sp4.shape[2], sp4.shape[3])

        path4 = self.fusion_4(ca4)
        path3 = self.fusion_3(ca3, path4)
        path2 = self.fusion_2(ca2, path3)
        path1 = self.fusion_1(ca1, path2)

        pred = self.out_conv(path1.permute(0, 2, 3, 1))  # logits
        pred = pred.permute(0, 3, 1, 2)
        pred = F.interpolate(pred, size=out_size, mode='bilinear', align_corners=True)
        pred = pred[:, 0]  # logits [B,H,W]
        return pred, path1, path2, path3, path4

    def _encode_tokens(self, x):
        with torch.no_grad():
            _ = self.dino_model(x)
        # copy out and clear
        l1 = self.activations["1"]
        l2 = self.activations["2"]
        l3 = self.activations["3"]
        l4 = self.activations["4"]
        activations.clear()
        return {"1": l1, "2": l2, "3": l3, "4": l4}

    def _tokens_to_spatial(self, tokens):
        # tokens: (B, T+1, 768) -> (B, 256, S, S)
        x = self.project_readout(tokens)
        x = self.unflatten(x)
        x = self.linear_proj(x)
        return x

    def _tile_concat_2x2(self, f00, f01, f10, f11):
        row0 = torch.cat([f00, f01], dim=3)
        row1 = torch.cat([f10, f11], dim=3)
        return torch.cat([row0, row1], dim=2)

    def forward(self, x, reference_locations):
        B, C, H, W = x.shape
        # pad working resolution to be divisible by 296 for integer strides {4,8,16,16}
        base = 296
        H_pad = ((H + base - 1) // base) * base
        W_pad = ((W + base - 1) // base) * base
        enc = self.encode_image(x)
        agg = enc["agg"]
        pred_pad, path1, path2, path3, path4 = self.forward_with_features(agg, reference_locations, out_size=(H_pad, W_pad))
        # resize logits back to original size
        scores = F.interpolate(pred_pad.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=True)[:, 0]
        layer_1, layer_2, layer_3, layer_4 = enc["layers"]
        # for compatibility with visualization (context_embeddings_*), return agg features
        agg_1, agg_2, agg_3, agg_4 = agg
        return scores, path1, path2, path3, path4, agg_1, agg_2, agg_3, agg_4, layer_1, layer_2, layer_3, layer_4


