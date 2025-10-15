import os
import sys
import argparse
from omegaconf import OmegaConf



def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments for training the model.')

    # Experiment arguments
    parser.add_argument('--config', type=str, default="./configs/transformer.cfg", help='Path to the config file.')
    parser.add_argument('--exp_name', type=str, default='materialistic_checkpoint_new', 
                        help='Name of the experiment.')
    parser.add_argument('--log_dir', type=str, default='../tensorboard/',
                        help='Path to the save directory.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/',
                        help='Path to the checkpoint directory.')
    parser.add_argument('--real_data', action='store_true', default=False, help="Pass to train with real data.")
    parser.add_argument('--resume', action='store_true', default=False, help="Whether to resume training from a checkpoint.")
    parser.add_argument('--num_gpus', type=int, default=2, help='Number of GPUs to use.')
    parser.add_argument('--precision', type=int, default=32, help='Precision level to use.')
    parser.add_argument('--train_workers', type=int, default=8, help='Number of DataLoader workers for training.')
    parser.add_argument('--val_workers', type=int, default=4, help='Number of DataLoader workers for validation.')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Accumulate gradients for this many batches to save memory.')
    parser.add_argument('--num_queries', type=int, default=1, help='Number of queries per image for multi-query training (set >1 to enable).')
    parser.add_argument('--max_batch_queries', type=int, default=4, help='Max queries processed together to control memory.')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default= '../data/materialistic_synthetic_dataset/',
                        help='Path to the data directory.')
    parser.add_argument('--image_size', type=int, default=1024, help='Size of the images (multiple of 14 recommended for DINOv2).')

    # Model arguments
    parser.add_argument('--model_name', type=str, default='unet', help='Name of the model.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay.')
    parser.add_argument('--print_every', type=int, default=500, help='Print loss every n iterations.')
    # Optional modality flags (also used by datasets)
    parser.add_argument('--use_depth', action='store_true', default=False, help='Whether to use depth maps.')
    parser.add_argument('--use_normal', action='store_true', default=False, help='Whether to use normal maps.')
    parser.add_argument('--use_chroma', action='store_true', default=False, help='Whether to use chroma maps.')
    
    args = parser.parse_args()
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_name)
    cfg = OmegaConf.load(args.config)
    return args, cfg
