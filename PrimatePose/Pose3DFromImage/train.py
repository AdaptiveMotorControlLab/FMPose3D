import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import time
from tqdm import tqdm
import logging
from datetime import datetime
import random
import shutil
import importlib.util

wandb = None
USE_WANDB = False
GLOBAL_STEP = 0

# Import our modules
from dataset import PrimateDataset, create_data_loaders
from models.model_graph import Pose3D
from loss import CombinedLoss


def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_loss': best_loss,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['best_loss']


def validate_model(model, val_loader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    loss_components = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
            # Move data to device
            images = batch['image'].to(device)
            pose_2d = batch['pose_2d'].to(device)
            
            # Create valid mask from visibility
            valid_mask = pose_2d[:, :, 2] > 0  # visibility > 0
            
            # Forward pass
            predictions = model(images, pose_2d, valid_mask)
            
            # Prepare targets
            targets = {'pose_2d': pose_2d}
            
            # Compute loss
            loss_dict = criterion(predictions, targets, valid_mask)
            
            # Accumulate losses
            total_loss += loss_dict['total'].item() * images.shape[0]
            total_samples += images.shape[0]
            
            # Accumulate loss components
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value.item() * images.shape[0]
    
    # Average losses
    avg_loss = total_loss / total_samples
    for key in loss_components:
        loss_components[key] /= total_samples
    
    return avg_loss, loss_components


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    global GLOBAL_STEP
    model.train()
    total_loss = 0.0
    total_samples = 0
    loss_components = {}
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        images = batch['image'].to(device)
        pose_2d = batch['pose_2d'].to(device)
        
        # Create valid mask from visibility
        valid_mask = pose_2d[:, :, 2] > 0  # visibility > 0
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(images, pose_2d, valid_mask)
        
        # Prepare targets
        targets = {'pose_2d': pose_2d}
        
        # Compute loss
        loss_dict = criterion(predictions, targets, valid_mask)
        loss = loss_dict['total']
        
        if loss.item() > 10000 and epoch >= 1:
            print(f"Loss is too high: {loss.item()}")
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item() * images.shape[0]
        total_samples += images.shape[0]
        
        # Accumulate loss components
        for key, value in loss_dict.items():
            if key not in loss_components:
                loss_components[key] = 0.0
            loss_components[key] += value.item() * images.shape[0]
        
        # Update progress bar
        avg_loss = total_loss / total_samples
        pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})
        
        # Log to Weights & Biases (every N batches)
        # if USE_WANDB and (batch_idx % 1 == 0):
        #     GLOBAL_STEP += 1
        #     wandb.log({'train/batch/loss': loss.item()}, step=GLOBAL_STEP)
    
    # Average losses
    print(f"Total samples: {total_samples}")
    avg_loss = total_loss / total_samples
    for key in loss_components:
        loss_components[key] /= total_samples
    
    return avg_loss, loss_components


def copy_source_files(experiment_dir):
    """
    Copy source files to experiment directory for reproducibility
    Args:
        experiment_dir: Path to experiment directory
    """
    # Create source code backup directory
    source_dir = os.path.join(experiment_dir, 'source_code')
    os.makedirs(source_dir, exist_ok=True)
    
    # Files to copy (keep minimal, per request)
    files_to_copy = [
        'train.py',
        'train.sh',
        'test.py',
        'run_test.sh',
        'models/model.py',
        'dataset.py',
    ]
    
    for file_path in files_to_copy:
        if os.path.exists(file_path):
            # Create subdirectories if needed
            dest_path = os.path.join(source_dir, file_path)
            dest_dir = os.path.dirname(dest_path)
            os.makedirs(dest_dir, exist_ok=True)
            
            # Copy file
            shutil.copy2(file_path, dest_path)
            print(f"Copied {file_path} to {dest_path}")
        else:
            print(f"Warning: {file_path} not found, skipping")


def create_experiment_dir(train_json_path, dataset_name=None, base_dir="experiments"):
    """
    Create a timestamped experiment directory
    Args:
        train_json_path: Path to training JSON file
        dataset_name: Custom dataset name (if None, extracts from JSON path)
        base_dir: Base directory for experiments
    Returns:
        str: Path to the created experiment directory
    """
    # Use provided dataset name or extract from JSON path
    if dataset_name is None:
        json_filename = os.path.basename(train_json_path)
        dataset_name = json_filename.replace('.json', '').replace('_train', '')
    
    # Create timestamp string (YYYYMMDD_HHMM)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Create experiment directory name
    experiment_name = f"{dataset_name}_{timestamp}"
    experiment_dir = os.path.join(base_dir, experiment_name)
    
    # Create directory
    os.makedirs(experiment_dir, exist_ok=True)
    
    return experiment_dir


def main():
    global wandb, USE_WANDB
    parser = argparse.ArgumentParser(description='Train 3D Pose Estimation Model')
    
    # Data arguments
    parser.add_argument('--train_json', type=str,
                       default='/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.3/splitted_train_datasets/ap10k_train.json',
                       help='Path to training JSON file')
    parser.add_argument('--val_json', type=str,
                       default='/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.3/splitted_test_datasets/ap10k_test.json',
                       help='Path to validation JSON file')
    parser.add_argument('--image_root', type=str, default='/home/ti_wang/data/tiwang/v8_coco/images',
                       help='Root directory for images')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                       help='Image size [width, height]')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet18', 'resnet34', 'resnet50', 'vit_s16_dino', 'vit_small_patch16_224.dino'],
                       help='Backbone architecture (torchvision ResNets or ViT DINO: vit_s16_dino)')
    parser.add_argument('--num_keypoints', type=int, default=37,
                       help='Number of keypoints')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Loss weights
    parser.add_argument('--reprojection_weight', type=float, default=1.0,
                       help='Weight for reprojection loss')
    parser.add_argument('--bone_length_weight', type=float, default=0.1,
                       help='Weight for bone length loss')
    
    # Training setup
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='Dataset name for experiment folder (if not provided, extracts from JSON filename)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for checkpoints and logs. If not provided, will create timestamped directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save_freq', type=int, default=1,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--val_freq', type=int, default=1,
                       help='Validate every N epochs')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device ID to use (e.g., 0, 1, 2). If not specified, uses automatic detection')
    # Reproducibility
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for reproducibility (default: 1)')
    # Weights & Biases
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project_name', type=str, default='Pose3D',
                        help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='W&B run name (defaults to experiment folder name)')
    parser.add_argument('--wandb_group', type=str, default=None,
                        help='W&B group name')
    
    args = parser.parse_args()
    
    # Create timestamped output directory if not provided
    if args.output_dir is None:
        args.output_dir = create_experiment_dir(args.train_json, args.dataset_name)
        print(f"Created experiment directory: {args.output_dir}")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Copy source files for reproducibility
    print("Copying source files for reproducibility...")
    copy_source_files(args.output_dir)
    
    # Save training arguments for reproducibility
    args_file = os.path.join(args.output_dir, 'training_args.txt')
    with open(args_file, 'w') as f:
        f.write("Training Arguments:\n")
        f.write("=" * 50 + "\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write("Command line:\n")
        f.write(" ".join(sys.argv) + "\n")
    print(f"Saved training arguments to: {args_file}")
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info(f"Experiment directory: {args.output_dir}")
    logger.info(f"Starting training with arguments: {vars(args)}")
    # Set random seed for reproducibility
    if args.seed <= 0:
        args.seed = random.randint(0, 2**32 - 1)
        logger.info(f"No seed provided. Generated random seed: {args.seed}")
    set_random_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")
    
    # Setup device
    if args.gpu is not None:
        # User specified a GPU
        if not torch.cuda.is_available():
            logger.error("CUDA is not available, but GPU was specified!")
            sys.exit(1)
        
        if args.gpu >= torch.cuda.device_count():
            logger.error(f"GPU {args.gpu} is not available! Available GPUs: 0-{torch.cuda.device_count()-1}")
            sys.exit(1)
        
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
        logger.info(f"Using specified GPU: {args.gpu} ({torch.cuda.get_device_name(args.gpu)})")
    else:
        # Automatic device selection
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using device: {device} ({gpu_name})")
        else:
            logger.info(f"Using device: {device}")
    
    # Display GPU memory info if using CUDA
    if device.type == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        logger.info(f"GPU memory: {gpu_memory:.1f} GB")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(
        train_json=args.train_json,
        test_json=args.val_json,
        image_root=args.image_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=tuple(args.image_size)
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = Pose3D(
        num_keypoints=args.num_keypoints,
        backbone=args.backbone,
        pretrained=True,
        image_size=tuple(args.image_size)
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    criterion = CombinedLoss(
        reprojection_weight=args.reprojection_weight,
        bone_length_weight=args.bone_length_weight,
        image_size=tuple(args.image_size)
    )
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
        
    # Optional: initialize Weights & Biases
    if args.wandb:
        spec = importlib.util.find_spec("wandb")
        if spec is None:
            logger.error("--wandb specified but the 'wandb' package is not installed. Please install with: pip install wandb")
            sys.exit(1)
        import wandb as _wandb
        global wandb, USE_WANDB
        wandb = _wandb
        run_name = args.wandb_run_name or os.path.basename(args.output_dir)
        wandb.init(
            project=args.wandb_project_name,
            name=run_name,
            config=vars(args),
            dir=args.output_dir,
            group=args.wandb_group,
            resume='never'
        )
        USE_WANDB = True
        logger.info('Weights & Biases logging enabled')
        # Define metric step mapping
        # wandb.define_metric('global_step')
        # wandb.define_metric('train/batch/*', step_metric='global_step')
        # wandb.define_metric('train/*', step_metric='epoch')
        # wandb.define_metric('val/*', step_metric='epoch')
        # # Metric summaries
        # wandb.define_metric('train/loss', summary='last')
        # wandb.define_metric('val/loss', summary='min')
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, scheduler, args.resume
        )
        start_epoch += 1
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        # Train for one epoch
        train_loss, train_components = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        if USE_WANDB:
            wandb.log({'train/lr': current_lr}, step=epoch)
        
        logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, LR = {current_lr:.6f}")
        # Log epoch-average training loss explicitly
        if USE_WANDB:
            wandb.log({'train/epoch_avg_loss': train_loss}, step=epoch)
        
        # Validation
        if epoch % args.val_freq == 0:
            val_loss, val_components = validate_model(
                model, val_loader, criterion, device, epoch
            )
            
            logger.info(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
            wandb.log({'val/avg_loss': val_loss}, step=epoch)

            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(args.output_dir, 'best_model.pth')
                save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, best_model_path)
                logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
                # if USE_WANDB:
                    # wandb.run.summary['best_val_loss'] = best_val_loss
        
        # Save checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, best_val_loss, final_model_path)
    logger.info(f"Final model saved: {final_model_path}")
    
    # Close tensorboard writer
    # Removed writer.close()
    # Close Weights & Biases run if enabled
    if USE_WANDB:
        wandb.finish()
    logger.info("Training completed!")


if __name__ == "__main__":
    main() 