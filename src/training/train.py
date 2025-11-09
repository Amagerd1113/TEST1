#!/usr/bin/env python3
"""
Main training script for VLA-GR navigation framework.
Handles distributed training, mixed precision, and experiment tracking.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.vla_gr_agent import VLAGRAgent, VLAGRState
from src.datasets.habitat_dataset import HabitatNavigationDataset
from src.training.losses import VLAGRLoss
from src.evaluation.evaluator import VLAGREvaluator

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Main training pipeline for VLA-GR framework."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = self._setup_device()
        self.distributed = config.hardware.distributed.enabled
        
        # Setup distributed training if enabled
        if self.distributed:
            self._setup_distributed()
            
        # Initialize model
        self.model = self._build_model()
        
        # Initialize datasets
        self.train_dataset, self.val_dataset = self._build_datasets()
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self._build_dataloaders()
        
        # Initialize loss function
        self.loss_fn = VLAGRLoss(config)

        # Initialize optimizer and scheduler
        self.optimizer, self.scheduler = self._build_optimizer()

        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler() if config.training.mixed_precision else None

        # Initialize evaluator (optional)
        try:
            self.evaluator = VLAGREvaluator(
                model=self.model,
                config=config
            )
        except Exception as e:
            logger.warning(f"Could not initialize evaluator: {e}")
            self.evaluator = None

        # Initialize metrics tracker
        self.metrics = {}

        # Setup experiment tracking
        self._setup_experiment_tracking()
        
        # Setup checkpointing
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if torch.cuda.is_available() and self.config.hardware.device == "cuda":
            device = torch.device(f"cuda:{self.config.hardware.gpu_ids[0]}")
            torch.cuda.set_device(device)
            logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device
    
    def _setup_distributed(self):
        """Setup distributed training."""
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            gpu = int(os.environ["LOCAL_RANK"])
        else:
            logger.warning("Distributed training requested but environment not setup")
            self.distributed = False
            return
            
        torch.cuda.set_device(gpu)
        dist.init_process_group(
            backend=self.config.hardware.distributed.backend,
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        
        self.rank = rank
        self.world_size = world_size
        logger.info(f"Initialized distributed training: rank {rank}/{world_size}")
    
    def _build_model(self) -> nn.Module:
        """Build and initialize model."""
        model = VLAGRAgent(self.config)
        
        # Load checkpoint if specified
        if self.config.training.resume:
            checkpoint_path = Path(self.config.training.resume)
            if checkpoint_path.exists():
                model.load_checkpoint(str(checkpoint_path))
                logger.info(f"Loaded checkpoint from {checkpoint_path}")
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                
        # Move to device
        model = model.to(self.device)
        
        # Wrap in DDP if distributed
        if self.distributed:
            model = DDP(
                model,
                device_ids=[self.device],
                output_device=self.device,
                find_unused_parameters=True
            )
            
        return model
    
    def _build_datasets(self) -> Tuple:
        """Build training and validation datasets."""
        train_dataset = HabitatNavigationDataset(
            config=self.config,
            split="train",
            transform=self._get_augmentation()
        )
        
        val_dataset = HabitatNavigationDataset(
            config=self.config,
            split="val",
            transform=None
        )
        
        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Val dataset: {len(val_dataset)} samples")
        
        return train_dataset, val_dataset
    
    def _build_dataloaders(self) -> Tuple:
        """Build data loaders."""
        # Training loader
        train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size if self.distributed else 1,
            rank=self.rank if self.distributed else 0,
            shuffle=True
        ) if self.distributed else None
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            prefetch_factor=self.config.training.prefetch_factor,
            persistent_workers=True if self.config.training.num_workers > 0 else False
        )
        
        # Validation loader
        val_sampler = DistributedSampler(
            self.val_dataset,
            num_replicas=self.world_size if self.distributed else 1,
            rank=self.rank if self.distributed else 0,
            shuffle=False
        ) if self.distributed else None
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.training.eval_batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory
        )
        
        return train_loader, val_loader
    
    def _build_optimizer(self) -> Tuple:
        """Build optimizer and scheduler."""
        # Parameter groups with different learning rates
        param_groups = [
            {
                "params": self.model.module.perception.parameters() 
                if hasattr(self.model, 'module') else self.model.perception.parameters(),
                "lr": self.config.training.learning_rate * 0.1,  # Lower LR for perception
                "name": "perception"
            },
            {
                "params": self.model.module.gr_field_manager.parameters()
                if hasattr(self.model, 'module') else self.model.gr_field_manager.parameters(),
                "lr": self.config.training.learning_rate,
                "name": "gr_field"
            },
            {
                "params": self.model.module.path_optimizer.parameters()
                if hasattr(self.model, 'module') else self.model.path_optimizer.parameters(),
                "lr": self.config.training.learning_rate,
                "name": "path_optimizer"
            }
        ]
        
        # Create optimizer
        if self.config.training.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
            
        # Create scheduler
        if self.config.training.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.max_steps,
                eta_min=1e-6
            )
        elif self.config.training.scheduler == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=self.config.training.max_steps
            )
        else:
            scheduler = None
            
        return optimizer, scheduler
    
    def _get_augmentation(self):
        """Get data augmentation transforms."""
        from torchvision import transforms
        
        augmentation = []
        
        if self.config.training.augment.random_crop:
            augmentation.append(transforms.RandomCrop(
                self.config.model.vision.input_size
            ))
            
        if self.config.training.augment.color_jitter:
            augmentation.append(transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ))
            
        if self.config.training.augment.gaussian_noise > 0:
            augmentation.append(transforms.Lambda(
                lambda x: x + torch.randn_like(x) * self.config.training.augment.gaussian_noise
            ))
            
        return transforms.Compose(augmentation) if augmentation else None
    
    def _setup_experiment_tracking(self):
        """Setup experiment tracking with W&B."""
        if self.config.logging.wandb.enabled:
            if not self.distributed or self.rank == 0:
                wandb.init(
                    project=self.config.logging.wandb.project,
                    entity=self.config.logging.wandb.entity,
                    name=self.config.project.name,
                    config=OmegaConf.to_container(self.config),
                    tags=self.config.logging.wandb.tags
                )
                logger.info("Initialized Weights & Biases tracking")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        global_step = 0
        best_metric = float('-inf')
        
        for epoch in range(self.config.training.max_steps // len(self.train_loader)):
            # Set epoch for distributed sampler
            if self.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
                
            # Training epoch
            train_metrics = self.train_epoch(epoch, global_step)
            global_step += len(self.train_loader)
            
            # Validation
            if (epoch + 1) % (self.config.training.eval_every // len(self.train_loader)) == 0:
                val_metrics = self.validate()
                
                # Log metrics
                self.log_metrics(train_metrics, val_metrics, global_step)
                
                # Save checkpoint
                if val_metrics['success_rate'] > best_metric:
                    best_metric = val_metrics['success_rate']
                    self.save_checkpoint(global_step, is_best=True)
                    
            # Regular checkpoint
            if (epoch + 1) % (self.config.training.save_every // len(self.train_loader)) == 0:
                self.save_checkpoint(global_step)
                
            # Check for early stopping
            if global_step >= self.config.training.max_steps:
                break
                
        logger.info("Training completed!")
        
    def train_epoch(self, epoch: int, global_step: int) -> Dict:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {}
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            disable=self.distributed and self.rank != 0
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self.move_batch_to_device(batch)
            
            # Create VLA-GR state
            state = VLAGRState(
                rgb_image=batch['rgb'],
                depth_map=batch['depth'],
                language_instruction=batch['instruction'],
                position=batch.get('position'),
                orientation=batch.get('orientation'),
                velocity=batch.get('velocity')
            )
            
            # Training step
            with autocast(enabled=self.config.training.mixed_precision):
                outputs = self.model(state)
                losses = self.loss_fn(outputs, batch['targets'])
                total_loss = losses['total']
                
            # Backward pass
            if self.scaler:
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip
                    )
                    
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip
                    )
                    
                self.optimizer.step()
                
            self.optimizer.zero_grad()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
                
            # Update metrics
            for key, value in losses.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value.item())
                
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Log to W&B
            if self.config.logging.wandb.enabled and not self.distributed or self.rank == 0:
                if (global_step + batch_idx) % 100 == 0:
                    wandb.log({
                        'train/loss': total_loss.item(),
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'step': global_step + batch_idx
                    })
                    
            # Clear cache periodically
            if batch_idx % self.config.hardware.memory.empty_cache_every == 0:
                torch.cuda.empty_cache()
                
        # Aggregate metrics
        for key in epoch_metrics:
            epoch_metrics[key] = np.mean(epoch_metrics[key])
            
        return epoch_metrics
    
    def validate(self) -> Dict:
        """Run validation."""
        self.model.eval()
        val_metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(
                self.val_loader,
                desc="Validation",
                disable=self.distributed and self.rank != 0
            ):
                batch = self.move_batch_to_device(batch)
                
                # Create state
                state = VLAGRState(
                    rgb_image=batch['rgb'],
                    depth_map=batch['depth'],
                    language_instruction=batch['instruction'],
                    position=batch.get('position'),
                    orientation=batch.get('orientation'),
                    velocity=batch.get('velocity')
                )
                
                # Forward pass
                with autocast(enabled=self.config.training.mixed_precision):
                    outputs = self.model(state, deterministic=True)
                    
                # Compute metrics
                metrics = self.evaluator.evaluate_batch(outputs, batch['targets'])
                
                # Accumulate metrics
                for key, value in metrics.items():
                    if key not in val_metrics:
                        val_metrics[key] = []
                    val_metrics[key].append(value)
                    
        # Aggregate metrics
        for key in val_metrics:
            val_metrics[key] = np.mean(val_metrics[key])
            
        # Synchronize metrics across processes if distributed
        if self.distributed:
            val_metrics = self.sync_metrics(val_metrics)
            
        return val_metrics
    
    def move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to compute device."""
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        return batch
    
    def sync_metrics(self, metrics: Dict) -> Dict:
        """Synchronize metrics across distributed processes."""
        if not self.distributed:
            return metrics
            
        synced_metrics = {}
        for key, value in metrics.items():
            tensor = torch.tensor(value, device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            synced_metrics[key] = tensor.item() / self.world_size
            
        return synced_metrics
    
    def log_metrics(self, train_metrics: Dict, val_metrics: Dict, step: int):
        """Log metrics to console and W&B."""
        # Console logging
        logger.info(f"Step {step}:")
        logger.info(f"  Train Loss: {train_metrics.get('total', 0):.4f}")
        logger.info(f"  Val Success Rate: {val_metrics.get('success_rate', 0):.2%}")
        logger.info(f"  Val SPL: {val_metrics.get('spl', 0):.3f}")
        
        # W&B logging
        if self.config.logging.wandb.enabled and not self.distributed or self.rank == 0:
            wandb.log({
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()},
                'step': step
            })
            
    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint."""
        if self.distributed and self.rank != 0:
            return
            
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.module.state_dict() 
            if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': OmegaConf.to_container(self.config)
        }
        
        # Save checkpoint
        checkpoint_name = 'best.pt' if is_best else f'checkpoint_{step}.pt'
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Keep only recent checkpoints
        if not is_best:
            self.cleanup_checkpoints()
            
    def cleanup_checkpoints(self, keep_last: int = 5):
        """Remove old checkpoints."""
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_*.pt'))
        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint}")


@hydra.main(config_path="../", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """Main training entry point."""
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.console.get('level', 'INFO')),
        format=cfg.logging.console.format
    )
    
    # Print config
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Create training pipeline
    pipeline = TrainingPipeline(cfg)
    
    # Start training
    pipeline.train()
    
    # Cleanup
    if cfg.logging.wandb.enabled:
        wandb.finish()
        
    if cfg.hardware.distributed.enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
