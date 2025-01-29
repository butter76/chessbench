"""PyTorch implementation of the training algorithm for action-value prediction."""

import copy
import os
from typing import cast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
torch.set_default_dtype(torch.bfloat16)
torch.set_printoptions(profile="full")
from tqdm import tqdm

from searchless_chess.src import config as config_lib
from searchless_chess.src import data_loader
from searchless_chess.src import tokenizer
from searchless_chess.src import utils

from searchless_chess.src.dataset import ChessDataset
from searchless_chess.src.models.transformer import TransformerConfig, ChessTransformer



def train(
    train_config: config_lib.TrainConfig,
    model_config: TransformerConfig,
    device: str | None = None,
) -> nn.Module:
    """Trains the model and returns it."""

    # Setup data
    train_dataset = ChessDataset(train_config.data)
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=train_config.data.worker_count,
    )
    val_dataset = ChessDataset(train_config.eval_data)
    val_dataloader = DataLoader(
        val_dataset,
        num_workers=train_config.data.worker_count,
    )

    # In the train function, modify the training loop:
    num_epochs = train_config.num_steps // train_config.ckpt_frequency
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load from checkpoint if it exists
    checkpoint_path = train_config.checkpoint_path
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model_config =  checkpoint['model_config']
        # Create model that matches the checkpoint
        model = ChessTransformer(
            config=model_config,
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from checkpoint: {checkpoint_path}")
    else:
        # Initialize model
        model = ChessTransformer(model_config).to(device)


    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )

    # After creating the optimizer, add the scheduler:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=train_config.ckpt_frequency * 2,  # First restart cycle length
        T_mult=2,  # Each cycle gets twice as long
        eta_min=train_config.learning_rate / 100  # Minimum learning rate
    )
    
    # Training loop
    step = 0
    for epoch in range(num_epochs):
        model.train()
        metrics = {}
        total_loss = 0
        pbar = tqdm(total=train_config.ckpt_frequency, desc=f'Epoch {epoch+1}/{num_epochs}')
        for step_in_epoch in range(train_config.ckpt_frequency):
            step += 1

            x, win_prob = next(iter(train_dataloader))
                
            x = x.to(device).squeeze(0)
            win_prob = win_prob.to(device).squeeze(0).unsqueeze(-1)

            target = {
                'value_head': win_prob,
            }
            
            # Forward pass
            value = model(x)


            
            # Compute loss
            losses = model.losses(value, target)
            loss = cast(torch.Tensor, sum(losses.values())) 

            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            if train_config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    train_config.max_grad_norm
                )
            optimizer.step()
            scheduler.step()

            # Update metrics
            metrics = {name: loss.item() + metrics.get(name, 0) for name, loss in losses.items()}
            total_loss += loss.item()

            # Update progress bar
            avg_loss = total_loss / (step_in_epoch + 1)
            metrics_loss = {name: loss / (step_in_epoch + 1) for name, loss in metrics.items()}
            
            pbar.set_postfix({
                'avg_loss': f'{avg_loss:.4f}',
                **{f'{k}': f'{v:.4f}' for k,v in metrics_loss.items()},
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
            pbar.update(1)
                
        pbar.close()

        # Evaluate on validation set
        model.eval()

        val_metrics = {}
        val_loss = 0
        val_steps = cast(int, train_config.eval_data.num_records) // train_config.eval_data.batch_size
        with torch.inference_mode():
            val_pbar = tqdm(total=val_steps, desc=f'Epoch {epoch+1}/{num_epochs}')
            for step_in_epoch in range(cast(int, val_steps)):
                x, win_prob = next(iter(val_dataloader))

                x = x.to(device).squeeze(0)
                win_prob = win_prob.to(device).squeeze(0).unsqueeze(-1)

                target = {
                    'value_head': win_prob,
                }

                # Forward pass
                value = model(x)

                # Compute loss
                losses = model.losses(value, target)
                loss = cast(torch.Tensor, sum(losses.values())) 

                # Update totals
                val_metrics = {name: loss.item() + val_metrics.get(name, 0) for name, loss in losses.items()}
                val_loss += loss.item()

                # Update progress bar
                avg_val_loss = val_loss / (step_in_epoch + 1)
                val_metrics_loss = {name: loss / (step_in_epoch + 1) for name, loss in val_metrics.items()}
                val_pbar.set_postfix({
                    'avg_val_loss': f'{avg_val_loss:.4f}',
                    **{f'{k}': f'{v:.4f}' for k,v in val_metrics_loss.items()},
                })

                val_pbar.update(1)

        val_pbar.close()

        avg_val_loss = val_loss / val_steps
        val_metrics_loss = {name: loss / val_steps for name, loss in val_metrics.items()}
        # Checkpointing
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_config': model_config,
            'step': step,
            "val_loss": avg_val_loss,
            **{f'val_{k}': f'{v:.4f}' for k,v in val_metrics_loss.items()},
        }
        checkpoint_dir = os.path.join(
            os.getcwd(),
            train_config.save_checkpoint_path
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            checkpoint,
            os.path.join(checkpoint_dir, f'checkpoint_{step}.pt')
        )
    
    return model


def main():
    """Main training function."""
    # Set constants
    num_return_buckets = 128
    policy = 'state_value'
    
    # Create model config
    model_config = TransformerConfig(
        embedding_dim=256,
        num_layers=8,
        num_heads=8,
        widening_factor=3,
        dropout=0.1,
    )
    
    # Create training config
    train_config = config_lib.TrainConfig(
        learning_rate=1e-4,
        data=config_lib.DataConfig(
            batch_size=512,
            shuffle=True,
            worker_count=0,  # 0 disables multiprocessing
            num_return_buckets=num_return_buckets,
            policy=policy,
            split='train',
        ),
        eval_data=config_lib.DataConfig(
            batch_size=512,
            shuffle=False,
            worker_count=0,  # 0 disables multiprocessing
            num_return_buckets=num_return_buckets,
            policy=policy,
            split='test',
            num_records=62000,
        ),
        log_frequency=1,
        num_steps=2000,
        ckpt_frequency=100,
        save_frequency=100,
        save_checkpoint_path='../checkpoints/local/'
    )
    
    # Train model
    model = train(
        train_config=train_config,
        model_config=model_config,
    )
    
    print("Training completed!")
    return model


if __name__ == "__main__":
    main()