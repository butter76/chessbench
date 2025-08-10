"""Post-training script to add U prediction head to existing trained model."""

import os
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')
torch.set_printoptions(profile="full")
from tqdm import tqdm

from searchless_chess.src import config as config_lib
from searchless_chess.src.dataset import load_datasource
from searchless_chess.src.models.transformer import ChessTransformer, TransformerConfig
from searchless_chess.src import data_loader
from searchless_chess.src.optimizer.soap import SOAP


def post_train(
    checkpoint_path: str,
    train_config: config_lib.TrainConfig,
    device: str | None = None,
) -> nn.Module:
    """Post-trains the model with U prediction and returns it."""
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the existing checkpoint
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model_config = checkpoint['model_config']
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    print(f"Checkpoint step: {checkpoint['step']}")
    
    # Create the extended model
    model = ChessTransformer(config=model_config).to(device)

    compiled = False
    if checkpoint.get('compiled', False):
        model = cast(ChessTransformer, torch.compile(model))
        print("Model compiled")
        compiled = True
    
    # Load the existing weights (this will load all except the new U head)
    model_state = checkpoint['model']
    model.load_state_dict(model_state, strict=False)
    print("Loaded existing model weights (excluding new U head)")
    
    # Freeze all existing parameters
    for name, param in model.named_parameters():
        if 'post_' not in name:
            param.requires_grad = False
            print(f"Frozen: {name}")
        else:
            print(f"Unfrozen: {name}")
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Setup optimizer (only for unfrozen parameters)
    optimizer = SOAP(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        precondition_frequency=10,
    )
    
    scaler = GradScaler(device)
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.25,
        total_iters=train_config.num_steps // train_config.ckpt_frequency,
        last_epoch=-1
    )
    
    # Load data
    train_dataloader = load_datasource(train_config.data)
    train_iter = train_dataloader.__iter__()
    
    # Training loop
    num_epochs = train_config.num_steps // train_config.ckpt_frequency
    step = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_u_loss = 0
        total_q_loss = 0
        total_d_loss = 0
        
        pbar = tqdm(total=train_config.ckpt_frequency, desc=f'Post-train Epoch {epoch+1}/{num_epochs}')
        
        for step_in_epoch in range(train_config.ckpt_frequency):
            step += 1
            
            # Get batch: state, legal_actions, U_values
            x, legal_actions, u_values, q_values, d_values = next(train_iter)
            
            x = x.to(torch.long).to(device)
            legal_actions = legal_actions.to(torch.float32).to(device)
            u_values = u_values.to(torch.float32).to(device)
            q_values = q_values.to(torch.float32).to(device)
            d_values = d_values.to(torch.float32).to(device)
            
            target = {
                'legal': legal_actions,
                'U': u_values,
                'Q': q_values,
                'D': d_values,
            }
            
            with autocast(device, dtype=torch.bfloat16):
                # Forward pass
                outputs = model(x)
                
                # Compute only U loss (since other weights are frozen)
                losses = model.post_losses(outputs, target)
                u_loss = losses['U']
                q_loss = losses['Q']
                d_loss = losses['D']

                loss = u_loss + q_loss + d_loss
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            if train_config.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    train_config.max_grad_norm
                )
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update metrics
            total_loss += loss.item()
            total_u_loss += u_loss.item()
            total_q_loss += q_loss.item()
            total_d_loss += d_loss.item()
            
            # Update progress bar
            avg_loss = total_loss / (step_in_epoch + 1)
            avg_u_loss = total_u_loss / (step_in_epoch + 1)
            avg_q_loss = total_q_loss / (step_in_epoch + 1)
            avg_d_loss = total_d_loss / (step_in_epoch + 1)
            
            pbar.set_postfix({
                'avg_loss': f'{avg_loss:.6f}',
                'u_loss': f'{avg_u_loss:.6f}',
                'q_loss': f'{avg_q_loss:.6f}',
                'd_loss': f'{avg_d_loss:.6f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
            pbar.update(1)
        
        pbar.close()
        scheduler.step()
        
        print({
            "epoch": epoch + 1,
            "u_loss": f'{avg_u_loss:.6f}',
            'q_loss': f'{avg_q_loss:.6f}',
            'd_loss': f'{avg_d_loss:.6f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}',
            'step': step,
        })
        
        # Save checkpoint
        post_train_checkpoint = {
            'model': model.state_dict(),
            'compiled': compiled,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'model_config': model_config,
            'step': step,
            'u_loss': avg_u_loss,
            'original_checkpoint': checkpoint_path,
        }
        
        checkpoint_dir = os.path.join(
            os.getcwd(),
            train_config.save_checkpoint_path
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            post_train_checkpoint,
            os.path.join(checkpoint_dir, f'post_train_checkpoint_{step}.pt')
        )
    
    return model


def load_post_trained_model(checkpoint_path: str, device: str | None = None) -> ChessTransformer:
    """Load a post-trained model with U prediction capability."""
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model_config = checkpoint['model_config']
    
    model = ChessTransformer(config=model_config).to(device)
    
    # Handle compilation if the checkpoint was compiled
    if checkpoint.get('compiled', False):
        model = cast(ChessTransformer, torch.compile(model))
        print("Model compiled")
    
    model.load_state_dict(checkpoint['model'])
    
    print(f"Loaded post-trained model from: {checkpoint_path}")
    if 'u_loss' in checkpoint:
        print(f"Final U loss: {checkpoint['u_loss']:.6f}")
    if 'original_checkpoint' in checkpoint:
        print(f"Original checkpoint: {checkpoint['original_checkpoint']}")
    
    return model


def main():
    """Main post-training function."""
    
    # Configuration for post-training
    train_config = config_lib.TrainConfig(
        learning_rate=4e-4,  # Lower learning rate for post-training
        data=config_lib.DataConfig(
            batch_size=2048,  # Smaller batch size
            shuffle=True,
            seed=42,
            worker_count=8,
            num_return_buckets=128,
            policy='lc0_data_with_U',  # Use the new datasource
            split='train',
            dataset_path='../data/child_data/child_data.bag',
        ),
        eval_data=config_lib.DataConfig(
            batch_size=1024,  # Smaller batch size
            shuffle=False,
            seed=42,
            worker_count=8,
            num_return_buckets=128,
            policy='lc0_data_with_U',  # Use the new datasource
            split='test',
            dataset_path='../data/child_data/child_data.bag',
        ),
        compile=True,
        max_grad_norm=1.0,
        num_steps=100_000,  # Fewer steps for post-training
        ckpt_frequency=1_500,
        save_checkpoint_path='../checkpoints/p2-dhl-2x-post-train-u/',
    )
    
    # Path to the original trained checkpoint
    checkpoint_path = '../checkpoints/p2-dhl-2x/checkpoint_300000.pt'  # Adjust as needed
    
    # Post-train the model
    model = post_train(
        checkpoint_path=checkpoint_path,
        train_config=train_config,
    )
    
    print("Post-training completed!")
    return model


if __name__ == "__main__":
    main()
