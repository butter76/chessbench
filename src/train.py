"""PyTorch implementation of the training algorithm for action-value prediction."""

import math
import os
from typing import cast

import chess
import pandas as pd
import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')
torch.set_printoptions(profile="full")
from tqdm import tqdm

from searchless_chess.src import config as config_lib
from searchless_chess.src.dataset import PrefetchIterator, load_datasource
from searchless_chess.src.engines.my_engine import MoveSelectionStrategy, MyTransformerEngine
from searchless_chess.src.models.transformer import ChessTransformer, TransformerConfig
from searchless_chess.src.optimizer.splus import SPlus
from searchless_chess.src.puzzles import evaluate_puzzle_from_pandas_row


def get_cosine_schedule_with_warmup(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Create a cosine learning rate schedule with warmup.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        min_lr_ratio: Final LR = base_lr * min_lr_ratio

    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return current_step / max(1, warmup_steps)
        else:
            # Cosine decay
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(
    train_config: config_lib.TrainConfig,
    model_config: TransformerConfig,
    device: str | None = None,
    warmup_steps: int = 1000,
    min_lr_ratio: float = 0.1,
) -> nn.Module:
    """Trains the model and returns it."""

    train_dataloader = load_datasource(train_config.data)
    val_dataloader = load_datasource(train_config.eval_data)

    # In the train function, modify the training loop:
    num_epochs = train_config.num_steps // train_config.steps_per_epoch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load from checkpoint if it exists
    step = 0
    checkpoint_path = train_config.checkpoint_path
    checkpoint = None
    compiled = False
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model_config =  checkpoint['model_config']
        step = checkpoint['step']

        # Create model that matches the checkpoint
        model = ChessTransformer(
            config=model_config,
        ).to(device)

        if checkpoint['compiled']:
            model = cast('ChessTransformer', torch.compile(model))
            compiled = True

        model.load_state_dict(checkpoint['model'])
        print(f"Loaded model from checkpoint: {checkpoint_path}")

    else:
        # Initialize model
        model = ChessTransformer(model_config)
        if train_config.compile:
            model = cast('ChessTransformer', torch.compile(model))
            compiled = True
        model = model.to(device)

    # Setup optimizer (SPlus with jax_base.yaml settings)
    optimizer = SPlus(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        b1=0.9,
        b2=0.999,
        ema_rate=0.999,
        inverse_every=100,
        max_dim=10000,
    )

    if checkpoint is not None and 'optimizer' in checkpoint:
        print("Loading Optimizer from checkpoint...")
        optimizer.load_state_dict(checkpoint['optimizer'])

    scaler = GradScaler(device)
    if checkpoint is not None and 'scaler' in checkpoint:
        print("Loading Scaler from checkpoint...")
        scaler.load_state_dict(checkpoint['scaler'])

    # Cosine schedule with warmup (matching jax_base.yaml)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=train_config.num_steps,
        min_lr_ratio=min_lr_ratio,
    )

    if checkpoint is not None and 'scheduler' in checkpoint:
        print("Loading Scheduler from checkpoint...")
        scheduler.load_state_dict(checkpoint['scheduler'])

    # Create iterator with policy type
    policy = train_config.data.policy
    train_iter = PrefetchIterator(train_dataloader, device=device, policy=policy)



    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    # Training loop
    for epoch in range(num_epochs):
        optimizer.train()
        model.train()
        metrics = {}
        total_loss = 0
        avg_loss = 0
        metrics_loss = {}
        pbar = tqdm(total=train_config.steps_per_epoch, desc=f'Epoch {epoch+1}/{num_epochs}')
        for step_in_epoch in range(train_config.steps_per_epoch):
            step += 1

            # New training_bag format: (state, policy, hard_policy, hl, value_prob)
            x, train_policy, hard_policy, hl, value_prob = next(train_iter)

            target = {
                'self': x,
                'hl': hl,
                'value': value_prob,
                'policy': train_policy,
                'hard_policy': hard_policy,
            }

            with autocast(device, dtype=torch.bfloat16):
                # Forward pass
                output = model(x)

                # Compute loss
                losses = model.losses(output, target)
                loss = cast('torch.Tensor', sum(v for k, v in losses.items() if k not in ['value', 'draw']))


            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if train_config.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train_config.max_grad_norm
                )
            scaler.step(optimizer)
            scaler.update()

            # Step the scheduler every step (not every epoch)
            scheduler.step()

            # Update metrics
            metrics = {name: loss_val.item() + metrics.get(name, 0) for name, loss_val in losses.items()}
            total_loss += loss.item()

            # Update progress bar
            avg_loss = total_loss / (step_in_epoch + 1)
            metrics_loss = {name: loss_val / (step_in_epoch + 1) for name, loss_val in metrics.items()}

            pbar.set_postfix({
                'avg_loss': f'{avg_loss:.5f}',
                **{f'{k}': f'{v:.5f}' for k,v in metrics_loss.items()},
                'lr': f'{scheduler.get_last_lr()[0]:.5f}'
            })
            pbar.update(1)

        pbar.close()

        # Evaluate on validation set
        optimizer.eval()
        model.eval()

        val_metrics = {}
        val_loss = 0
        val_steps = cast('int', train_config.eval_data.num_records) // train_config.eval_data.batch_size
        val_iter = PrefetchIterator(val_dataloader, device=device, policy=policy)
        with torch.inference_mode():
            val_pbar = tqdm(total=val_steps, desc=f'Val Epoch {epoch+1}/{num_epochs}')
            for step_in_epoch in range(cast('int', val_steps)):
                # New training_bag format: (state, policy, hard_policy, hl, value_prob)
                x, val_policy, val_hard_policy, val_hl, val_value_prob = next(val_iter)

                target = {
                    'self': x,
                    'hl': val_hl,
                    'value': val_value_prob,
                    'policy': val_policy,
                    'hard_policy': val_hard_policy,
                }

                with torch.inference_mode(), autocast(device, dtype=torch.bfloat16):
                    output = model(x)

                # Compute loss
                losses = model.losses(output, target)
                loss = cast('torch.Tensor', sum(v for v in losses.values()))
                # Update totals
                val_metrics = {name: loss_val.item() + val_metrics.get(name, 0) for name, loss_val in losses.items()}
                val_loss += loss.item()

                # Update progress bar
                avg_val_loss = val_loss / (step_in_epoch + 1)
                val_metrics_loss = {name: loss_val / (step_in_epoch + 1) for name, loss_val in val_metrics.items()}
                val_pbar.set_postfix({
                    'avg_val_loss': f'{avg_val_loss:.5f}',
                    **{f'{k}': f'{v:.5f}' for k,v in val_metrics_loss.items()},
                })

                val_pbar.update(1)

        val_pbar.close()

        avg_val_loss = val_loss / val_steps
        val_metrics_loss = {name: loss_val / val_steps for name, loss_val in val_metrics.items()}
        print({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            **{f'{k}': f'{v:.6f}' for k,v in metrics_loss.items()},
            "val_loss": avg_val_loss,
            **{f'val_{k}': f'{v:.6f}' for k,v in val_metrics_loss.items()},
            'lr': f'{scheduler.get_last_lr()[0]:.5f}',
            'step': step,
        })

        # Checkpointing
        checkpoint = {
            'model': model.state_dict(),
            'compiled': compiled,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'model_config': model_config,
            'step': step,
            "val_loss": avg_val_loss,
            **{f'val_{k}': f'{v:.5f}' for k,v in val_metrics_loss.items()},
        }
        checkpoint_dir = os.path.join(
            os.getcwd(),
            train_config.save_checkpoint_path
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Always save/overwrite the latest checkpoint
        torch.save(
            checkpoint,
            os.path.join(checkpoint_dir, 'checkpoint_last.pt')
        )

        # Save a periodic checkpoint every `save_frequency` epochs
        if (step // train_config.steps_per_epoch) % train_config.save_frequency == 0:
            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, f'checkpoint_{step}.pt')
            )

    return model


def main():
    """Main training function.

    Configuration matches jax_base.yaml:
    - Model: 512 hidden, 16 layers, 32 heads, ff_dim=1536, Smolgen enabled
    - Optimizer: SPlus with lr=0.095
    - Scheduler: Cosine with 1000 warmup steps, min_lr_ratio=0.1
    - Data: training_bag format from ~/training_bag/
    """
    # Set constants
    num_return_buckets = 128
    policy = 'training_bag'

    # Create model config (matching jax_base.yaml)
    model_config = TransformerConfig(
        embedding_dim=512,
        num_layers=16,
        num_heads=32,
        widening_factor=3,  # ff_dim = 512 * 3 = 1536
        dropout=0.0,
        # Smolgen config
        use_smolgen=True,
        smolgen_hidden_channels=32,
        smolgen_hidden_size=256,
        smolgen_gen_size=256,
        use_attention_bias=False,
        # Output heads config
        self_weight=0.1,
        value_weight=0.7,
        policy_weight=1.5,
        hard_policy_temperature=0.25,
        hard_policy_weight=0.1,
        policy_qk_dim=32,
    )

    # Create training config (matching jax_base.yaml)
    train_config = config_lib.TrainConfig(
        learning_rate=0.095,  # From jax_base.yaml
        weight_decay=0.01,
        data=config_lib.DataConfig(
            batch_size=1024,  # From jax_base.yaml
            shuffle=True,
            seed=42,  # From jax_base.yaml
            worker_count=4,  # From jax_base.yaml data.num_workers
            num_return_buckets=num_return_buckets,
            policy=policy,
            split='train',
            # Data paths from jax_base.yaml
            dataset_path='~/training_bag/training-run1-test80-202507*.bag',
        ),
        eval_data=config_lib.DataConfig(
            batch_size=1024,
            shuffle=False,
            worker_count=4,
            num_return_buckets=num_return_buckets,
            policy=policy,
            split='test',
            # Validation data path from jax_base.yaml
            dataset_path='~/training_bag/training-run1-test80-202508*.bag',
            num_records=204_800  # Limit validation samples
        ),
        compile=True,
        max_grad_norm=1.0,  # From jax_base.yaml training.gradient_clip
        num_steps=600_000,  # From jax_base.yaml training.max_steps
        steps_per_epoch=6000,  # From jax_base.yaml training.steps_per_epoch
        save_frequency=1,  # From jax_base.yaml checkpoint.save_every_epochs
        save_checkpoint_path='../checkpoints/catgpt_test/',
    )

    # Train model (with scheduler settings from jax_base.yaml)
    model = train(
        train_config=train_config,
        model_config=model_config,
        warmup_steps=1000,  # From jax_base.yaml scheduler.warmup_steps
        min_lr_ratio=0.1,  # From jax_base.yaml scheduler.min_lr_ratio
    )

    print("Training completed!")

    return model


if __name__ == "__main__":
    main()
