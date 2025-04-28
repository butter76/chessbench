"""PyTorch implementation of the training algorithm for action-value prediction."""

from itertools import cycle
import os
from typing import Any, cast

import chess
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
torch.set_default_dtype(torch.float32)
torch.set_printoptions(profile="full")
from tqdm import tqdm

from searchless_chess.src import config as config_lib
from searchless_chess.src.engines.my_engine import MoveSelectionStrategy, MyTransformerEngine
from searchless_chess.src.puzzles import evaluate_puzzle_from_pandas_row
from searchless_chess.src.dataset import load_datasource
from searchless_chess.src.models.transformer import TransformerConfig, ChessTransformer


def train(
    train_config: config_lib.TrainConfig,
    model_config: TransformerConfig,
    device: str | None = None,
) -> nn.Module:
    """Trains the model and returns it."""

    train_dataloader = load_datasource(train_config.data)
    val_dataloader = load_datasource(train_config.eval_data)

    # In the train function, modify the training loop:
    num_epochs = train_config.num_steps // train_config.ckpt_frequency
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load from checkpoint if it exists
    step = 0
    checkpoint_path = train_config.checkpoint_path
    checkpoint = None
    compiled = False
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model_config =  checkpoint['model_config']
        step = checkpoint['step']

        # Create model that matches the checkpoint
        model = ChessTransformer(
            config=model_config,
        ).to(device)

        if checkpoint['compiled']:
            model = cast(ChessTransformer, torch.compile(model))
            compiled = True

        model.load_state_dict(checkpoint['model'])
        print(f"Loaded model from checkpoint: {checkpoint_path}")

    else:
        # Initialize model
        model = ChessTransformer(model_config)
        if train_config.compile:
            model = cast(ChessTransformer, torch.compile(model))
            compiled = True
        model = model.to(device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )

    if checkpoint is not None and 'optimizer' in checkpoint:
        print("Loading Optimizer from checkpoint...")
        optimizer.load_state_dict(checkpoint['optimizer'])


    scaler = GradScaler(device)
    if checkpoint is not None and 'scaler' in checkpoint:
        print("Loading Scaler from checkpoint...")
        scaler.load_state_dict(checkpoint['scaler'])

    # # After creating the optimizer, add the scheduler:
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=train_config.ckpt_frequency * 2,  # First restart cycle length
    #     T_mult=2,  # Each cycle gets twice as long
    #     eta_min=train_config.learning_rate / 100  # Minimum learning rate
    # )

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.25,  # 4e-4 -> 1e-4
        total_iters=100,  # Number of epochs for the decay
        last_epoch=-1
    )

    if checkpoint is not None and 'scheduler' in checkpoint:
        print("Loading Scheduler from checkpoint...")
        scheduler.load_state_dict(checkpoint['scheduler'])

    train_iter = train_dataloader.__iter__()



    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        metrics = {}
        total_loss = 0
        avg_loss = 0
        metrics_loss = {}
        pbar = tqdm(total=train_config.ckpt_frequency, desc=f'Epoch {epoch+1}/{num_epochs}')
        for step_in_epoch in range(train_config.ckpt_frequency):
            step += 1

            x, legal_actions, avs, hl, value_prob, policy, weights = next(train_iter)
                
            x = x.to(torch.long).to(device)
            legal_actions = legal_actions.to(torch.float32).to(device)
            avs = avs.to(torch.float32).to(device)
            hl = hl.to(torch.float32).to(device)
            value_prob = value_prob.to(torch.float32).to(device)
            policy = policy.to(torch.float32).to(device)
            weights = weights.to(torch.float32).to(device)

            target = {
                'self': x,
                'legal': legal_actions,
                'avs': avs,
                'hl': hl,
                'value': value_prob,
                'policy': policy,
                'weights': weights,
            }
            
            with autocast(device, dtype=torch.bfloat16):
                # Forward pass
                value = model(x)
                
                # Compute loss
                losses = model.losses(value, target)
                loss = cast(torch.Tensor, sum(v for k, v in losses.items() if k not in ['value', 'avs', 'avs2', 'avs_accuracy', 'avs2_accuracy', 'policy_accuracy']))

            
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

            # Update metrics
            metrics = {name: loss.item() + metrics.get(name, 0) for name, loss in losses.items()}
            total_loss += loss.item()

            # Update progress bar
            avg_loss = total_loss / (step_in_epoch + 1)
            metrics_loss = {name: loss / (step_in_epoch + 1) for name, loss in metrics.items()}
            
            pbar.set_postfix({
                'avg_loss': f'{avg_loss:.5f}',
                **{f'{k}': f'{v:.5f}' for k,v in metrics_loss.items()},
                'lr': f'{scheduler.get_last_lr()[0]:.5f}'
            })
            pbar.update(1)
                
        pbar.close()

        # Evaluate on validation set
        model.eval()

        val_metrics = {}
        val_loss = 0
        val_steps = cast(int, train_config.eval_data.num_records) // train_config.eval_data.batch_size
        val_iter = iter(val_dataloader)
        with torch.inference_mode():
            val_pbar = tqdm(total=val_steps, desc=f'Epoch {epoch+1}/{num_epochs}')
            for step_in_epoch in range(cast(int, val_steps)):
                x, legal_actions, avs, hl, value_prob, policy, weights = next(val_iter)
                
                x = x.to(torch.long).to(device)
                legal_actions = legal_actions.to(torch.float32).to(device)
                avs = avs.to(torch.float32).to(device)
                hl = hl.to(torch.float32).to(device)
                value_prob = value_prob.to(torch.float32).to(device)
                policy = policy.to(torch.float32).to(device)
                weights = weights.to(torch.float32).to(device)

                target = {
                    'self': x,
                    'legal': legal_actions,
                    'avs': avs,
                    'hl': hl,
                    'value': value_prob,
                    'policy': policy,
                    'weights': weights,
                }
                
                with torch.inference_mode(), autocast(device, dtype=torch.bfloat16):
                    value = model(x)

                # Compute loss
                losses = model.losses(value, target)
                loss = cast(torch.Tensor, sum(v for k, v in losses.items() if k not in ['value', 'avs', 'avs2', 'avs_accuracy', 'avs2_accuracy', 'policy_accuracy']))
                # Update totals
                val_metrics = {name: loss.item() + val_metrics.get(name, 0) for name, loss in losses.items()}
                val_loss += loss.item()

                # Update progress bar
                avg_val_loss = val_loss / (step_in_epoch + 1)
                val_metrics_loss = {name: loss / (step_in_epoch + 1) for name, loss in val_metrics.items()}
                val_pbar.set_postfix({
                    'avg_val_loss': f'{avg_val_loss:.5f}',
                    **{f'{k}': f'{v:.5f}' for k,v in val_metrics_loss.items()},
                })

                val_pbar.update(1)

        val_pbar.close()

        avg_val_loss = val_loss / val_steps
        val_metrics_loss = {name: loss / val_steps for name, loss in val_metrics.items()}

        scheduler.step()
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
        torch.save(
            checkpoint,
            os.path.join(checkpoint_dir, f'checkpoint_{step}.pt')
        )
    
    return model


def main():
    """Main training function."""
    # Set constants
    num_return_buckets = 128
    policy = 'action_values'
    
    # Create model config
    model_config = TransformerConfig(
        embedding_dim=256,
        num_layers=16,
        num_heads=16,
        widening_factor=3,
        dropout=0,
    )
    
    # Create training config
    train_config = config_lib.TrainConfig(
        learning_rate=4e-4,
        data=config_lib.DataConfig(
            batch_size=2048,
            shuffle=True,
            seed=42143242,
            worker_count=16,  # 0 disables multiprocessing
            num_return_buckets=num_return_buckets,
            policy=policy,
            split='train',
            dataset_path='../data/output/new@24.bag',
        ),
        eval_data=config_lib.DataConfig(
            batch_size=2048,
            shuffle=False,
            worker_count=16,  # 0 disables multiprocessing
            num_return_buckets=num_return_buckets,
            policy=policy,
            split='test',
            dataset_path='../data/output/validation.bag',
            num_records=1_000_000
        ),
        compile=True,
        max_grad_norm=1.0,
        log_frequency=1,
        num_steps=100 * 1000 * 3,
        ckpt_frequency=1000 * 3,
        save_frequency=1000 * 3,
        save_checkpoint_path='../checkpoints/p1-standard/',
    )
    
    # Train model
    model = train(
        train_config=train_config,
        model_config=model_config,
    )

    puzzles_path = os.path.join(
        os.getcwd(),
        '../data/puzzles.csv',
    )
    puzzles = pd.read_csv(puzzles_path, nrows=10000)
    for strategy in [MoveSelectionStrategy.VALUE, MoveSelectionStrategy.AVS, MoveSelectionStrategy.AVS2, MoveSelectionStrategy.POLICY, MoveSelectionStrategy.OPT_POLICY_SPLIT]:
        engine = MyTransformerEngine(
            f'{train_config.save_checkpoint_path}checkpoint_{train_config.num_steps}.pt',
            chess.engine.Limit(nodes=1),
            strategy=strategy,
        )
        with open(f'puzzles-{strategy}.txt', 'w') as f:
            num_correct = 0
            for puzzle_id, puzzle in puzzles.iterrows():
                correct = evaluate_puzzle_from_pandas_row(
                    puzzle=puzzle,
                    engine=engine,
                )
                num_correct += correct
                f.write(str({'puzzle_id': puzzle_id, 'correct': correct, 'rating': puzzle['Rating']}) + '\n')
            print(f'{strategy}: {num_correct / len(puzzles):.2%}')

    
    print("Training completed!")


    return model


if __name__ == "__main__":
    main()