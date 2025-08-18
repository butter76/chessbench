"""PyTorch implementation of the training algorithm for action-value prediction."""

import os
import random
from typing import cast

import chess
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')
torch.set_printoptions(profile="full")
from tqdm import tqdm
import numpy as np

from searchless_chess.src import config as config_lib
from searchless_chess.src.engines.my_engine import MoveSelectionStrategy, MyTransformerEngine
from searchless_chess.src.puzzles import evaluate_puzzle_from_pandas_row
from searchless_chess.src.dataset import load_datasource, PrefetchIterator
from searchless_chess.src.models.transformer import TransformerConfig, ChessTransformer
from searchless_chess.src.optimizer.splus import SPlus


def train(
    train_config: config_lib.TrainConfig,
    model_config: TransformerConfig,
    device: str | None = None,
) -> nn.Module:
    """Trains the model and returns it."""

    # Distributed initialization
    distributed = False
    local_rank = 0
    world_size = 1
    rank = 0
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')
        distributed = True
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))

    def is_main_process() -> bool:
        return (not distributed) or rank == 0

    def unwrap_model(m: nn.Module) -> nn.Module:
        return m.module if isinstance(m, nn.parallel.DistributedDataParallel) else m

    # Device and seeds per rank
    if device is None:
        if torch.cuda.is_available():
            if distributed:
                torch.cuda.set_device(local_rank)
                device = f"cuda:{local_rank}"
            else:
                device = "cuda"
        else:
            device = "cpu"

    base_seed = train_config.data.seed if hasattr(train_config, 'data') else 0
    rank_seed = base_seed + rank
    torch.manual_seed(rank_seed)
    np.random.seed(rank_seed)
    random.seed(rank_seed)

    # Data loaders (train sharded per-rank; eval on main process only)
    train_dataloader = load_datasource(
        train_config.data,
        world_size=world_size if distributed else None,
        rank=rank if distributed else None,
    )
    if is_main_process():
        val_dataloader = load_datasource(train_config.eval_data)
    else:
        val_dataloader = None

    # In the train function, modify the training loop:
    num_epochs = train_config.num_steps // train_config.steps_per_epoch
    
    # device already determined above

    # Load from checkpoint if it exists
    step = 0
    checkpoint_path = train_config.checkpoint_path
    checkpoint = None
    compiled = False
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        map_location = {'cuda': f'cuda:{local_rank}'} if device.startswith('cuda') else 'cpu'
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=map_location)
        model_config =  checkpoint['model_config']
        step = checkpoint['step']

        # Create model that matches the checkpoint
        base_model = ChessTransformer(
            config=model_config,
        ).to(device)

        if checkpoint['compiled']:
            base_model = cast(ChessTransformer, torch.compile(base_model))
            compiled = True

        # Wrap with DDP if distributed
        model: ChessTransformer | nn.Module
        if distributed:
            model = nn.parallel.DistributedDataParallel(
                base_model,
                device_ids=[local_rank] if device.startswith('cuda') else None,
                output_device=local_rank if device.startswith('cuda') else None,
            )
            model.module.load_state_dict(checkpoint['model'])  # type: ignore[attr-defined]
        else:
            model = base_model
            model.load_state_dict(checkpoint['model'])
        if is_main_process():
            print(f"Loaded model from checkpoint: {checkpoint_path}")

    else:
        # Initialize model
        base_model = ChessTransformer(model_config)
        if train_config.compile:
            base_model = cast(ChessTransformer, torch.compile(base_model))
            compiled = True
        base_model = base_model.to(device)
        if distributed:
            model = nn.parallel.DistributedDataParallel(
                base_model,
                device_ids=[local_rank] if device.startswith('cuda') else None,
                output_device=local_rank if device.startswith('cuda') else None,
            )
        else:
            model = base_model

    # Setup optimizer
    optimizer = SPlus(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    if checkpoint is not None and 'optimizer' in checkpoint:
        if is_main_process():
            print("Loading Optimizer from checkpoint...")
        optimizer.load_state_dict(checkpoint['optimizer'])


    scaler = GradScaler(device)
    if checkpoint is not None and 'scaler' in checkpoint:
        if is_main_process():
            print("Loading Scaler from checkpoint...")
        scaler.load_state_dict(checkpoint['scaler'])

    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.77,
        total_iters=90,
    )

    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=80,
        eta_min=0.01,
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[90],
    )

    if checkpoint is not None and 'scheduler' in checkpoint:
        if is_main_process():
            print("Loading Scheduler from checkpoint...")
        scheduler.load_state_dict(checkpoint['scheduler'])

    train_iter = PrefetchIterator(train_dataloader, device=device)



    total_params = sum(p.numel() for p in model.parameters())
    if is_main_process():
        print(f"Total number of parameters: {total_params:,}")
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.train()
        model.train()
        metrics = {}
        total_loss = 0
        avg_loss = 0
        metrics_loss = {}
        pbar = tqdm(total=train_config.steps_per_epoch, desc=f'Epoch {epoch+1}/{num_epochs}') if is_main_process() else None
        for step_in_epoch in range(train_config.steps_per_epoch):
            step += 1

            x, legal_actions, policy, soft_policy, hard_policy, hardest_policy, hl, dhl, wdl, value_prob, draw_prob, plies_left = next(train_iter)

            target = {
                'self': x,
                'legal': legal_actions,
                'hl': hl,
                'value': value_prob,
                'policy': policy,
                'soft_policy': soft_policy,
                'hard_policy': hard_policy,
                'hardest_policy': hardest_policy,
                'dhl': dhl,
                'wdl': wdl,
                'draw': draw_prob,
            }
            
            with autocast(device, dtype=torch.bfloat16):
                # Forward pass
                value = model(x)
                
                # Compute loss
                losses = unwrap_model(model).losses(value, target)
                loss = cast(torch.Tensor, sum(v for k, v in losses.items() if k not in ['value', 'draw']))

            
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
            
            if is_main_process() and pbar is not None:
                pbar.set_postfix({
                    'avg_loss': f'{avg_loss:.5f}',
                    **{f'{k}': f'{v:.5f}' for k,v in metrics_loss.items()},
                    'lr': f'{scheduler.get_last_lr()[0]:.5f}'
                })
                pbar.update(1)
                
        if is_main_process() and pbar is not None:
            pbar.close()

        # Evaluate on validation set
        optimizer.eval()
        model.eval()
        if distributed:
            dist.barrier()

        if is_main_process():
            assert val_dataloader is not None
            val_metrics = {}
            val_loss = 0
            val_steps = cast(int, train_config.eval_data.num_records) // train_config.eval_data.batch_size
            val_iter = PrefetchIterator(val_dataloader, device=device)
            with torch.inference_mode():
                val_pbar = tqdm(total=val_steps, desc=f'Epoch {epoch+1}/{num_epochs}')
                for step_in_epoch in range(cast(int, val_steps)):
                    x, legal_actions, policy, soft_policy, hard_policy, hardest_policy, hl, dhl, wdl, value_prob, draw_prob, plies_left = next(val_iter)

                    target = {
                        'self': x,
                        'legal': legal_actions,
                        'hl': hl,
                        'value': value_prob,
                        'policy': policy,
                        'soft_policy': soft_policy,
                        'hard_policy': hard_policy,
                        'hardest_policy': hardest_policy,
                        'dhl': dhl,
                        'wdl': wdl,
                        'draw': draw_prob,
                    }
                    
                    with torch.inference_mode(), autocast(device, dtype=torch.bfloat16):
                        value = model(x)

                    # Compute loss
                    losses = unwrap_model(model).losses(value, target)
                    loss = cast(torch.Tensor, sum(v for k, v in losses.items() if k not in ['value', 'draw']))
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

        if is_main_process():
            avg_val_loss = val_loss / val_steps
            val_metrics_loss = {name: loss / val_steps for name, loss in val_metrics.items()}

        scheduler.step()
        if is_main_process():
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
        if is_main_process():
            checkpoint = {
                'model': (model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict()),
                'compiled': compiled,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'model_config': model_config,
                'step': step,
                "val_loss": avg_val_loss,
                **{f'val_{k}': f'{v:.5f}' for k,v in val_metrics_loss.items()},
                'world_size': world_size,
                'rank': rank,
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
        if distributed:
            dist.barrier()
    
    if distributed:
        dist.destroy_process_group()
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model


def main():
    """Main training function."""
    # Set constants
    num_return_buckets = 128
    policy = 'lc0_data'
    
    # Create model config
    model_config = TransformerConfig(
        embedding_dim=512,
        num_layers=16,
        num_heads=32,
        widening_factor=3,
        dropout=0,
    )
    
    # Create training config
    train_config = config_lib.TrainConfig(
        learning_rate=0.1,
        data=config_lib.DataConfig(
            batch_size=1024,
            shuffle=True,
            seed=4213242,
            worker_count=16,  # 0 disables multiprocessing
            num_return_buckets=num_return_buckets,
            policy=policy,
            split='train',
            dataset_path='/ephemeral/processed_data/*_0017.bag',
        ),
        eval_data=config_lib.DataConfig(
            batch_size=1024,
            shuffle=False,
            worker_count=16,  # 0 disables multiprocessing
            num_return_buckets=num_return_buckets,
            policy=policy,
            split='test',
            dataset_path='/ephemeral/processed_data/processed_lc0_data_202307*.bag',
            num_records=1_000_000
        ),
        compile=True,
        max_grad_norm=1.0,
        num_steps=170 * 1000 * 3,
        steps_per_epoch=1000 * 3,
        save_frequency=5,
        save_checkpoint_path='../checkpoints/p2/',
    )
    
    # Train model
    model = train(
        train_config=train_config,
        model_config=model_config,
    )

    # Only perform puzzle evaluation on main process
    if int(os.environ.get("RANK", "0")) == 0:
        puzzles_path = os.path.join(
            os.getcwd(),
            '../data/puzzles.csv',
        )
        puzzles = pd.read_csv(puzzles_path, nrows=10000)
        for strategy in [
            MoveSelectionStrategy.VALUE, 
            MoveSelectionStrategy.POLICY,
            MoveSelectionStrategy.SOFT_POLICY,
            MoveSelectionStrategy.HARD_POLICY,
            MoveSelectionStrategy.HARDEST_POLICY
        ]:
            engine = MyTransformerEngine(
                f"{train_config.save_checkpoint_path}checkpoint_last.pt",
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

    
    if int(os.environ.get("RANK", "0")) == 0:
        print("Training completed!")


    return model


if __name__ == "__main__":
    main()