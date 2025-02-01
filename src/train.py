"""PyTorch implementation of the training algorithm for action-value prediction."""

from itertools import cycle
import os
from typing import Any, cast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
torch.set_default_dtype(torch.bfloat16)
torch.set_printoptions(profile="full")
from tqdm import tqdm

from searchless_chess.src import config as config_lib
from searchless_chess.src.data_loader import build_data_loader, _TRANSFORMATION_BY_POLICY
from searchless_chess.src import tokenizer
from searchless_chess.src import utils

from searchless_chess.src.dataset import ChessDataset
from searchless_chess.src.models.transformer import TransformerConfig, ChessTransformer
import grain.python as pygrain
import bagz

from searchless_chess.src import constants
from searchless_chess.src.data_loader import _process_fen
import numpy as np

# class TupleBatchTransform(pygrain.MapTransform):
#     def __init__(self, batch_size):
#         self.batch_size = batch_size
#         self.batch = []
    
#     def map(self, element):
#         self.batch.append(element)
#         if len(self.batch) == self.batch_size:
#             # Stack each tensor in the tuple separately
#             result = tuple(torch.stack([b[i] for b in self.batch]) for i in range(len(batch[0])))
#             batch = []
#             return result
#         return None
    

# def custom_batch_function(record_metadata_list: list[pygrain.RecordMetadata], 
#                           dataset: pygrain.MapDataset,
#                           batch_size: int = 32) -> tuple[torch.Tensor, ...]:
#     """
#     Custom batch function that creates batches of PyTorch tensors.
    
#     Args:
#         record_metadata_list: List of RecordMetadata objects.
#         dataset: The PyGrain dataset.
#         batch_size: The desired batch size (default: 32).
    
#     Returns:
#         A tuple of batched PyTorch tensors.
#     """
#     # Ensure we don't exceed the number of available records
#     batch_size = min(batch_size, len(record_metadata_list))
    
#     # Initialize lists to hold individual tensors
#     batched_data = [[] for _ in range(len(dataset[0]))]
    
#     for metadata in record_metadata_list[:batch_size]:
#         # Get the record data (assumed to be a tuple of tensors)
#         record_data = dataset[metadata.index]
        
#         # Add each tensor to its respective list
#         for i, tensor in enumerate(record_data):
#             batched_data[i].append(tensor)
    
#     # Stack the tensors in each list
#     batched_tensors = tuple(torch.stack(tensor_list) for tensor_list in batched_data)
    
#     return batched_tensors
class ConvertToTorch(pygrain.MapTransform):
    def map(self, element):
        return tuple(torch.from_numpy(arr) for arr in element)

        

def load_datasource(config: config_lib.DataConfig):
    data_path = os.path.join(
        os.getcwd(),
        f'../data/{config.split}/{config.policy}_data.bag'
    )
    bag_source = bagz.BagDataSource(data_path)
    sampler = pygrain.IndexSampler(
        num_records=len(bag_source),
        shard_options=pygrain.NoSharding(),
        shuffle=False,
        num_epochs=None,
    )

    transformations = (
        _TRANSFORMATION_BY_POLICY[config.policy](num_return_buckets=config.num_return_buckets),
        pygrain.Batch(batch_size=config.batch_size),
        ConvertToTorch(),
    )

    return pygrain.DataLoader(
        data_source=bag_source,
        sampler=sampler,
        operations=transformations,
        worker_count=config.worker_count,
        read_options=None,
    )

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
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model_config =  checkpoint['model_config']
        step = checkpoint['step']
        # Create model that matches the checkpoint
        model = ChessTransformer(
            config=model_config,
        ).to(device)

        if train_config.compile:
            model = cast(ChessTransformer, torch.compile(model))

        model.load_state_dict(checkpoint['model'])
        print(f"Loaded model from checkpoint: {checkpoint_path}")

    else:
        # Initialize model
        model = ChessTransformer(model_config).to(device)
        if train_config.compile:
            model = cast(ChessTransformer, torch.compile(model))

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )

    if checkpoint is not None and 'optimizer' in checkpoint:
        print("Loading Optimizer from checkpoint...")
        optimizer.load_state_dict(checkpoint['optimizer'])

    # After creating the optimizer, add the scheduler:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=train_config.ckpt_frequency * 2,  # First restart cycle length
        T_mult=2,  # Each cycle gets twice as long
        eta_min=train_config.learning_rate / 100  # Minimum learning rate
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

            x, win_prob = next(train_iter)
                
            x = x.to(device)
            win_prob = win_prob.to(torch.bfloat16).to(device)

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
                x, win_prob = next(val_iter)

                x = x.to(device)
                win_prob = win_prob.to(torch.bfloat16).to(device)

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
                    'avg_val_loss': f'{avg_val_loss:.5f}',
                    **{f'{k}': f'{v:.5f}' for k,v in val_metrics_loss.items()},
                })

                val_pbar.update(1)

        val_pbar.close()

        avg_val_loss = val_loss / val_steps
        val_metrics_loss = {name: loss / val_steps for name, loss in val_metrics.items()}
        print({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            **{f'{k}': f'{v:.5f}' for k,v in metrics_loss.items()},
            "val_loss": avg_val_loss,
            **{f'val_{k}': f'{v:.5f}' for k,v in val_metrics_loss.items()},
            'lr': f'{scheduler.get_last_lr()[0]:.5f}',
            'step': step,
        })
        
        # Checkpointing
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
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
    policy = 'state_value'
    
    # Create model config
    model_config = TransformerConfig(
        embedding_dim=256,
        num_layers=8,
        num_heads=16,
        widening_factor=3,
        dropout=0.1,
    )
    
    # Create training config
    train_config = config_lib.TrainConfig(
        learning_rate=1e-4,
        data=config_lib.DataConfig(
            batch_size=1024,
            shuffle=True,
            worker_count=4,  # 0 disables multiprocessing
            num_return_buckets=num_return_buckets,
            policy=policy,
            split='train',
        ),
        eval_data=config_lib.DataConfig(
            batch_size=1024,
            shuffle=False,
            worker_count=4,  # 0 disables multiprocessing
            num_return_buckets=num_return_buckets,
            policy=policy,
            split='test',
            num_records=62000,
        ),
        compile=True,
        log_frequency=1,
        num_steps=60000,
        ckpt_frequency=20,
        save_frequency=1000,
        save_checkpoint_path='../checkpoints/local/',
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