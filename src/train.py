"""PyTorch implementation of the training algorithm for action-value prediction."""

from itertools import cycle
import os
from typing import Any, cast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
torch.set_default_dtype(torch.float32)
torch.set_printoptions(profile="full")
from tqdm import tqdm

from searchless_chess.src import config as config_lib

from searchless_chess.src.dataset import load_datasource
from searchless_chess.src.models.transformer import TransformerConfig, ChessTransformer
from searchless_chess.src.optimizer.soap import SOAP


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
    optimizer = SOAP(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        precondition_frequency=30,
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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=170 * 3000,  # Total number of steps for one cosine cycle
        eta_min=train_config.learning_rate / 10  # Minimum learning rate (1/10th of initial)
    )

    if checkpoint is not None and 'scheduler' in checkpoint:
        print("Loading Scheduler from checkpoint...")
        scheduler.load_state_dict(checkpoint['scheduler'])

    # Setup SWA utilities
    swa_model = None
    swa_scheduler = None
    if train_config.use_swa:
        print("Using SWA...")
        swa_model = AveragedModel(model, device=device)
        swa_scheduler = SWALR(optimizer, swa_lr=train_config.swa_lr)
        if checkpoint is not None and 'swa_model' in checkpoint:
            print("Loading SWA model from checkpoint...")
            swa_model.load_state_dict(checkpoint['swa_model'])
        # Note: SWALR state is implicitly handled by the base optimizer state loading,
        # but we might need to load its last_epoch if explicitly saved.
        # For simplicity, we'll re-initialize its state based on the current step below.

    train_iter = train_dataloader.__iter__()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    # Recalculate starting epoch and adjust scheduler/SWA states if loading checkpoint
    start_epoch = step // train_config.ckpt_frequency
    scheduler.last_epoch = start_epoch -1 # Adjust base scheduler epoch
    if train_config.use_swa and swa_scheduler is not None and step > train_config.swa_start_step:
         # Adjust SWA scheduler based on steps taken *within* the SWA phase
         swa_steps_taken = step - train_config.swa_start_step
         # SWALR's internal step counter relies on optimizer steps, so we need to simulate
         # Note: This assumes swa_scheduler.step() was called appropriately in the past.
         # A potentially more robust way if saving/loading SWALR state directly isn't done:
         # Set swa_scheduler.start_epoch = train_config.swa_start_step // train_config.ckpt_frequency
         # and swa_scheduler._step_count based on steps into SWA. Let's keep it simple for now.
         # For SWALR, the LR depends on the *base* optimizer's LR scheduler state and swa_lr.
         # We loaded the base scheduler state, so it should be okay.
         pass # SWALR state depends on the base optimizer/scheduler state which are loaded

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        metrics = {}
        total_loss = 0
        avg_loss = 0
        metrics_loss = {}
        pbar = tqdm(total=train_config.ckpt_frequency, desc=f'Epoch {epoch+1}/{num_epochs}', initial=step % train_config.ckpt_frequency)
        
        epoch_start_step = step # Track step at the beginning of the epoch for SWA update logic

        for step_in_epoch in range(train_config.ckpt_frequency):
            # Check if we have already completed steps for this epoch from a checkpoint
            if step < epoch_start_step + step_in_epoch:
                 continue # Skip steps already done

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
                loss = cast(torch.Tensor, sum(v for k, v in losses.items() if k not in ['value', 'avs']))

            
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
            current_step_in_epoch = (step % train_config.ckpt_frequency) + 1
            avg_loss = total_loss / current_step_in_epoch
            metrics_loss = {name: loss / current_step_in_epoch for name, loss in metrics.items()}

            pbar.set_postfix({
                'avg_loss': f'{avg_loss:.5f}',
                **{f'{k}': f'{v:.5f}' for k,v in metrics_loss.items()},
                'lr': f'{optimizer.param_groups[0]["lr"]:.5f}' # Get actual LR from optimizer
            })
            pbar.update(1)
            step += 1 # Increment global step counter

            # Scheduler Step
            if train_config.use_swa and step > train_config.swa_start_step:
                 swa_scheduler.step() # Step SWA scheduler after optimizer step
                 if (step - train_config.swa_start_step) % train_config.swa_update_freq == 0:
                     swa_model.update_parameters(model) # Update SWA model periodically using the base model
            else:
                scheduler.step() # Step base scheduler if not in SWA phase


        pbar.close()

        # Evaluate on validation set (using the *base* model during training epochs)
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
                loss = cast(torch.Tensor, sum(v for k, v in losses.items() if k not in ['value', 'avs']))
                # Update totals
                val_metrics = {name: loss.item() + val_metrics.get(name, 0) for name, loss in losses.items()}
                val_loss += loss.item()

                # Update progress bar
                avg_val_loss = val_loss / (step_in_epoch + 1)
                val_metrics_loss = {name: l / (step_in_epoch + 1) for name, l in val_metrics.items()}
                val_pbar.set_postfix({
                    'avg_val_loss': f'{avg_val_loss:.5f}',
                    **{f'{k}': f'{v:.5f}' for k,v in val_metrics_loss.items()},
                })
                val_pbar.update(1)

        val_pbar.close()

        avg_val_loss = val_loss / val_steps
        val_metrics_loss = {name: loss / val_steps for name, loss in val_metrics.items()}

        # Final calculation for average validation loss
        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
        val_metrics_loss = {name: loss / val_steps if val_steps > 0 else 0 for name, loss in val_metrics.items()}

        print({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            **{f'{k}': f'{v:.6f}' for k,v in metrics_loss.items()},
            "val_loss": avg_val_loss,
            **{f'val_{k}': f'{v:.6f}' for k,v in val_metrics_loss.items()},
            'lr': f'{optimizer.param_groups[0]["lr"]:.5f}',
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
        if train_config.use_swa and swa_model is not None:
             checkpoint['swa_model'] = swa_model.state_dict()
             # Optionally save swa_scheduler state if needed, though often recalculated
             # checkpoint['swa_scheduler'] = swa_scheduler.state_dict()

        checkpoint_dir = os.path.join(
            os.getcwd(),
            train_config.save_checkpoint_path
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save checkpoint based on global step count
        save_path = os.path.join(checkpoint_dir, f'checkpoint_{step}.pt')
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    # === SWA Finalization ===
    if train_config.use_swa and swa_model is not None:
        print("Updating SWA Batch Norm statistics...")
        # Ensure train_iter is reset or grab a new dataloader instance
        bn_update_loader = load_datasource(train_config.data) # Get a fresh loader
        bn_update_iter = iter(bn_update_loader)
        # update_bn requires the SWA model to be in evaluation mode potentially? Check docs.
        # It internally sets train(True) then train(False). Let's ensure it's eval before/after.
        swa_model.eval()
        update_bn(bn_update_iter, swa_model, device=device)
        # Run final validation on the SWA model
        print("Evaluating SWA model on validation set...")
        swa_model.eval() # Ensure evaluation mode

        val_metrics = {}
        val_loss = 0
        val_steps = (train_config.eval_data.num_records + train_config.eval_data.batch_size - 1) // train_config.eval_data.batch_size
        val_iter = iter(val_dataloader) # Reset validation loader iterator
        with torch.inference_mode():
            val_pbar = tqdm(total=val_steps, desc='SWA Final Validation')
            for current_val_step in range(val_steps):
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

                # Use swa_model for inference
                with autocast(device, dtype=torch.bfloat16):
                    value = swa_model(x)

                # Compute loss
                losses = swa_model.module.losses(value, target) # Access original model's loss fn
                loss = cast(torch.Tensor, sum(v for k, v in losses.items() if k not in ['value', 'avs']))
                val_metrics = {name: loss.item() + val_metrics.get(name, 0) for name, loss in losses.items()}
                val_loss += loss.item()

                avg_val_loss = val_loss / (current_val_step + 1)
                val_metrics_loss = {name: l / (current_val_step + 1) for name, l in val_metrics.items()}
                val_pbar.set_postfix({
                    'avg_swa_val_loss': f'{avg_val_loss:.5f}',
                     **{f'swa_{k}': f'{v:.5f}' for k,v in val_metrics_loss.items()},
                })
                val_pbar.update(1)
            val_pbar.close()

        avg_swa_val_loss = val_loss / val_steps if val_steps > 0 else 0
        swa_val_metrics_loss = {name: loss / val_steps if val_steps > 0 else 0 for name, loss in val_metrics.items()}
        print("SWA Final Validation Results:")
        print({
            "avg_swa_val_loss": avg_swa_val_loss,
            **{f'swa_val_{k}': f'{v:.6f}' for k,v in swa_val_metrics_loss.items()},
        })
        # Save the final SWA model state separately if desired
        swa_final_path = os.path.join(checkpoint_dir, f'swa_model_final_{step}.pt')
        torch.save({'model': swa_model.state_dict(), 'model_config': model_config}, swa_final_path)
        print(f"Final SWA model saved to {swa_final_path}")
        return swa_model # Return the averaged model

    return model # Return the base model if SWA is not used


def main():
    """Main training function."""
    # Set constants
    num_return_buckets = 128
    policy = 'action_values'
    
    # Create model config
    model_config = TransformerConfig(
        embedding_dim=480,
        num_layers=16,
        num_heads=15,
        widening_factor=3,
        dropout=0,
    )
    
    # Create training config
    total_steps = 170 * 3000
    ckpt_freq = 3000
    train_config = config_lib.TrainConfig(
        learning_rate=5.6e-4,
        data=config_lib.DataConfig(
            batch_size=2048,
            shuffle=True,
            seed=42173522,
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
        num_steps=total_steps,
        ckpt_frequency=ckpt_freq,
        save_frequency=ckpt_freq,
        save_checkpoint_path='../checkpoints/layer-16-480-15-56lr-swa',
        # --- SWA Configuration ---
        use_swa=True, # Enable SWA
        # Start SWA after ~80% of training steps. Adjust as needed.
        swa_start_step=int(total_steps * 0.8),
        # SWA LR is typically smaller than the initial LR. Adjust as needed.
        swa_lr=5.6e-5,
        # Update SWA model every 30 steps during the SWA phase.
        swa_update_freq=30,
        # --- End SWA Configuration ---
    )
    
    # Train model
    final_model = train(
        train_config=train_config,
        model_config=model_config,
    )
    
    print("Training completed!")
    # The returned 'final_model' will be the SWA model if use_swa=True, else the base model.
    return final_model


if __name__ == "__main__":
    main()