#!/usr/bin/env python3
"""
Export a trained ChessTransformer checkpoint to ONNX.

Usage:
  python scripts/export_onnx.py \
    --checkpoint ../checkpoints/p2-newer/checkpoint_last.pt \
    --output model.onnx \
    --batch 16 \
    --opset 17 \
    [--int32-input]

Notes:
  - Exports the following outputs: value, hl, hardest_policy, U, Q
  - Input is token indices shaped [batch, S] where S is tokenizer.SEQUENCE_LENGTH
  - Batch dimension is dynamic in the exported ONNX graph
  - If --int32-input is used, the input ONNX dtype is int32 and we cast to int64 inside the wrapper
"""

import argparse
import os
from pathlib import Path
import sys
from typing import Tuple

import torch


def _prepare_imports() -> None:
    """Attempt to make project modules importable regardless of CWD."""
    repo_root = Path(__file__).resolve().parents[1]
    # Common patterns used across this repo
    sys.path.append(str(repo_root / 'src'))
    sys.path.append(str(repo_root.parent))


_prepare_imports()

from searchless_chess.src.models.transformer import ChessTransformer  # type: ignore  # noqa: E402
from searchless_chess.src import tokenizer  # type: ignore  # noqa: E402


def load_model_from_checkpoint(checkpoint_path: str, device: str) -> ChessTransformer:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ckpt['model_config']

    model = ChessTransformer(config=model_config).to(device)
    state_dict = ckpt['model']

    # Handle compiled checkpoints with _orig_mod. prefix
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k[len('_orig_mod.'):]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


class InferenceWrapper(torch.nn.Module):
    """Wrap model to expose stable ONNX I/O and handle index dtype casting."""

    def __init__(self, model: ChessTransformer, cast_input_to_long: bool = True) -> None:
        super().__init__()
        self.model = model
        self.cast_input_to_long = cast_input_to_long

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cast_input_to_long:
            tokens = tokens.long()
        out = self.model(tokens)
        # Return requested outputs
        return out['value'], out['hl'], out['hardest_policy'], out['U'], out['Q']


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    batch_size: int,
    opset: int,
    int32_input: bool,
) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model_from_checkpoint(checkpoint_path, device)
    wrapper = InferenceWrapper(model, cast_input_to_long=True).to(device)

    seq_len = tokenizer.SEQUENCE_LENGTH
    vocab_size = len(tokenizer._CHARACTERS)
    dtype = torch.int32 if int32_input else torch.int64

    dummy = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=dtype, device=device)

    dynamic_axes = {
        'tokens': {0: 'batch'},
        'value': {0: 'batch'},
        'hl': {0: 'batch'},
        'hardest_policy': {0: 'batch'},
        'U': {0: 'batch'},
        'Q': {0: 'batch'},
    }

    output_names = ['value', 'hl', 'hardest_policy', 'U', 'Q']

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy,
            output_path,
            input_names=['tokens'],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )

    print(f"Exported ONNX to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Export ChessTransformer checkpoint to ONNX')
    parser.add_argument('--checkpoint', required=True, help='Path to PyTorch checkpoint .pt')
    parser.add_argument('--output', required=True, help='Path to write ONNX model')
    parser.add_argument('--batch', type=int, default=16, help='Dummy batch size for export (batch is dynamic in ONNX)')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    parser.add_argument('--int32-input', action='store_true', help='Export ONNX input as int32 (cast to int64 inside)')
    args = parser.parse_args()

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch,
        opset=args.opset,
        int32_input=args.int32_input,
    )


if __name__ == '__main__':
    main()


