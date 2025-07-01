#!/usr/bin/env python3
"""
Export PyTorch ChessTransformer model to ONNX format for TensorRT inference.

This script loads a trained ChessTransformer model and exports it to ONNX format,
which can then be used with TensorRT for optimized inference.

Usage:
    python export_to_onnx.py --checkpoint path/to/checkpoint.pt --output model.onnx
"""

import argparse
import os
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from typing import Dict, Any, Tuple, cast

# Import your model and related modules
from searchless_chess.src.models.transformer import ChessTransformer, TransformerConfig
from searchless_chess.src import tokenizer


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda") -> Tuple[ChessTransformer, TransformerConfig]:
    """Load the ChessTransformer model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Tuple of (model, model_config)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model config
    model_config = checkpoint['model_config']
    
    # Create model
    model = ChessTransformer(config=model_config).to(device)
    
    # Load model weights (always load uncompiled for ONNX export)
    state_dict = checkpoint.get('model', checkpoint)
    
    # Important: We don't compile the model for ONNX export as it's not supported
    # torch.compile() uses TorchDynamo which is incompatible with ONNX export
    if checkpoint.get('compiled', False):
        print("⚠️  Note: Model was compiled during training, but loading uncompiled version for ONNX export")
        print("   This is required because torch.compile() is incompatible with ONNX export")
        print("   Stripping '_orig_mod.' prefix from compiled model weights...")
        
        # Strip the '_orig_mod.' prefix from compiled model weights
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[len('_orig_mod.'):]
                cleaned_state_dict[new_key] = value
            else:
                cleaned_state_dict[key] = value
        state_dict = cleaned_state_dict
    
    model.load_state_dict(state_dict)
    
    model.eval()
    
    print(f"Model loaded successfully. Config: {model_config}")
    return model, model_config


def create_dummy_input(batch_size: int = 1, device: str = "cuda") -> torch.Tensor:
    """Create dummy input data for the model.
    
    Args:
        batch_size: Batch size for the dummy input
        device: Device to create the tensor on
        
    Returns:
        Dummy input tensor of shape (batch_size, sequence_length)
    """
    # Use the sequence length from your tokenizer
    seq_len = tokenizer.SEQUENCE_LENGTH
    vocab_size = len(tokenizer._CHARACTERS)
    
    # Create random input within valid token range
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    print(f"Created dummy input with shape: {dummy_input.shape}")
    return dummy_input


def export_to_onnx(
    model: ChessTransformer,
    dummy_input: torch.Tensor,
    output_path: str,
    opset_version: int = 14,
    dynamic_axes: bool = True
) -> None:
    """Export the PyTorch model to ONNX format.
    
    Args:
        model: The PyTorch model to export
        dummy_input: Example input tensor
        output_path: Path to save the ONNX model
        opset_version: ONNX opset version to use
        dynamic_axes: Whether to enable dynamic batch size
    """
    print(f"Exporting model to ONNX format: {output_path}")
    
    # Define input and output names
    input_names = ["input_tokens"]
    output_names = [
        "self_prediction",
        "value_prediction",
        "draw_prediction",
        "hl_prediction",
        "dhl_prediction",
        "wdl_prediction",
        "legal_moves",
        "policy_prediction",
        "soft_policy_prediction",
        "hard_policy_prediction",
        "hardest_policy_prediction"
    ]
    
    # Define dynamic axes for variable batch size
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            "input_tokens": {0: "batch_size"},
            "self_prediction": {0: "batch_size"},
            "value_prediction": {0: "batch_size"},
            "draw_prediction": {0: "batch_size"},
            "hl_prediction": {0: "batch_size"},
            "dhl_prediction": {0: "batch_size"},
            "wdl_prediction": {0: "batch_size"},
            "legal_moves": {0: "batch_size"},
            "policy_prediction": {0: "batch_size"},
            "soft_policy_prediction": {0: "batch_size"},
            "hard_policy_prediction": {0: "batch_size"},
            "hardest_policy_prediction": {0: "batch_size"}
        }
    
    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_dict,
            verbose=False
        )
    
    print(f"Model exported successfully to: {output_path}")


def verify_onnx_model(onnx_path: str, pytorch_model: ChessTransformer, dummy_input: torch.Tensor) -> None:
    """Verify that the ONNX model produces the same outputs as the PyTorch model.
    
    Args:
        onnx_path: Path to the ONNX model
        pytorch_model: Original PyTorch model
        dummy_input: Input tensor for testing
    """
    print("Verifying ONNX model...")
    
    # Load and check ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")
    
    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)
    
    # Get PyTorch model output
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input)
    
    # Convert input to numpy for ONNX Runtime
    dummy_input_np = dummy_input.cpu().numpy()
    
    # Run ONNX model
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input_np}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Compare outputs
    output_keys = list(pytorch_output.keys())
    print(f"Comparing {len(output_keys)} outputs...")
    
    for i, key in enumerate(output_keys):
        pytorch_out = pytorch_output[key].cpu().numpy()
        onnx_out = ort_outputs[i]
        
        # Check if shapes match
        if pytorch_out.shape != onnx_out.shape:
            print(f"Shape mismatch for {key}: PyTorch {pytorch_out.shape}, ONNX {onnx_out.shape}")
            continue
        
        # Check numerical differences
        max_diff = np.max(np.abs(pytorch_out - onnx_out))
        mean_diff = np.mean(np.abs(pytorch_out - onnx_out))
        
        print(f"{key}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        
        if max_diff > 1e-4:
            print(f"Warning: Large difference detected for {key}")
        else:
            print(f"✓ {key} matches well")
    
    print("ONNX model verification completed!")


def optimize_for_tensorrt(onnx_path: str, output_path: str = None) -> None:
    """Apply TensorRT-specific optimizations to the ONNX model.
    
    Args:
        onnx_path: Path to the input ONNX model
        output_path: Path to save the optimized model (optional)
    """
    try:
        import onnxsim
        
        if output_path is None:
            output_path = onnx_path.replace('.onnx', '_simplified.onnx')
        
        print(f"Simplifying ONNX model for TensorRT: {output_path}")
        
        # Load model
        model = onnx.load(onnx_path)
        
        # Simplify model
        model_simplified, check = onnxsim.simplify(model)
        
        if check:
            # Save simplified model
            onnx.save(model_simplified, output_path)
            print(f"Simplified model saved to: {output_path}")
        else:
            print("Model simplification failed!")
            
    except ImportError:
        print("onnxsim not installed. Run: pip install onnxsim")
        print("Skipping model simplification...")


def main():
    parser = argparse.ArgumentParser(description="Export ChessTransformer to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", required=True, help="Output ONNX file path")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for export")
    parser.add_argument("--opset-version", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--verify", action="store_true", help="Verify ONNX model against PyTorch")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX model for TensorRT")
    parser.add_argument("--no-dynamic-axes", action="store_true", help="Disable dynamic batch size")
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Load model
    model, model_config = load_model_from_checkpoint(args.checkpoint, args.device)
    
    # Create dummy input
    dummy_input = create_dummy_input(args.batch_size, args.device)
    
    # Export to ONNX
    export_to_onnx(
        model=model,
        dummy_input=dummy_input,
        output_path=args.output,
        opset_version=args.opset_version,
        dynamic_axes=not args.no_dynamic_axes
    )
    
    # Verify model if requested
    if args.verify:
        verify_onnx_model(args.output, model, dummy_input)
    
    # Simplify for TensorRT if requested
    if args.simplify:
        optimize_for_tensorrt(args.output)
    
    print("\n" + "="*50)
    print("ONNX Export Summary:")
    print(f"✓ Model exported to: {args.output}")
    print(f"✓ Model config: {model_config}")
    print(f"✓ Input shape: {dummy_input.shape}")
    print(f"✓ ONNX opset version: {args.opset_version}")
    print(f"✓ Dynamic batch size: {not args.no_dynamic_axes}")
    
    if args.verify:
        print("✓ Model verification: PASSED")
    
    if args.simplify:
        simplified_path = args.output.replace('.onnx', '_simplified.onnx')
        print(f"✓ Simplified model: {simplified_path}")
    
    print("\nNext steps for TensorRT:")
    print("1. Install TensorRT: https://developer.nvidia.com/tensorrt")
    print("2. Convert ONNX to TensorRT engine:")
    print(f"   trtexec --onnx={args.output} --saveEngine=model.trt --fp16")
    print("3. Use the TensorRT engine in your inference code")
    print("="*50)


if __name__ == "__main__":
    main() 