#!/usr/bin/env python3
"""
Quick FP16 profiler for ChessTransformer model.
Loads PyTorch checkpoint and profiles inference with float16 precision.
"""

import argparse
import time
import torch
import numpy as np
from contextlib import contextmanager

from src.models.transformer import ChessTransformer, TransformerConfig
from src import tokenizer

import os
os.makedirs("./checkpoints/sample-nets", exist_ok=True)


@contextmanager
def timer():
    """Simple timing context manager."""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
    

def load_model_fp16(checkpoint_path: str, device: str = "cuda") -> ChessTransformer:
    """Load model from checkpoint and convert to FP16."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # # Load checkpoint
    # checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # model_config = checkpoint['model_config']
    
    # # Create model
    # model = ChessTransformer(config=model_config).to(device)
    
    # # Handle compiled models
    # state_dict = checkpoint.get('model', checkpoint)
    # if checkpoint.get('compiled', False):
    #     print("Detected compiled model, compiling model...")
    #     model = torch.compile(model)
    
    # model.load_state_dict(state_dict)

    model_config = TransformerConfig(
        embedding_dim=1024,
        num_layers=40,
        num_heads=1024 // 16,
        widening_factor=3,
    )
    model = ChessTransformer(config=model_config).to(device)

    model = torch.compile(model)
    
    # Export model to checkpoint
    checkpoint = {
        'model': model.state_dict(),
        'model_config': model_config,
        'compiled': True,
        'step': 0,
    }
    path = f"./checkpoints/sample-nets/net_{model_config.embedding_dim}_{model_config.num_layers}_{model_config.num_heads}_{int(model_config.widening_factor)}_{tokenizer.SEQUENCE_LENGTH}_smolgen_large.pt"
    torch.save(checkpoint, path)
    print(f"Model exported to {path}")
    
    # Convert to FP16
    model = model.half()  # Convert to float16
    model.eval()
    
    print(f"Model loaded and converted to FP16!")
    # print(f"Model config: {model_config}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def create_input_data(batch_size: int, device: str = "cuda") -> torch.Tensor:
    """Create random input data."""
    seq_len = tokenizer.SEQUENCE_LENGTH
    vocab_size = len(tokenizer._CHARACTERS)
    
    input_tokens = torch.randint(
        0, vocab_size, 
        (batch_size, seq_len), 
        device=device, 
        dtype=torch.long  # Keep input as int64, model will handle FP16 internally
    )
    
    return input_tokens


def profile_model(model: ChessTransformer, input_tokens: torch.Tensor, num_runs: int = 100) -> dict:
    """Profile model inference."""
    print(f"Profiling with input shape: {input_tokens.shape}")
    print(f"Number of runs: {num_runs}")
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tokens)
    
    torch.cuda.synchronize()
    
    # Profile
    print("Profiling...")
    times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            if (i + 1) % 20 == 0:
                print(f"  Run {i+1}/{num_runs}")
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            outputs = model(input_tokens)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Calculate statistics
    times = np.array(times)
    results = {
        'avg_time_ms': np.mean(times),
        'std_time_ms': np.std(times),
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times),
        'median_time_ms': np.median(times),
        'p95_time_ms': np.percentile(times, 95),
        'p99_time_ms': np.percentile(times, 99),
        'throughput_samples_per_sec': input_tokens.shape[0] / (np.mean(times) / 1000),
        'batch_size': input_tokens.shape[0],
        'sequence_length': input_tokens.shape[1]
    }
    
    return results, outputs


def check_memory_usage():
    """Check GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated
        }
    return {}


def print_results(results: dict, memory_info: dict):
    """Print profiling results."""
    print("\n" + "="*60)
    print("🚀 FP16 PROFILING RESULTS")
    print("="*60)
    
    print(f"📊 Batch Size: {results['batch_size']}")
    print(f"📏 Sequence Length: {results['sequence_length']}")
    print(f"🎯 Total Samples: {results['batch_size'] * results['sequence_length']:,}")
    
    print(f"\n⏱️  Timing Results:")
    print(f"  Average:    {results['avg_time_ms']:.2f} ms ± {results['std_time_ms']:.2f} ms")
    print(f"  Median:     {results['median_time_ms']:.2f} ms")
    print(f"  Min/Max:    {results['min_time_ms']:.2f} / {results['max_time_ms']:.2f} ms")
    print(f"  95th %ile:  {results['p95_time_ms']:.2f} ms")
    print(f"  99th %ile:  {results['p99_time_ms']:.2f} ms")
    
    print(f"\n🚀 Throughput:")
    print(f"  Samples/sec: {results['throughput_samples_per_sec']:.0f}")
    print(f"  Batches/sec: {results['throughput_samples_per_sec'] / results['batch_size']:.2f}")
    
    if memory_info:
        print(f"\n💾 GPU Memory:")
        print(f"  Allocated:   {memory_info['allocated_gb']:.2f} GB")
        print(f"  Reserved:    {memory_info['reserved_gb']:.2f} GB") 
        print(f"  Peak:        {memory_info['max_allocated_gb']:.2f} GB")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Quick FP16 profiler for ChessTransformer")
    parser.add_argument("--checkpoint", required=True, help="Path to PyTorch checkpoint")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for profiling")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of inference runs")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs")
    
    args = parser.parse_args()
    
    print("🔥 Quick FP16 Profiler for ChessTransformer")
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"🎯 Runs: {args.num_runs}")
    print(f"💻 Device: {args.device}")
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("❌ CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    if args.device == "cuda":
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # Load model
    model = load_model_fp16(args.checkpoint, args.device)
    
    # Create input data
    input_tokens = create_input_data(args.batch_size, args.device)
    
    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Profile model
    results, outputs = profile_model(model, input_tokens, args.num_runs)
    
    # Check memory usage
    memory_info = check_memory_usage()
    
    # Print results
    print_results(results, memory_info)
    
    # Show output shapes
    print("\n📋 Model Outputs:")
    for name, output in outputs.items():
        print(f"  {name}: {list(output.shape)} ({output.dtype})")
    
    print(f"\n✅ Profiling completed! Average: {results['avg_time_ms']:.2f}ms per batch")


if __name__ == "__main__":
    main() 