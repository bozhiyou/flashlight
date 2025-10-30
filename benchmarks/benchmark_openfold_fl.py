#!/usr/bin/env python3
"""
OpenFold Inference Benchmark Script

Usage:
    python benchmark_openfold.py --model_name model_1 --n_blocks 4 --n_iterations 5
"""

import argparse
import time
import torch
import torch.nn as nn
from pathlib import Path

from openfold.config import model_config
from openfold.data import data_transforms
from openfold.model.model import AlphaFold
from openfold.utils.tensor_utils import tensor_tree_map

import torch._dynamo.config
torch._dynamo.config.cache_size_limit = 65536
import monkeypatch.fusion.triton_heuristics
# torch._dynamo.config.suppress_errors = True
# import torch._inductor.config
# torch._inductor.config.aggressive_fusion = True
# torch._inductor.config.max_autotune = True


import torch._dynamo.config
torch._dynamo.config.cache_size_limit = 65536

ENABLE_FLASHLIGHT = False
def apply_patch():
    global ENABLE_FLASHLIGHT
    from monkeypatch import disable_flashattention_replacement
    from monkeypatch.fusion import dependent_reduction_fusion
    from monkeypatch.fusion import block_reduction
    from monkeypatch.fusion import reduction_kernel_fusion
    ENABLE_FLASHLIGHT = True
    disable_flashattention_replacement()
    print("FLASHLIGHT enabled")
apply_patch()


def generate_random_batch(c, n_seq, n_templ, n_res, n_extra_seq, is_multimer=False):
    """Generate random input batch for OpenFold model."""
    batch = {}
    
    # Target features
    tf = torch.randint(c.model.input_embedder.tf_dim - 1, size=(n_res,))
    batch["target_feat"] = nn.functional.one_hot(
        tf, c.model.input_embedder.tf_dim
    ).float()
    batch["aatype"] = torch.argmax(batch["target_feat"], dim=-1)
    batch["residue_index"] = torch.arange(n_res)
    
    # MSA features
    batch["msa_feat"] = torch.rand((n_seq, n_res, c.model.input_embedder.msa_dim))
    batch["msa_mask"] = torch.randint(low=0, high=2, size=(n_seq, n_res)).float()
    batch["seq_mask"] = torch.randint(low=0, high=2, size=(n_res,)).float()
    
    # Template features
    from tests.data_utils import random_template_feats, random_extra_msa_feats
    t_feats = random_template_feats(n_templ, n_res)
    batch.update({k: torch.tensor(v) for k, v in t_feats.items()})
    
    # Extra MSA features
    extra_feats = random_extra_msa_feats(n_extra_seq, n_res)
    batch.update({k: torch.tensor(v) for k, v in extra_feats.items()})
    
    # Atom14 masks
    batch.update(data_transforms.make_atom14_masks(batch))
    
    # Recycling iterations
    batch["no_recycling_iters"] = torch.tensor(2.)
    
    # Multimer-specific features
    if is_multimer:
        from tests.data_utils import random_asym_ids
        batch["asym_id"] = torch.as_tensor(random_asym_ids(n_res))
        batch["entity_id"] = batch["asym_id"].clone()
        batch["sym_id"] = torch.ones(n_res)
        batch["extra_deletion_matrix"] = torch.randint(0, 2, size=(n_extra_seq, n_res))
    
    # Add recycling dimensions
    add_recycling_dims = lambda t: (
        t.unsqueeze(-1).expand(*t.shape, c.data.common.max_recycling_iters)
    )
    batch = tensor_tree_map(add_recycling_dims, batch)
    
    return batch


def benchmark_model(model, batch, n_iterations=10, warmup_iterations=2):
    """Benchmark model inference time."""
    # model.eval()
    
    print(f"\nRunning {warmup_iterations} warmup iterations...")
    with torch.no_grad():
        for i in range(warmup_iterations):
            _ = model(batch)
            torch.cuda.synchronize()
    
    print(f"Running {n_iterations} timed iterations...")
    times = []
    
    with torch.no_grad():
        for i in range(n_iterations):
            torch.cuda.synchronize()
            start = time.time()
            
            out = model(batch)
            
            torch.cuda.synchronize()
            end = time.time()
            
            elapsed = end - start
            times.append(elapsed)
            print(f"  Iteration {i+1}/{n_iterations}: {elapsed:.4f}s")
    
    return times


def print_statistics(times):
    """Print benchmark statistics."""
    import statistics
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Number of iterations: {len(times)}")
    print(f"Mean time:   {statistics.mean(times):.4f}s")
    print(f"Median time: {statistics.median(times):.4f}s")
    print(f"Std dev:     {statistics.stdev(times):.4f}s" if len(times) > 1 else "Std dev: N/A")
    print(f"Min time:    {min(times):.4f}s")
    print(f"Max time:    {max(times):.4f}s")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark OpenFold inference")
    parser.add_argument("--model_name", type=str, default="model_1",
                        help="Model configuration name (e.g., model_1, model_2)")
    parser.add_argument("--n_seq", type=int, default=32,
                        help="Number of MSA sequences")
    parser.add_argument("--n_templ", type=int, default=4,
                        help="Number of templates")
    parser.add_argument("--n_res", type=int, default=256,
                        help="Number of residues")
    parser.add_argument("--n_extra", type=int, default=1024,
                        help="Number of extra MSA sequences")
    parser.add_argument("--n_blocks", type=int, default=48,
                        help="Number of Evoformer blocks (default 48 for full model)")
    parser.add_argument("--n_iterations", type=int, default=10,
                        help="Number of benchmark iterations")
    parser.add_argument("--warmup_iterations", type=int, default=2,
                        help="Number of warmup iterations")
    parser.add_argument("--use_checkpoint", action="store_true",
                        help="Enable gradient checkpointing (blocks_per_ckpt)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile() on the model")
    parser.add_argument("--compile_mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="Torch compile mode")
    parser.add_argument("--compile_selective", action="store_true",
                        help="Compile only specific modules instead of entire model")
    
    args = parser.parse_args()
    
    print("="*60)
    print("OpenFold Inference Benchmark")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Sequences: {args.n_seq}, Templates: {args.n_templ}")
    print(f"Residues: {args.n_res}, Extra MSA: {args.n_extra}")
    print(f"Evoformer blocks: {args.n_blocks}")
    print(f"Torch compile: {args.compile}" + (f" (mode: {args.compile_mode})" if args.compile else ""))
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a GPU.")
    
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Load model configuration
    print("\nLoading model configuration...")
    c = model_config(args.model_name)
    c.model.evoformer_stack.no_blocks = args.n_blocks
    
    if not args.use_checkpoint:
        c.model.evoformer_stack.blocks_per_ckpt = None
    
    # Check if multimer
    is_multimer = "multimer" in args.model_name.lower()
    
    # Create model
    print("Creating model...")
    model = AlphaFold(c).cuda()

    model.eval()

    if args.compile:
        print(f"Compiling model with mode='{args.compile_mode}'...")
        try:
            if args.compile_selective:
                # Compile only specific modules - ipa
                print("Using selective compilation (structure module only)...")
                if hasattr(model, 'structure_module'):
                    model.structure_module.ipa = torch.compile(dynamic=False)(model.structure_module.ipa)
                # if hasattr(model, 'evoformer'):
                #     model.evoformer = torch.compile(model.evoformer, mode=args.compile_mode)
            else:
                # Compile entire model
                model = torch.compile(model, mode=args.compile_mode, fullgraph=False)
            print("Compilation successful!")
        except Exception as e:
            print(f"Warning: torch.compile failed with error: {e}")
            print("Falling back to eager mode execution...")
            # Model remains in eager mode


    # Generate random batch
    print("Generating random input batch...")
    batch = generate_random_batch(
        c, args.n_seq, args.n_templ, args.n_res, args.n_extra, is_multimer
    )
    
    # Move to CUDA
    to_cuda = lambda t: t.cuda()
    batch = tensor_tree_map(to_cuda, batch)
    
    # Print memory usage
    print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"GPU Memory reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
    
    # Run benchmark
    times = benchmark_model(model, batch, args.n_iterations, args.warmup_iterations)
    
    # Print results
    print_statistics(times)
    
    # Print final memory usage
    print(f"\nFinal GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Final GPU Memory reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")


if __name__ == "__main__":
    main()