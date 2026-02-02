#!/usr/bin/env python3
"""
Quick verification script to show GPU mappings work correctly.
This demonstrates the difference between physical GPU indices and CUDA device indices.
"""

import os
import torch

print("="*70)
print("GPU Mapping Verification")
print("="*70)

cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
print(f"\nCUDA_VISIBLE_DEVICES: {cuda_visible}\n")

if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices available: {num_devices}")
    print("NOTE: This script shows PyTorch's CUDA device mapping, NOT nvidia-smi ordering!\n")
    
    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        
        # Get memory information
        free_memory, total_memory = torch.cuda.mem_get_info(i)
        
        print(f"CUDA Device {i} (PyTorch Mapping):")
        print(f"  Name: {props.name}")
        print(f"  Total Memory: {total_memory / (1024**3):.2f} GB")
        print(f"  Available Memory: {free_memory / (1024**3):.2f} GB")
        print(f"  Used Memory: {(total_memory - free_memory) / (1024**3):.2f} GB")
        
        # Get the actual physical device ID if we can
        if cuda_visible and cuda_visible != 'Not set':
            visible_gpus = [int(x.strip()) for x in cuda_visible.split(',') if x.strip()]
            if i < len(visible_gpus):
                print(f"  Physical GPU Index (nvidia-smi): {visible_gpus[i]}")
        print()
else:
    print("CUDA is not available!")

print("="*70)
print("\n💡 To use a specific GPU in PyTorch:")
print("   device = torch.device('cuda:X')  # X is the CUDA device index")
print("\n💡 To set CUDA_VISIBLE_DEVICES before running:")
print("   export CUDA_VISIBLE_DEVICES=0,1  # Only GPUs 0 and 1 visible")
print("   # Then CUDA device 0 → Physical GPU 0, CUDA device 1 → Physical GPU 1")
print("="*70)
