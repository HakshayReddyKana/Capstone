"""
YOLOv8 Pruning - Conservative Approach for Reliable Results
This script implements a safer pruning methodology that preserves model functionality.
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import os

def gentle_prune_yolo_model(model_path, prune_ratio=0.1):
    """
    Conservative pruning approach that preserves model functionality
    """
    print("YOLOv8 Gentle Pruning - Conservative Approach")
    print("=" * 60)
    print(f"Target pruning ratio: {prune_ratio}")

    # Load model
    model = YOLO(model_path)
    print("✓ Model loaded")

    # Get model info
    orig_params = sum(p.numel() for p in model.model.parameters())
    print(f"Original parameters: {orig_params:,}")

    # Conservative pruning: only prune non-critical layers
    bn_layers_pruned = 0
    total_channels_pruned = 0

    for name, module in model.model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            # Only prune if layer has many channels and we're confident
            total_channels = module.weight.data.size()[0]

            # Skip small layers (likely critical)
            if total_channels < 32:
                continue

            # Calculate how many channels to prune (very conservative)
            channels_to_prune = max(1, int(total_channels * prune_ratio * 0.5))  # Half the target ratio

            # Find channels with smallest gamma values
            gamma_abs = module.weight.data.abs()
            _, indices = torch.topk(gamma_abs, channels_to_prune, largest=False)

            # Create mask (1 for keep, 0 for prune)
            mask = torch.ones(total_channels)
            mask[indices] = 0

            # Apply mask
            module.weight.data.mul_(mask)
            module.bias.data.mul_(mask)

            pruned_count = channels_to_prune
            total_channels_pruned += pruned_count
            bn_layers_pruned += 1

            print(f"  {name}: pruned {pruned_count}/{total_channels} channels")

    # Calculate final stats
    final_params = sum(p.numel() for p in model.model.parameters())
    compression = ((orig_params - final_params) / orig_params) * 100

    print("\nPruning Summary:")
    print(f"  BN layers modified: {bn_layers_pruned}")
    print(f"  Total channels pruned: {total_channels_pruned}")
    print(f"  Parameter compression: {compression:.1f}%")
    print(f"  Actual pruning ratio: {total_channels_pruned / sum(module.weight.data.size()[0] for name, module in model.model.named_modules() if isinstance(module, nn.BatchNorm2d)):.3f}")

    # Save pruned model
    output_path = f"pruned_yolo_gentle_ratio_{prune_ratio:.2f}.pt"
    torch.save({
        'model': model.model,
        'prune_ratio': prune_ratio,
        'compression': compression,
        'method': 'gentle_masking'
    }, output_path)

    print(f"✓ Pruned model saved: {output_path}")
    return model, compression

def main():
    # Use very conservative pruning ratio
    model_path = 'Object-Detection/Yolo-V8/weights/best.pt'
    prune_ratio = 0.05  # Only 5% pruning - much safer

    print(f"Starting gentle pruning with {prune_ratio:.1%} ratio...")

    # Prune model
    pruned_model, compression = gentle_prune_yolo_model(model_path, prune_ratio)

    print("\n=== Gentle Pruning Complete ===")
    print(f"Compression achieved: {compression:.1f}%")
    print("This conservative approach should preserve model functionality.")
    print("Recommend fine-tuning for 10-20 epochs to recover any minor accuracy loss.")

if __name__ == "__main__":
    main()