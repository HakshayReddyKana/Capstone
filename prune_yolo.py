#!/usr/bin/env python3
"""
YOLOv8 Pruning Script - Compatible with current ultralytics
Based on yolov8-prune repository methodology
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from pathlib import Path
from ultralytics import YOLO

def prune_yolo_model(model, prune_ratio=0.3):
    """
    Prune YOLOv8 model based on BatchNorm gamma coefficients
    """
    print(f"Pruning model with ratio: {prune_ratio}")

    # Get all BatchNorm layers and their gamma values
    bn_weights = []
    bn_layers = []

    for name, module in model.model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            # Skip layers with residual connections to maintain tensor dimensions
            if 'cv1' in name and any('cv2' in sibling for sibling in str(module).split()):
                continue
            bn_weights.extend(module.weight.data.abs().cpu().numpy())
            bn_layers.append((name, module))

    # Sort weights and find threshold
    sorted_weights = sorted(bn_weights)
    threshold_idx = int(len(sorted_weights) * prune_ratio)
    threshold = sorted_weights[threshold_idx]

    print(f"Pruning threshold: {threshold:.6f}")
    print(f"Total BN weights: {len(bn_weights)}")

    # Count pruned channels
    total_channels = 0
    pruned_channels = 0

    for name, module in bn_layers:
        mask = module.weight.data.abs() > threshold
        pruned_count = (~mask).sum().item()
        total_count = mask.numel()

        total_channels += total_count
        pruned_channels += pruned_count

        print(f"{name}: {pruned_count}/{total_count} channels pruned")

        # Apply mask to gamma and beta
        module.weight.data.mul_(mask.float())
        if module.bias is not None:
            module.bias.data.mul_(mask.float())

    prune_ratio_actual = pruned_channels / total_channels
    print(".1f")

    return model

def main():
    parser = argparse.ArgumentParser(description='Prune YOLOv8 model')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='model weights path')
    parser.add_argument('--prune-ratio', type=float, default=0.3, help='prune ratio')
    parser.add_argument('--output', type=str, default='pruned_model.pt', help='output path')

    args = parser.parse_args()

    print("Loading model...")
    model = YOLO(args.weights)

    print("Applying pruning...")
    pruned_model = prune_yolo_model(model, args.prune_ratio)

    print(f"Saving pruned model to {args.output}")
    pruned_model.save(args.output)

    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    orig_params = count_parameters(model.model)
    pruned_params = count_parameters(pruned_model.model)

    print("\nResults:")
    print(f"Original parameters: {orig_params:,}")
    print(f"Pruned parameters: {pruned_params:,}")
    print(".1f")

    print("✓ Pruning completed successfully!")

if __name__ == "__main__":
    main()