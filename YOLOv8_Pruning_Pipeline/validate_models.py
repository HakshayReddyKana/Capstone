#!/usr/bin/env python3
"""
YOLOv8 Model Validation and Comparison Script
"""

import torch
from ultralytics import YOLO
import argparse
import os

def validate_model(model_path, data_path, device='cuda'):
    """Validate a single model"""
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print("Running validation...")
    results = model.val(
        data=data_path,
        split='val',
        device=device,
        verbose=False
    )

    return {
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map,
        'precision': results.box.mp,
        'recall': results.box.mr
    }

def count_parameters(model_path):
    """Count parameters in a model"""
    model = YOLO(model_path)
    return sum(p.numel() for p in model.parameters())

def get_model_size(model_path):
    """Get model file size in MB"""
    return os.path.getsize(model_path) / (1024 * 1024)

def main():
    parser = argparse.ArgumentParser(description='Compare YOLOv8 models')
    parser.add_argument('--original', type=str, default='yolov8-prune/yolov8n.pt', help='original model path')
    parser.add_argument('--pruned', type=str, default='pruned_yolo_model.pt', help='pruned model path')
    parser.add_argument('--finetuned', type=str, help='fine-tuned model path')
    parser.add_argument('--data', type=str, default='yolov8-prune/ultralytics/cfg/datasets/VisDrone.yaml', help='dataset config')

    args = parser.parse_args()

    # Print device information
    print("=== GPU Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device.upper()}")
    print()

    # Validate all models
    results = {}

    print("\n=== VALIDATING MODELS ===")

    # Original model
    results['Original'] = validate_model(args.original, args.data, device)

    # Pruned model
    results['Pruned'] = validate_model(args.pruned, args.data, device)

    # Fine-tuned model (if provided)
    if args.finetuned and os.path.exists(args.finetuned):
        results['Fine-tuned'] = validate_model(args.finetuned, args.data, device)

    # Get model stats
    stats = {}
    for name, path in [('Original', args.original), ('Pruned', args.pruned)]:
        stats[name] = {
            'parameters': count_parameters(path),
            'size_mb': get_model_size(path)
        }

    if args.finetuned and os.path.exists(args.finetuned):
        stats['Fine-tuned'] = {
            'parameters': count_parameters(args.finetuned),
            'size_mb': get_model_size(args.finetuned)
        }

    # Print results
    print("\n=== RESULTS SUMMARY ===")
    print("Model\t\tmAP50\t\tmAP50-95\tPrecision\tRecall\tParams\t\tSize_MB")
    print("-" * 80)

    for model_name in results.keys():
        r = results[model_name]
        s = stats[model_name]
        print(".4f")

    # Calculate improvements
    print("\n=== IMPROVEMENTS ===")
    orig_params = stats['Original']['parameters']
    pruned_params = stats['Pruned']['parameters']

    print(".1f")
    print(".1f")

    if 'Fine-tuned' in results:
        ft_params = stats['Fine-tuned']['parameters']
        print(".1f")

        # Accuracy recovery
        orig_map = results['Original']['mAP50']
        pruned_map = results['Pruned']['mAP50']
        ft_map = results['Fine-tuned']['mAP50']

        print(".1f")
        print(".1f")

if __name__ == "__main__":
    main()