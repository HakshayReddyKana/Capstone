"""
Simple YOLOv8 Pruning Script
Run this to prune your trained model
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np

def load_model_safely(model_path):
    """Load model with multiple fallback methods"""
    print(f"Loading model from: {model_path}")

    try:
        # Method 1: Direct load
        model = YOLO(model_path)
        print("✓ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Direct load failed: {e}")

    try:
        # Method 2: Load checkpoint and transfer weights
        print("Trying alternative loading method...")

        # Load checkpoint
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        print("✓ Checkpoint loaded")

        # Create fresh model
        fresh_model = YOLO('yolov8s.pt')  # Adjust size if needed

        # Extract state dict
        if isinstance(ckpt, dict):
            if 'model' in ckpt:
                state_dict = ckpt['model'].state_dict()
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt.state_dict()

        # Load compatible weights
        compatible = {}
        for k, v in state_dict.items():
            if k in fresh_model.model.state_dict():
                if fresh_model.model.state_dict()[k].shape == v.shape:
                    compatible[k] = v

        fresh_model.model.load_state_dict(compatible, strict=False)
        print(f"✓ Loaded {len(compatible)}/{len(state_dict)} layers")
        return fresh_model

    except Exception as e:
        print(f"All loading methods failed: {e}")
        return None

def prune_yolo_model(model, prune_ratio=0.3):
    """Apply L1 BN pruning to YOLO model"""
    print(f"\n=== Pruning Model (ratio: {prune_ratio}) ===")

    # Collect BN weights
    bn_weights = []
    bn_layers = {}

    for name, module in model.model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers[name] = module
            bn_weights.extend(module.weight.data.abs().cpu().numpy())

    print(f"Found {len(bn_layers)} BN layers with {len(bn_weights)} total channels")

    # Calculate threshold
    sorted_weights = np.sort(bn_weights)
    threshold_idx = int(len(sorted_weights) * prune_ratio)
    threshold = sorted_weights[threshold_idx]

    print(f"Pruning threshold: {threshold:.6f}")

    # Apply pruning
    total_channels = 0
    pruned_channels = 0

    for name, module in bn_layers.items():
        # Create mask
        mask = (module.weight.data.abs() > threshold).float()

        # Apply to BN parameters
        module.weight.data.mul_(mask)
        module.bias.data.mul_(mask)
        module.running_mean.data.mul_(mask)
        module.running_var.data.mul_(mask)

        # Count
        layer_total = mask.numel()
        layer_pruned = (mask == 0).sum().item()
        total_channels += layer_total
        pruned_channels += layer_pruned

        print(f"  {name}: {layer_pruned}/{layer_total} pruned")

    compression = (pruned_channels / total_channels) * 100
    print(".1f"
    return compression

def main():
    # Configuration
    MODEL_PATH = "Object-Detection/Yolo-V8/weights/best.pt"
    DATA_PATH = "datasets/VisDrone/VisDrone.yaml"  # Adjust path
    PRUNE_RATIO = 0.3  # 30% pruning
    OUTPUT_PATH = "pruned_yolo_model.pt"

    print("YOLOv8 Pruning Pipeline")
    print("=" * 50)

    # Load model
    model = load_model_safely(MODEL_PATH)
    if model is None:
        print("❌ Could not load model. Please check the path.")
        return

    # Apply pruning
    compression = prune_yolo_model(model, PRUNE_RATIO)

    # Save pruned model
    model.save(OUTPUT_PATH)
    print(f"\n✓ Pruned model saved to: {OUTPUT_PATH}")

    # Optional: Quick validation
    print("\n=== Quick Validation ===")
    try:
        results = model.val(data=DATA_PATH, split='val')
        print("Validation completed!")
    except Exception as e:
        print(f"Validation failed: {e}")

    print("\n=== Summary ===")
    print(".1f"    print(f"Model saved: {OUTPUT_PATH}"))
    print("\nNext steps:")
    print("1. Fine-tune the pruned model: model.train(data=DATA_PATH, epochs=50)")
    print("2. Test inference speed improvement")
    print("3. Compare mAP with original model")

if __name__ == "__main__":
    main()