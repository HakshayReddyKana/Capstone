#!/usr/bin/env python3
"""
Simple YOLOv8 Fine-tuning Script - No multiprocessing issues
"""

import torch
from ultralytics import YOLO

def main():
    print("Loading pruned model...")
    model = YOLO('pruned_yolo_model.pt')

    print("Starting fine-tuning with safe settings...")

    # Use minimal settings to avoid DataLoader issues
    results = model.train(
        data='yolov8-prune/ultralytics/cfg/datasets/VisDrone.yaml',
        epochs=5,  # Short training for testing
        batch=2,   # Very small batch size
        lr0=1e-4,
        patience=5,
        project='finetune_results',
        name='pruned_finetune_safe',
        save=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        workers=0,  # Disable multiprocessing
        exist_ok=True  # Allow overwriting
    )

    print("✓ Fine-tuning completed!")
    print(f"Best model saved at: {results.save_dir}")

    # Validate the fine-tuned model
    print("\nValidating fine-tuned model...")
    val_results = model.val(
        data='yolov8-prune/ultralytics/cfg/datasets/VisDrone.yaml',
        split='val',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("Fine-tuned model results:")
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")

if __name__ == "__main__":
    main()