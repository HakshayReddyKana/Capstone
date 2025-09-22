#!/usr/bin/env python3
"""
YOLOv8 Fine-tuning Script for Pruned Models
"""

import torch
from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser(description='Fine-tune pruned YOLOv8 model')
    parser.add_argument('--weights', type=str, default='pruned_yolo_model.pt', help='pruned model weights path')
    parser.add_argument('--data', type=str, default='yolov8-prune/ultralytics/cfg/datasets/VisDrone.yaml', help='dataset config')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    args = parser.parse_args()

    print("Loading pruned model...")
    model = YOLO(args.weights)

    print("Starting fine-tuning...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        lr0=args.lr,
        patience=10,
        project='finetune_results',
        name='pruned_finetune',
        save=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        workers=0  # Disable multiprocessing to avoid DataLoader worker issues
    )

    print("✓ Fine-tuning completed!")
    print(f"Best model saved at: {results.save_dir}")

if __name__ == "__main__":
    main()