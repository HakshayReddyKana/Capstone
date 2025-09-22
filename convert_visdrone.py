"""
Convert VisDrone annotations to YOLO format
Based on the visdrone2yolo function from yolov8-prune
"""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def convert_box(size, box):
    """Convert VisDrone box to YOLO xywh box (normalized)"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

def visdrone2yolo(dir_path):
    """Convert VisDrone annotations to YOLO labels"""
    dir_path = Path(dir_path)

    # Create labels directory if it doesn't exist
    labels_dir = dir_path / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting annotations in {dir_path}")
    print(f"Annotations: {dir_path / 'annotations'}")
    print(f"Labels: {labels_dir}")

    # Get all annotation files
    annotation_files = list((dir_path / 'annotations').glob('*.txt'))

    if not annotation_files:
        print(f"No annotation files found in {dir_path / 'annotations'}")
        return

    print(f"Found {len(annotation_files)} annotation files")

    pbar = tqdm(annotation_files, desc=f'Converting {dir_path.name}')
    for f in pbar:
        # Get corresponding image to get dimensions
        img_path = dir_path / 'images' / f.name.replace('.txt', '.jpg')
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue

        try:
            img_size = Image.open(img_path).size
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            continue

        lines = []
        try:
            with open(f, 'r') as file:
                for row in [x.split(',') for x in file.read().strip().splitlines()]:
                    if len(row) < 6:
                        continue

                    if row[4] == '0':  # VisDrone 'ignored regions' class 0
                        continue

                    try:
                        cls = int(row[5]) - 1  # VisDrone classes start from 1, YOLO from 0
                        box = convert_box(img_size, tuple(map(int, row[:4])))
                        lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing row {row}: {e}")
                        continue

        except Exception as e:
            print(f"Error reading annotation file {f}: {e}")
            continue

        # Write YOLO format labels
        label_file = labels_dir / f.name
        try:
            with open(label_file, 'w') as fl:
                fl.writelines(lines)
        except Exception as e:
            print(f"Error writing label file {label_file}: {e}")

    print(f"Conversion completed for {dir_path.name}")

def main():
    """Convert all VisDrone datasets"""
    base_path = Path(r"c:\Users\haksh\Documents\CALSS MATERIALS\SEM7\Capstone\datasets\VisDrone")

    datasets = [
        'VisDrone2019-DET-train',
        'VisDrone2019-DET-val',
        'VisDrone2019-DET-test-dev'
    ]

    for dataset in datasets:
        dataset_path = base_path / dataset
        if dataset_path.exists():
            print(f"\n{'='*50}")
            print(f"Processing {dataset}")
            print('='*50)
            visdrone2yolo(dataset_path)
        else:
            print(f"Dataset not found: {dataset_path}")

    print("\n✅ VisDrone to YOLO conversion completed!")
    print("You can now run validation on your models.")

if __name__ == "__main__":
    main()