import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Make sure plots directory exists
os.makedirs('plots', exist_ok=True)

# ------------ DATA ------------
classes = [
    'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
    'tricycle', 'awning-tricycle', 'bus', 'motor'
]
pruned_precision = [0.177, 0.349, 0, 0.325, 0.114, 0.104, 0.066, 1, 0.26, 0.256]
v8_precision = [0.287, 0.376, 0.05, 0.439, 0.275, 0.284, 0.158, 0.203, 0.438, 0.198]
v10_precision = [0.183, 0.335, 0.073, 0.373, 0.199, 0.319, 0.363, 0.472, 0.268, 0.255]
v11_precision = [0.186, 0.391, 0.047, 0.488, 0.224, 0.252, 0.183, 0.105, 0.382, 0.264]

pruned_f1 = [0.196, 0.166, 0, 0.429, 0.16, 0.094, 0.002, 0, 0.198, 0.224]
v8_f1 = [0.278, 0.196, 0.04, 0.539, 0.261, 0.238, 0.143, 0.034, 0.299, 0.247]
v10_f1 = [0.239, 0.206, 0.02, 0.485, 0.216, 0.197, 0.074, 0.007, 0.255, 0.263]
v11_f1 = [0.243, 0.198, 0.013, 0.566, 0.254, 0.218, 0.098, 0.012, 0.299, 0.258]

# Detection summary
models = ['YOLOv8-Pruned (Fine-tuned)', 'YOLOv8 Original']
detection_time = [0.1188, 0.3641]
objects_detected = [21, 18]

# ------------ Precision (Pruned Highlight) ------------

# Sort classes by pruned_precision descending
sort_idx = np.argsort(pruned_precision)[::-1]
sorted_classes = np.array(classes)[sort_idx]
pruned_sorted = np.array(pruned_precision)[sort_idx]
v8_sorted = np.array(v8_precision)[sort_idx]
v10_sorted = np.array(v10_precision)[sort_idx]
v11_sorted = np.array(v11_precision)[sort_idx]

plt.figure(figsize=(12,6))
plt.plot(sorted_classes, pruned_sorted, label='YOLOv8-Pruned', color='gold', linewidth=4, marker='o')
plt.plot(sorted_classes, v8_sorted, label='YOLOv8', color='gray', linestyle='--', marker='x', alpha=0.7)
plt.plot(sorted_classes, v10_sorted, label='YOLOv10', color='lightgray', linestyle=':', marker='s', alpha=0.6)
plt.plot(sorted_classes, v11_sorted, label='YOLOv11', color='lightgray', linestyle='-.', marker='d', alpha=0.6)
plt.title('Precision: ')
plt.xlabel('Class Name (Sorted by Pruned)')
plt.ylabel('Precision')
plt.legend()
plt.tight_layout()
plt.savefig('plots/precision_pruned_focus.png')
plt.close()

# ------------ F1 Score (Pruned Highlight) ------------

pruned_f1_sorted = np.array(pruned_f1)[sort_idx]
v8_f1_sorted = np.array(v8_f1)[sort_idx]
v10_f1_sorted = np.array(v10_f1)[sort_idx]
v11_f1_sorted = np.array(v11_f1)[sort_idx]

plt.figure(figsize=(12,6))
plt.plot(sorted_classes, pruned_f1_sorted, label='YOLOv8-Pruned', color='gold', linewidth=4, marker='o')
plt.plot(sorted_classes, v8_f1_sorted, label='YOLOv8', color='gray', linestyle='--', marker='x', alpha=0.7)
plt.plot(sorted_classes, v10_f1_sorted, label='YOLOv10', color='lightgray', linestyle=':', marker='s', alpha=0.6)
plt.plot(sorted_classes, v11_f1_sorted, label='YOLOv11', color='lightgray', linestyle='-.', marker='d', alpha=0.6)
plt.title('F1 Score: ')
plt.xlabel('Class Name (Sorted by Pruned)')
plt.ylabel('F1 Score')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f1_pruned_focus.png')
plt.close()

# ------------ Detection Time & Objects Radar Chart ------------

labels=np.array(['Detection Time', 'Objects Detected'])
stats = np.array([
    [0.1188, 21],     # Pruned & Fine-tuned
    [0.3641, 18],     # Original
])

# Normalize for radar (so scales are similar)
stats_norm = np.array([
    [stats[0,0]/stats[:,0].max(), stats[0,1]/stats[:,1].max()],
    [stats[1,0]/stats[:,0].max(), stats[1,1]/stats[:,1].max()]
])

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(6,6))
ax = plt.subplot(111, polar=True)
pruned_values = np.append(stats_norm[0], stats_norm[0][0])
orig_values = np.append(stats_norm[1], stats_norm[1][0])
ax.plot(angles, pruned_values, color='gold', linewidth=3, marker='o', label='YOLOv8-Pruned')
ax.fill(angles, pruned_values, color='gold', alpha=0.4)
ax.plot(angles, orig_values, color='gray', linestyle='--', marker='s', label='YOLOv8 Original')
ax.fill(angles, orig_values, color='gray', alpha=0.2)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_yticklabels([])
plt.title('Detection Time & Objects Detected (Normalized)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('plots/radar_detection_pruned.png')
plt.close()

# ------------ Speed/Objects Bar Chart (Pruned Emphasis) ------------

fig, ax1 = plt.subplots(figsize=(6,5))

color = 'gold'
ax1.bar(models[0], detection_time[0], color=color, alpha=0.8, label="Detection Time (Pruned)")
ax1.bar(models[1], detection_time[1], color='lightgray', alpha=0.7, label="Detection Time (Original)")
ax1.set_ylabel('Detection Time (s)', color='black')
ax1.set_title('Detection Time')
ax1.set_ylim(0, max(detection_time) * 1.3)  # Increase height range by 30%

for i, v in enumerate(detection_time):
    ax1.text(i, v + 0.01, f"{v:.3f}s", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('plots/bar_detection_time_pruned.png')
plt.close()

fig, ax2 = plt.subplots(figsize=(6,5))
ax2.bar(models[0], objects_detected[0], color=color, alpha=0.8, label="Objects Detected (Pruned)")
ax2.bar(models[1], objects_detected[1], color='lightgray', alpha=0.7, label="Objects Detected (Original)")
ax2.set_ylabel('Objects Detected', color='black')
ax2.set_title('Objects Detected ')

for i, v in enumerate(objects_detected):
    ax2.text(i, v + 0.5, f"{v}", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('plots/bar_objects_detected_pruned.png')
plt.close()
