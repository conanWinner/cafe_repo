#!/usr/bin/env python3
"""
YOLO Training: DPCL-BRACOL Coffee Leaf Disease Detection

Dataset: DPCL-BRACOL-COMPLETE v5
Classes: Cercospora, Miner, Phoma, Rust
"""

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import json
import pandas as pd
import numpy as np
import random

# ==========================================
# 1. SETUP & LOAD DATASET
# ==========================================
print("="*70)
print("🍃 YOLO TRAINING: DPCL-BRACOL COFFEE DISEASE DETECTION")
print("="*70)

# Dataset path
dataset_path = Path("../DPCL-BRACOL-COMPLETE.v5i.yolov11")
yaml_file = dataset_path / "data.yaml"

# Load config
with open(yaml_file, 'r') as f:
    config = yaml.safe_load(f)

print(f"\n📊 DATASET INFORMATION")
print("="*70)
print(f"📁 Path: {dataset_path}")
print(f"🏷️  Classes ({config['nc']}): {config['names']}")

# Count images
train_imgs = list((dataset_path / "train" / "images").glob("*"))
val_imgs = list((dataset_path / "valid" / "images").glob("*"))
test_imgs = list((dataset_path / "test" / "images").glob("*"))

print(f"\n📈 Dataset Split:")
print(f"  • Train:      {len(train_imgs):,} images")
print(f"  • Validation: {len(val_imgs):,} images")
print(f"  • Test:       {len(test_imgs):,} images")
print(f"  • Total:      {len(train_imgs) + len(val_imgs) + len(test_imgs):,} images")

# ==========================================
# 2. LOAD MODEL
# ==========================================
print(f"\n🤖 Loading YOLO model...")

MODEL_SIZE = 'n'  # n, s, m, l, x

# Check task type
sample_label = list((dataset_path / "train" / "labels").glob("*.txt"))[0]
with open(sample_label, 'r') as f:
    is_seg = len(f.readline().split()) > 5

model_name = f'yolo11{MODEL_SIZE}-seg.pt' if is_seg else f'yolo11{MODEL_SIZE}.pt'
model = YOLO(model_name)

task = 'Segmentation' if is_seg else 'Detection'
print(f"  Task: {task}")
print(f"  Model: {model_name}")
print(f"  Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

# ==========================================
# 3. TRAINING CONFIGURATION
# ==========================================
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
PATIENCE = 30
DEVICE = 0

print(f"\n⚙️  Training Configuration:")
print(f"  • Epochs: {EPOCHS}")
print(f"  • Batch: {BATCH_SIZE}")
print(f"  • Image size: {IMAGE_SIZE}")
print(f"  • Early stopping: {PATIENCE}")

# ==========================================
# 4. TRAIN MODEL
# ==========================================
print(f"\n🔥 Starting training...\n")

results = model.train(
    data=str(yaml_file),
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    project='runs/dpcl_bracol',
    name='coffee_disease_detection',
    patience=PATIENCE,
    save=True,
    plots=True,
    verbose=True
)

best_model_path = model.trainer.best
print(f"\n✅ Training completed!")
print(f"📁 Best model: {best_model_path}")

# ==========================================
# 5. EVALUATE MODEL
# ==========================================
print(f"\n📊 Evaluating model...")

best_model = YOLO(best_model_path)
metrics = best_model.val()

print(f"\n📈 EVALUATION METRICS")
print("="*70)
print(f"Box Detection:")
print(f"  • mAP@0.5:      {metrics.box.map50:.4f}")
print(f"  • mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"  • Precision:    {metrics.box.mp:.4f}")
print(f"  • Recall:       {metrics.box.mr:.4f}")

if hasattr(metrics, 'seg') and metrics.seg:
    print(f"\nSegmentation:")
    print(f"  • mAP@0.5:      {metrics.seg.map50:.4f}")
    print(f"  • mAP@0.5:0.95: {metrics.seg.map:.4f}")

print(f"\nPer-Class mAP@0.5:")
for i, name in enumerate(config['names']):
    if i < len(metrics.box.ap50):
        print(f"  • {name:12s}: {metrics.box.ap50[i]:.4f}")

# ==========================================
# 6. DEMO INFERENCE - TEST THỬ MODEL
# ==========================================
print(f"\n🎯 Testing model on validation images...")

# Chọn ngẫu nhiên 6 ảnh validation
num_samples = min(6, len(val_imgs))
test_samples = random.sample(val_imgs, num_samples)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for idx, img_path in enumerate(test_samples):
    # Inference
    results = best_model(img_path, verbose=False)
    result = results[0]
    
    # Annotate image
    annotated = result.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    # Plot
    axes[idx].imshow(annotated_rgb)
    n_detections = len(result.boxes)
    axes[idx].set_title(f"{img_path.name}\n{n_detections} diseases detected", fontsize=10)
    axes[idx].axis('off')

plt.suptitle('🔍 Model Predictions - Coffee Disease Detection', fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig('./dpcl_bracol_outputs/predictions_demo.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Visualization saved: dpcl_bracol_outputs/predictions_demo.png")

# ==========================================
# 7. DETAILED PREDICTION ON ONE IMAGE
# ==========================================
print(f"\n📋 Detailed prediction example:")
print("="*70)

test_img = random.choice(test_imgs)
results = best_model(test_img, verbose=False)
result = results[0]

print(f"\n🖼️  Image: {test_img.name}")
print(f"📦 Detected: {len(result.boxes)} objects\n")

if len(result.boxes) > 0:
    for i, (box, cls, conf) in enumerate(zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf)):
        disease = result.names[int(cls)]
        x1, y1, x2, y2 = [float(x) for x in box]
        print(f"  Detection {i+1}:")
        print(f"    • Disease: {disease}")
        print(f"    • Confidence: {conf:.2%}")
        print(f"    • Bbox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        print(f"    • Size: {x2-x1:.0f}x{y2-y1:.0f} pixels")
        print()
else:
    print("  ✅ No diseases detected (healthy leaf)")

# Save this prediction
output_path = Path("./dpcl_bracol_outputs/detailed_prediction.jpg")
result.save(str(output_path))
print(f"💾 Saved annotated image: {output_path}")

# ==========================================
# 8. EXPORT DETECTION RESULTS (BATCH)
# ==========================================
print(f"\n📄 Exporting detection results for all test images...")

output_dir = Path("./dpcl_bracol_outputs")
output_dir.mkdir(exist_ok=True)

all_detections = []

for img_path in test_imgs[:20]:  # First 20 test images
    results = best_model(img_path, verbose=False)
    result = results[0]
    
    for i, (box, cls, conf) in enumerate(zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf)):
        all_detections.append({
            'image': img_path.name,
            'object_id': i,
            'class': result.names[int(cls)],
            'confidence': float(conf),
            'x1': float(box[0]),
            'y1': float(box[1]),
            'x2': float(box[2]),
            'y2': float(box[3]),
            'width': float(box[2] - box[0]),
            'height': float(box[3] - box[1])
        })

# Save CSV
df = pd.DataFrame(all_detections)
csv_file = output_dir / "detections.csv"
df.to_csv(csv_file, index=False)

# Save JSON
json_file = output_dir / "detections.json"
with open(json_file, 'w') as f:
    json.dump(all_detections, f, indent=2)

print(f"  ✅ CSV: {csv_file}")
print(f"  ✅ JSON: {json_file}")
print(f"  📊 Total: {len(all_detections)} detections")

# Stats by class
if len(all_detections) > 0:
    print(f"\n📊 Detections by class:")
    for name, count in df['class'].value_counts().items():
        print(f"  • {name:12s}: {count:4d}")

# ==========================================
# 7. EXPORT MODEL
# ==========================================
print(f"\n📦 Exporting model...")

onnx_path = best_model.export(format='onnx')
print(f"  ✅ ONNX: {onnx_path}")

# ==========================================
# 10. FUNCTION ĐỂ TEST TRÊN ẢNH MỚI
# ==========================================
def predict_coffee_disease(image_path, model_path=None, save_result=True):
    """
    Test model trên ảnh mới
    
    Args:
        image_path: Đường dẫn ảnh
        model_path: Đường dẫn model (None = dùng best_model hiện tại)
        save_result: Lưu kết quả
    
    Returns:
        List of detections
    """
    # Load model
    if model_path:
        model = YOLO(model_path)
    else:
        model = best_model
    
    # Inference
    results = model(image_path, verbose=False)
    result = results[0]
    
    # Display
    annotated = result.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_rgb)
    plt.title(f"Coffee Disease Detection: {Path(image_path).name}", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print results
    detections = []
    print(f"\n{'='*60}")
    print(f"📋 Results for: {Path(image_path).name}")
    print(f"{'='*60}\n")
    
    if len(result.boxes) == 0:
        print("✅ No diseases detected - Leaf appears healthy!")
    else:
        print(f"⚠️  Detected {len(result.boxes)} disease(s):\n")
        for i, (box, cls, conf) in enumerate(zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf)):
            disease = result.names[int(cls)]
            detections.append({
                'disease': disease,
                'confidence': float(conf),
                'bbox': [float(x) for x in box]
            })
            print(f"  {i+1}. 🦠 {disease}")
            print(f"     Confidence: {conf:.1%}")
            print(f"     Location: ({box[0]:.0f}, {box[1]:.0f}) to ({box[2]:.0f}, {box[3]:.0f})")
            print()
    
    # Save
    if save_result:
        output_path = output_dir / f"prediction_{Path(image_path).name}"
        result.save(str(output_path))
        print(f"💾 Saved to: {output_path}")
    
    return detections

print(f"\n✅ Function 'predict_coffee_disease()' defined!")
print(f"\n💡 Usage:")
print(f"   predict_coffee_disease('path/to/image.jpg')")

# ==========================================
# SUMMARY
# ==========================================
print(f"\n" + "="*70)
print("🎉 TRAINING & TESTING COMPLETE!")
print("="*70)
print(f"\n📁 Outputs:")
print(f"  • PyTorch model: {best_model_path}")
print(f"  • ONNX model: {onnx_path}")
print(f"  • Detections CSV: {csv_file}")
print(f"  • Predictions demo: dpcl_bracol_outputs/predictions_demo.png")
print(f"  • Training plots: {Path(best_model_path).parent.parent}")
print(f"\n💡 Test trên ảnh mới:")
print(f"  predict_coffee_disease('your_image.jpg')")
print(f"\n💡 Load model sau này:")
print(f"  from ultralytics import YOLO")
print(f"  model = YOLO('{best_model_path}')")
print(f"  results = model('image.jpg')")
print(f"  results[0].show()")
print("="*70)

