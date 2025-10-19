#!/usr/bin/env python3
"""
Evaluation Script: ResNet50 on Pre-cropped Test Data
Evaluates the trained ResNet50 model on pre-cropped test images
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# 1) Cấu hình
DATA_ROOT = "data/test4"  # thư mục test đã crop sẵn: data/cropped_test/<class_name>/*.jpg
RESNET_WEIGHTS = "ckpt/final_model4.pth"  # ResNet50 đã train
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2) Biến đổi cho ResNet50
cls_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet
                         std=[0.229, 0.224, 0.225]),
])

# 3) Dataset: đọc ảnh đã crop sẵn
class PreCroppedEvalDS(Dataset):
    def __init__(self, root: str, transform):
        self.samples = []  # (img_path, label_idx)
        self.class_to_idx = {}
        self.transform = transform

        root = Path(root)
        # Use the same class order as train.py
        classes = ['Cercospora', 'Corticium', 'mealy', 'Miner', 'Phoma', 'Rust']
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        
        for c in classes:
            class_dir = root / c
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist!")
                continue
                
            for p in class_dir.glob("*.*"):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((str(p), self.class_to_idx[c]))
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root}. Please check the directory structure.")
        
        print(f"Found {len(self.samples)} images")
        
        # Print class distribution
        class_counts = {c: 0 for c in classes}
        for _, label in self.samples:
            class_counts[classes[label]] += 1
        
        print("Class distribution:")
        for cls, count in class_counts.items():
            print(f"  {cls:12s}: {count:4d} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Transform
        x = self.transform(img)
        return x, label, img_path

# 4) Model ResNet50 - Architecture must match train.py's CoffeeLeafClassifier
def build_resnet50(n_classes: int) -> nn.Module:
    """Build ResNet50 classifier matching train.py architecture"""
    class CoffeeLeafClassifier(nn.Module):
        def __init__(self, num_classes=6):
            super(CoffeeLeafClassifier, self).__init__()
            # Encoder: ResNet50 without last 2 layers (no avgpool, no fc)
            resnet = models.resnet50(weights=None)
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            
            # Add adaptive pooling and classifier head
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            
            # Custom classifier head
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = self.classifier(x)
            return x
    
    return CoffeeLeafClassifier(num_classes=n_classes)

print(f"{'='*60}")
print(f"ResNet50 Evaluation on Pre-Cropped Test Data")
print(f"{'='*60}\n")
print(f"Device: {DEVICE}")
print(f"Data root: {DATA_ROOT}")
print(f"Checkpoint: {RESNET_WEIGHTS}\n")

# Chuẩn bị dataset/loader
print("Loading dataset...")
tmp_ds = PreCroppedEvalDS(DATA_ROOT, cls_transform)
n_classes = len(tmp_ds.class_to_idx)
print(f"\nClasses: {tmp_ds.class_to_idx}\n")

loader = DataLoader(tmp_ds, batch_size=BATCH_SIZE, shuffle=False, 
                   num_workers=NUM_WORKERS, pin_memory=True)

# 5) Load ResNet50 & evaluate
print("Loading model...")
model = build_resnet50(n_classes).to(DEVICE)

# Check if checkpoint exists
if not os.path.exists(RESNET_WEIGHTS):
    print(f"❌ Error: Checkpoint not found at {RESNET_WEIGHTS}")
    print("Please check the path and try again.")
    exit(1)

checkpoint = torch.load(RESNET_WEIGHTS, map_location=DEVICE)

# Extract model_state_dict from checkpoint
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_acc' in checkpoint:
        print(f"  Validation Accuracy: {checkpoint.get('val_acc', 0):.4f}")
elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# Load weights
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing:
    print(f"⚠ Warning: Missing keys in checkpoint (showing first 5): {missing[:5]}")
if unexpected and len(unexpected) > 0 and 'epoch' not in unexpected:
    print(f"⚠ Warning: Unexpected keys in checkpoint (showing first 5): {unexpected[:5]}")

print("✓ Model loaded successfully\n")
model.eval()

# Evaluate
print("Evaluating...")
correct, total = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for x, y, _ in tqdm(loader, desc="Evaluating ResNet50"):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        all_preds.append(pred.cpu())
        all_labels.append(y.cpu())

acc = correct / max(total, 1)
print(f"\n{'='*60}")
print(f"Overall Accuracy: {acc:.4f} ({correct}/{total})")
print(f"{'='*60}\n")

# Detailed metrics with sklearn
try:
    import sklearn.metrics as skm
    from datetime import datetime
    import json
    
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    
    # Get class names in correct order
    class_names = [k for k, _ in sorted(tmp_ds.class_to_idx.items(), key=lambda x: x[1])]
    
    # Confusion matrix
    cm = skm.confusion_matrix(labels, preds, labels=list(range(n_classes)))
    print("Confusion Matrix:")
    print(cm)
    print()
    
    # Classification report
    report = skm.classification_report(labels, preds, target_names=class_names, output_dict=True)
    print(skm.classification_report(labels, preds, target_names=class_names))
    
    # Calculate and display average metrics
    macro_prec = report['macro avg']['precision']
    macro_rec = report['macro avg']['recall']
    macro_f1 = report['macro avg']['f1-score']
    weighted_prec = report['weighted avg']['precision']
    weighted_rec = report['weighted avg']['recall']
    weighted_f1 = report['weighted avg']['f1-score']
    
    print(f"\n{'='*60}")
    print(f"Average Metrics:")
    print(f"  Macro Avg    - Precision: {macro_prec:.4f}, Recall: {macro_rec:.4f}, F1: {macro_f1:.4f}")
    print(f"  Weighted Avg - Precision: {weighted_prec:.4f}, Recall: {weighted_rec:.4f}, F1: {weighted_f1:.4f}")
    print(f"{'='*60}\n")
    
    # Create output directory
    output_dir = Path("evaluation_results_cropped")
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save confusion matrix as text file
    cm_file = output_dir / f"confusion_matrix_{timestamp}.txt"
    with open(cm_file, 'w') as f:
        f.write("Confusion Matrix\n")
        f.write("=" * 60 + "\n\n")
        f.write("Classes: " + str(class_names) + "\n\n")
        f.write(str(cm) + "\n\n")
        f.write("Row: True label, Column: Predicted label\n")
    print(f"✓ Confusion matrix saved to: {cm_file}")
    
    # Save confusion matrix as numpy file
    cm_npy_file = output_dir / f"confusion_matrix_{timestamp}.npy"
    np.save(cm_npy_file, cm)
    print(f"✓ Confusion matrix (numpy) saved to: {cm_npy_file}")
    
    # Save detailed metrics per class
    metrics_file = output_dir / f"per_class_metrics_{timestamp}.txt"
    with open(metrics_file, 'w') as f:
        f.write("Per-Class Evaluation Metrics (Pre-Cropped Test Data)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy: {acc:.4f} ({correct}/{total})\n\n")
        f.write("Average Metrics:\n")
        f.write(f"  Macro Avg    - Precision: {macro_prec:.4f}, Recall: {macro_rec:.4f}, F1: {macro_f1:.4f}\n")
        f.write(f"  Weighted Avg - Precision: {weighted_prec:.4f}, Recall: {weighted_rec:.4f}, F1: {weighted_f1:.4f}\n\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-" * 60 + "\n")
        
        for class_name in class_names:
            metrics = report[class_name]
            f.write(f"{class_name:<15} "
                   f"{metrics['precision']:<12.4f} "
                   f"{metrics['recall']:<12.4f} "
                   f"{metrics['f1-score']:<12.4f} "
                   f"{int(metrics['support']):<10}\n")
        
        f.write("-" * 60 + "\n")
        f.write(f"{'Macro Avg':<15} "
               f"{report['macro avg']['precision']:<12.4f} "
               f"{report['macro avg']['recall']:<12.4f} "
               f"{report['macro avg']['f1-score']:<12.4f} "
               f"{int(report['macro avg']['support']):<10}\n")
        f.write(f"{'Weighted Avg':<15} "
               f"{report['weighted avg']['precision']:<12.4f} "
               f"{report['weighted avg']['recall']:<12.4f} "
               f"{report['weighted avg']['f1-score']:<12.4f} "
               f"{int(report['weighted avg']['support']):<10}\n")
    
    print(f"✓ Per-class metrics saved to: {metrics_file}")
    
    # Save as JSON for easy parsing
    json_file = output_dir / f"evaluation_results_{timestamp}.json"
    results = {
        'timestamp': timestamp,
        'data_source': 'pre-cropped test data',
        'checkpoint': RESNET_WEIGHTS,
        'overall_accuracy': float(acc),
        'total_samples': int(total),
        'correct_predictions': int(correct),
        'classes': class_names,
        'per_class_metrics': {
            class_name: {
                'precision': float(report[class_name]['precision']),
                'recall': float(report[class_name]['recall']),
                'f1_score': float(report[class_name]['f1-score']),
                'support': int(report[class_name]['support'])
            }
            for class_name in class_names
        },
        'macro_avg': {
            'precision': float(report['macro avg']['precision']),
            'recall': float(report['macro avg']['recall']),
            'f1_score': float(report['macro avg']['f1-score'])
        },
        'weighted_avg': {
            'precision': float(report['weighted avg']['precision']),
            'recall': float(report['weighted avg']['recall']),
            'f1_score': float(report['weighted avg']['f1-score'])
        },
        'confusion_matrix': cm.tolist()
    }
    
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ JSON results saved to: {json_file}")
    print(f"\n{'='*60}")
    print(f"All evaluation results saved to: {output_dir}/")
    print(f"{'='*60}")
    
except ImportError as e:
    print(f"⚠ sklearn not available: {e}")
    print("Install with: pip install scikit-learn")
except Exception as e:
    print(f"⚠ Error generating reports: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Evaluation completed!")
