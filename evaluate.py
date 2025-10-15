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
DATA_ROOT = "data/ft_data/test"  # thư mục test: data/test/<class_name>/*.jpg
YOLO_WEIGHTS = "ckpt/yolom.pt"  # YOLO đã train
RESNET_WEIGHTS = "ckpt/symp_151025.pth"       # ResNet50 đã train (n_class khớp)
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # Set to 0 to avoid CUDA multiprocessing issues with YOLO
YOLO_CONF = 0.25
YOLO_IOU = 0.45
YOLO_CLASSES = None   # ví dụ: [0, 3] nếu chỉ muốn lấy bbox từ các lớp này; None = tất cả
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2) Load YOLO (Ultralytics)
from ultralytics import YOLO
yolo = YOLO(YOLO_WEIGHTS)

# 3) Biến đổi cho ResNet50
cls_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet
                         std=[0.229, 0.224, 0.225]),
])

# 4) Dataset: đọc ảnh gốc, cắt bằng YOLO -> trả về crop cho ResNet
class YoloCropEvalDS(Dataset):
    def __init__(self, root: str, transform, yolo_model: YOLO,
                 yolo_conf=0.25, yolo_iou=0.45, yolo_classes=None):
        self.samples = []  # (img_path, label_idx)
        self.class_to_idx = {}
        self.transform = transform
        self.yolo = yolo_model
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou
        self.yolo_classes = yolo_classes

        root = Path(root)
        classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            for p in (root / c).glob("*.*"):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((str(p), self.class_to_idx[c]))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _safe_crop(img: Image.Image, xyxy: np.ndarray, pad_ratio: float = 0.02):
        """Crop có padding nhẹ để tránh cắt hụt viền."""
        w, h = img.size
        x1, y1, x2, y2 = xyxy
        # padding theo cạnh ngắn
        pad = int(pad_ratio * min(w, h))
        x1 = max(0, int(x1) - pad)
        y1 = max(0, int(y1) - pad)
        x2 = min(w, int(x2) + pad)
        y2 = min(h, int(y2) + pad)
        return img.crop((x1, y1, x2, y2))

    def _detect_best_crop(self, img: Image.Image) -> Image.Image:
        # YOLO nhận ndarray/BGR hoặc path; ở đây truyền PIL → tự chuyển
        res = self.yolo.predict(
            img, conf=self.yolo_conf, iou=self.yolo_iou,
            classes=self.yolo_classes, verbose=False
        )
        r = res[0]
        if r.boxes is None or len(r.boxes) == 0:
            # Không phát hiện: fallback = resize ảnh gốc
            return img

        boxes = r.boxes
        conf = boxes.conf.cpu().numpy()  # (N,)
        xyxy = boxes.xyxy.cpu().numpy()  # (N, 4) theo toạ độ ảnh gốc

        # Lấy bbox có confidence cao nhất
        best_idx = int(conf.argmax())
        crop = self._safe_crop(img, xyxy[best_idx])
        return crop

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        # detect & crop
        crop = self._detect_best_crop(img)

        # transform cho ResNet
        x = self.transform(crop)
        return x, label, img_path

# 5) Model ResNet50 (n_classes tự suy ra từ folder)
# Architecture must match train.py's CoffeeLeafClassifier
def build_resnet50(n_classes: int) -> nn.Module:
    """Build ResNet50 classifier matching train.py architecture"""
    class CoffeeLeafClassifier(nn.Module):
        def __init__(self, num_classes=6):
            super(CoffeeLeafClassifier, self).__init__()
            # Encoder: ResNet50 without last 2 layers (no avgpool, no fc)
            resnet = models.resnet50(weights=None)  # Use weights=None instead of pretrained=False
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

# Chuẩn bị dataset/loader
tmp_ds = YoloCropEvalDS(DATA_ROOT, cls_transform, yolo, YOLO_CONF, YOLO_IOU, YOLO_CLASSES)
n_classes = len(tmp_ds.class_to_idx)
print("Classes:", tmp_ds.class_to_idx)

loader = DataLoader(tmp_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# 6) Load ResNet50 đã train & evaluate
model = build_resnet50(n_classes).to(DEVICE)
checkpoint = torch.load(RESNET_WEIGHTS, map_location=DEVICE)

# Extract model_state_dict from checkpoint
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val Acc: {checkpoint.get('val_acc', 'N/A'):.4f}" if 'val_acc' in checkpoint else "")
elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing:
    print("Missing keys:", missing[:5], "..." if len(missing) > 5 else "")
if unexpected:
    print("Unexpected keys:", unexpected[:5], "..." if len(unexpected) > 5 else "")
model.eval()

correct, total = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for x, y, _ in tqdm(loader, desc="Evaluating ResNet50 on YOLO-crops"):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        all_preds.append(pred.cpu())
        all_labels.append(y.cpu())

acc = correct / max(total, 1)
print(f"Accuracy: {acc:.4f}")

# (tuỳ chọn) Confusion matrix & classification report (cần scikit-learn)
try:
    import sklearn.metrics as skm
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    cm = skm.confusion_matrix(labels, preds, labels=list(range(n_classes)))
    print("Confusion matrix:\n", cm)
    print(skm.classification_report(labels, preds, target_names=[k for k,_ in sorted(tmp_ds.class_to_idx.items(), key=lambda x: x[1])]))
except Exception as e:
    print("Bỏ qua sklearn report:", e)
