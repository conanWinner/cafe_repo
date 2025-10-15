# Coffee Leaf Disease Classification - Training Guide

## Mô tả

Script `train.py` được thiết kế để:
1. **Load pretrained checkpoint** từ `ckpt/resnet50_ssl_141025.pt`
2. **Fine-tune trên ft_data** với 6 classes: Cercospora, Corticium, mealy, Miner, Phoma, Rust
3. **Lưu kết quả và checkpoints** sau khi training
4. **KHÔNG chạy test set** (chỉ train và validation)

## Cấu trúc dữ liệu

```
data/ft_data/
├── train/
│   ├── Cercospora/    (233 images)
│   ├── Corticium/     (254 images)
│   ├── mealy/         (146 images)
│   ├── Miner/         (369 images)
│   ├── Phoma/         (399 images)
│   └── Rust/          (1414 images)
└── valid/
    ├── Cercospora/
    ├── Corticium/
    ├── mealy/
    ├── Miner/
    ├── Phoma/
    └── Rust/
```

## Cài đặt

```bash
# Cài đặt các thư viện cần thiết
pip install torch torchvision tqdm matplotlib pillow numpy
```

## Chạy training

```bash
# Chạy training với cấu hình mặc định
python train.py
```

## Cấu hình

Các tham số có thể chỉnh sửa trong biến `CONFIG` ở đầu file `train.py`:

```python
CONFIG = {
    'data_root': 'data/ft_data',              # Thư mục chứa dữ liệu
    'checkpoint_path': 'ckpt/resnet50_ssl_141025.pt',  # Pretrained checkpoint
    'output_dir': 'outputs',                   # Thư mục lưu kết quả
    'batch_size': 32,                          # Batch size
    'num_epochs': 50,                          # Số epochs
    'learning_rate': 1e-4,                     # Learning rate
    'num_classes': 6,                          # Số classes
    'image_size': 224,                         # Kích thước ảnh
    'num_workers': 4,                          # Số workers cho DataLoader
    'save_every': 5,                           # Lưu checkpoint mỗi N epochs
    'classes': ['Cercospora', 'Corticium', 'mealy', 'Miner', 'Phoma', 'Rust']
}
```

## Kết quả

Sau khi training, các file sẽ được lưu trong thư mục `outputs/train_YYYYMMDD_HHMMSS/`:

### 1. Checkpoints
- `checkpoints/best_model.pth` - Model có validation accuracy cao nhất
- `checkpoints/final_model.pth` - Model ở epoch cuối cùng
- `checkpoints/checkpoint_epoch_N.pth` - Checkpoint mỗi 5 epochs

### 2. Training history
- `training_history.json` - Lịch sử training (loss, accuracy, learning rate)
- `training_curves.png` - Đồ thị loss và accuracy

### 3. Configuration
- `config.json` - Cấu hình đã sử dụng cho training

## Cấu trúc Checkpoint

Mỗi checkpoint chứa:
```python
{
    'epoch': int,                      # Epoch number
    'model_state_dict': dict,          # Model weights
    'optimizer_state_dict': dict,      # Optimizer state
    'scheduler_state_dict': dict,      # Scheduler state
    'val_acc': float,                  # Validation accuracy
    'val_loss': float,                 # Validation loss
    'per_class_acc': dict,            # Per-class accuracy (chỉ có trong best_model)
    'config': dict                     # Configuration
}
```

## Load checkpoint để inference

```python
import torch
from train import CoffeeLeafClassifier

# Load model
model = CoffeeLeafClassifier(num_classes=6)
checkpoint = torch.load('outputs/train_YYYYMMDD_HHMMSS/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
# ... your inference code ...
```

## Tính năng

### 1. Data Augmentation
Training data được augment với:
- Random horizontal flip
- Random vertical flip
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation)

### 2. Learning Rate Scheduling
Sử dụng `CosineAnnealingLR` để giảm learning rate theo cosine

### 3. Early Stopping
Model tốt nhất được lưu dựa trên validation accuracy

### 4. Per-class Metrics
Hiển thị accuracy cho từng class trong quá trình training

### 5. Comprehensive Logging
- Loss và accuracy cho mỗi epoch
- Per-class validation accuracy
- Training curves visualization

## Ví dụ output

```
================================================================================
🍃 COFFEE LEAF DISEASE CLASSIFICATION - FINE-TUNING
================================================================================

Loading datasets...
TRAIN set: 2815 images
  - Cercospora: 233 images
  - Corticium: 254 images
  - mealy: 146 images
  - Miner: 369 images
  - Phoma: 399 images
  - Rust: 1414 images

VALID set: XXX images
  - Cercospora: XX images
  - Corticium: XX images
  ...

================================================================================
Creating model...
Loading pretrained weights from ckpt/resnet50_ssl_141025.pt...
✓ Pretrained weights loaded successfully!
✓ Model created with 25,070,086 parameters
✓ Optimizer: AdamW (lr=0.0001)
✓ Scheduler: CosineAnnealingLR

================================================================================
Starting training for 50 epochs...
================================================================================

Epoch 1/50
----------------------------------------------------------------------
Training: 100%|████████| 88/88 [00:45<00:00,  1.93it/s, loss=1.2345, acc=65.43%]
Validation: 100%|████████| 22/22 [00:08<00:00,  2.67it/s, loss=0.9876, acc=72.15%]

📊 Epoch 1 Summary:
  Train Loss: 1.2345 | Train Acc: 65.43%
  Val Loss:   0.9876 | Val Acc:   72.15%
  Learning Rate: 0.000100

  Per-class Validation Accuracy:
    - Cercospora   : 68.50%
    - Corticium    : 75.20%
    - mealy        : 65.30%
    - Miner        : 70.80%
    - Phoma        : 78.90%
    - Rust         : 85.60%

  ✓ New best model saved! (Val Acc: 72.15%)
...
```

## Lưu ý

1. **Không chạy test set**: Script này chỉ train và validate, không evaluate trên test set
2. **GPU Memory**: Nếu gặp lỗi out of memory, giảm `batch_size` trong CONFIG
3. **Training time**: Với dataset ~2800 ảnh và 50 epochs, training có thể mất vài giờ tùy GPU
4. **Pretrained weights**: Script sẽ tự động load weights từ checkpoint nếu tồn tại, nếu không sẽ train from scratch

## Troubleshooting

### 1. CUDA out of memory
```python
CONFIG['batch_size'] = 16  # Giảm batch size
```

### 2. Pretrained checkpoint không tồn tại
Script sẽ báo warning và train from scratch

### 3. Dataset path không đúng
Kiểm tra lại `CONFIG['data_root']` và cấu trúc thư mục

## Contact

Nếu có vấn đề, hãy kiểm tra:
1. Cấu trúc thư mục data có đúng không
2. Checkpoint file có tồn tại không
3. Các thư viện đã được cài đặt đầy đủ chưa
