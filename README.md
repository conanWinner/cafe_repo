# Coffee Leaf Disease Classification

Hệ thống phân loại bệnh lá cà phê sử dụng Deep Learning với pretrained ResNet50.

## 📋 Tổng quan

Dự án này thực hiện:
1. **Pretraining**: Self-supervised learning với colorization task trên unlabeled data
2. **Fine-tuning**: Transfer learning trên labeled data với 6 loại bệnh
3. **Evaluation**: Đánh giá model với confusion matrix, classification report
4. **Inference**: Dự đoán bệnh trên ảnh mới

## 🗂️ Cấu trúc dự án

```
cafe_repo/
├── pretrain.py              # Pretraining script (colorization)
├── train.py                 # Fine-tuning script
├── inference.py             # Inference script
├── evaluate.py              # Evaluation script
├── TRAINING_GUIDE.md        # Hướng dẫn chi tiết
├── ckpt/
│   └── resnet50_ssl_141025.pt  # Pretrained checkpoint
├── data/
│   ├── pretrain/            # Unlabeled data for pretraining
│   └── ft_data/             # Labeled data for fine-tuning
│       ├── train/           # Training set
│       │   ├── Cercospora/  (233 images)
│       │   ├── Corticium/   (254 images)
│       │   ├── mealy/       (146 images)
│       │   ├── Miner/       (369 images)
│       │   ├── Phoma/       (399 images)
│       │   └── Rust/        (1414 images)
│       ├── valid/           # Validation set
│       └── test/            # Test set
└── outputs/                 # Training results and checkpoints
```

## 🚀 Cài đặt

### Requirements

```bash
pip install torch torchvision tqdm matplotlib pillow numpy scikit-learn seaborn
```

Hoặc:

```bash
pip install -r requirements.txt
```

## 📊 Dataset

### Classes (6 loại bệnh)

1. **Cercospora** - Bệnh đốm lá Cercospora
2. **Corticium** - Bệnh Corticium
3. **mealy** - Bệnh rệp sáp
4. **Miner** - Bệnh sâu đục lá
5. **Phoma** - Bệnh Phoma
6. **Rust** - Bệnh gỉ sắt

### Thống kê

- **Train set**: 2,815 images
- **Validation set**: ~400 images
- **Test set**: ~100 images

## 🎯 Workflow

### 1. Pretraining (Tùy chọn)

```bash
python pretrain.py
```

Script này sử dụng self-supervised learning với colorization task.

### 2. Fine-tuning

```bash
python train.py
```

**Tính năng:**
- Load pretrained checkpoint từ `ckpt/resnet50_ssl_141025.pt`
- Train trên ft_data với data augmentation
- Cosine annealing learning rate
- Lưu best model và checkpoints định kỳ
- Visualize training curves

**Kết quả lưu trong:**
- `outputs/train_YYYYMMDD_HHMMSS/checkpoints/best_model.pth`
- `outputs/train_YYYYMMDD_HHMMSS/checkpoints/final_model.pth`
- `outputs/train_YYYYMMDD_HHMMSS/training_history.json`
- `outputs/train_YYYYMMDD_HHMMSS/training_curves.png`

### 3. Evaluation

```bash
python evaluate.py outputs/train_YYYYMMDD_HHMMSS/checkpoints/best_model.pth
```

**Tạo ra:**
- Classification report (text + JSON)
- Confusion matrix
- Per-class metrics (Precision, Recall, F1)
- Top-k accuracy plot
- Detailed predictions JSON

### 4. Inference

```bash
python inference.py outputs/train_YYYYMMDD_HHMMSS/checkpoints/best_model.pth path/to/image.jpg
```

**Output:**
```
🔍 Predicted Class: Rust
📈 Confidence: 95.32%

Top 3 Predictions:
  1. Rust        : 95.32% ██████████████████████████████████████████████████
  2. Cercospora  :  3.21% ███
  3. Phoma       :  1.15% █
```

## 📝 Ví dụ sử dụng

### Training với custom config

Chỉnh sửa `CONFIG` trong `train.py`:

```python
CONFIG = {
    'batch_size': 16,        # Giảm nếu GPU memory nhỏ
    'num_epochs': 100,       # Tăng để train lâu hơn
    'learning_rate': 5e-5,   # Learning rate
    'save_every': 10,        # Lưu checkpoint mỗi 10 epochs
}
```

### Load model để inference

```python
import torch
from train import CoffeeLeafClassifier
from PIL import Image
from torchvision import transforms

# Load model
model = CoffeeLeafClassifier(num_classes=6)
checkpoint = torch.load('path/to/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Predict
image = Image.open('image.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)
output = model(image_tensor)
_, predicted = output.max(1)
```

## 📈 Kết quả mẫu

### Training Curves

![Training Curves](outputs/example/training_curves.png)

### Confusion Matrix

![Confusion Matrix](outputs/example/confusion_matrix.png)

### Per-class Metrics

| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Cercospora | 0.85      | 0.82   | 0.83     | 45      |
| Corticium  | 0.88      | 0.90   | 0.89     | 50      |
| mealy      | 0.79      | 0.75   | 0.77     | 30      |
| Miner      | 0.92      | 0.89   | 0.90     | 75      |
| Phoma      | 0.87      | 0.91   | 0.89     | 80      |
| Rust       | 0.95      | 0.97   | 0.96     | 280     |

## 🔧 Troubleshooting

### 1. CUDA out of memory

```python
CONFIG['batch_size'] = 8  # Giảm batch size
CONFIG['num_workers'] = 0  # Giảm workers
```

### 2. Import errors

```bash
# Đảm bảo chạy từ thư mục gốc
cd /path/to/cafe_repo
python train.py
```

### 3. Pretrained checkpoint không tìm thấy

Script sẽ tự động train from scratch nếu không tìm thấy checkpoint.

## 📚 Tài liệu tham khảo

- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Hướng dẫn chi tiết về training
- [pretrain.py](pretrain.py) - Self-supervised pretraining
- [train.py](train.py) - Fine-tuning script với comments chi tiết

## 🤝 Contributing

Nếu có vấn đề hoặc đề xuất, vui lòng tạo issue hoặc pull request.

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Dataset: Coffee Leaf Disease Dataset
- Architecture: ResNet50
- Framework: PyTorch
