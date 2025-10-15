# 🏗️ Model Architecture Diagram

## Checkpoint Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  PRETRAINED CHECKPOINT (resnet50_ssl_141025.pt)             │
│  Architecture: nn.Sequential(*list(resnet50.children())[:-2])│
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │   Input Image                     │
        │   Shape: [B, 3, 224, 224]        │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │   Conv1 + BN + ReLU + MaxPool     │
        │   ResNet50 Layer 1                │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │   ResNet50 Layer 2                │
        │   (Residual blocks)               │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │   ResNet50 Layer 3                │
        │   (Residual blocks)               │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │   ResNet50 Layer 4                │
        │   (Residual blocks)               │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │   Output from Checkpoint          │
        │   Shape: [B, 2048, 7, 7]         │
        └───────────────────────────────────┘
                            │
          ╔═════════════════╩═════════════════╗
          ║   STOPPED HERE IN CHECKPOINT     ║
          ║   (No avgpool, No fc)            ║
          ╚══════════════════════════════════╝


## Full Model Architecture (CoffeeLeafClassifier)

```
┌─────────────────────────────────────────────────────────────┐
│  Input Image                                                 │
│  Shape: [B, 3, 224, 224]                                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
╔═════════════════════════════════════════════════════════════╗
║  self.features (PRETRAINED ENCODER)                         ║
║  Loaded from: ckpt/resnet50_ssl_141025.pt                  ║
║  ─────────────────────────────────────────────────────      ║
║  │ ResNet50 layers (conv1 → layer4)                  │      ║
║  │ nn.Sequential(*list(resnet50.children())[:-2])    │      ║
║  └──────────────────────────────────────────────────┘      ║
║                                                              ║
║  Output Shape: [B, 2048, 7, 7]                             ║
╚═════════════════════════════════════════════════════════════╝
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  self.avgpool                     │
        │  AdaptiveAvgPool2d((1, 1))       │
        │                                   │
        │  [B, 2048, 7, 7] → [B, 2048, 1, 1]│
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  self.classifier                  │
        │  ────────────────────────         │
        │  │ Flatten()                  │   │
        │  │ [B, 2048, 1, 1] → [B, 2048]│   │
        │  └──────────────────────────┘    │
        │                                   │
        │  │ Dropout(0.5)               │   │
        │  └──────────────────────────┘    │
        │                                   │
        │  │ Linear(2048 → 512)         │   │
        │  └──────────────────────────┘    │
        │                                   │
        │  │ ReLU()                     │   │
        │  └──────────────────────────┘    │
        │                                   │
        │  │ Dropout(0.3)               │   │
        │  └──────────────────────────┘    │
        │                                   │
        │  │ Linear(512 → 6)            │   │
        │  └──────────────────────────┘    │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  Output (Class Logits)            │
        │  Shape: [B, 6]                    │
        │                                   │
        │  Classes:                         │
        │  0: Cercospora                    │
        │  1: Corticium                     │
        │  2: mealy                         │
        │  3: Miner                         │
        │  4: Phoma                         │
        │  5: Rust                          │
        └───────────────────────────────────┘
```

## Data Flow Example

```
Example with batch_size = 4:

Input:
  [4, 3, 224, 224]  ← 4 images, RGB, 224x224
        ↓
Features (Pretrained):
  [4, 2048, 7, 7]   ← Feature maps from ResNet50
        ↓
AvgPool:
  [4, 2048, 1, 1]   ← Global average pooling
        ↓
Flatten:
  [4, 2048]         ← Flattened features
        ↓
Dropout:
  [4, 2048]         ← 50% dropout
        ↓
Linear 1:
  [4, 512]          ← First FC layer
        ↓
ReLU:
  [4, 512]          ← Activation
        ↓
Dropout:
  [4, 512]          ← 30% dropout
        ↓
Linear 2:
  [4, 6]            ← Output logits (6 classes)
        ↓
CrossEntropyLoss:
  scalar            ← Loss value
```

## Training Flow

```
┌────────────────────────────────────────────────────┐
│  1. Load Pretrained Checkpoint                     │
│     checkpoint = torch.load('ckpt/resnet50_...')  │
│     self.features = checkpoint                     │
└────────────────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────┐
│  2. Load Training Data                             │
│     train_loader = DataLoader(train_dataset)      │
│     val_loader = DataLoader(val_dataset)          │
└────────────────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────┐
│  3. Training Loop (num_epochs)                     │
│     for epoch in range(num_epochs):               │
│       ├─ Train on train_loader                    │
│       ├─ Validate on val_loader                   │
│       ├─ Save checkpoint if best                  │
│       └─ Save periodic checkpoint                 │
└────────────────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────┐
│  4. Save Results                                   │
│     ├─ best_model.pth                             │
│     ├─ final_model.pth                            │
│     ├─ training_log.json                          │
│     ├─ training_curves.png                        │
│     └─ checkpoints/                               │
└────────────────────────────────────────────────────┘
```

## Checkpoint Structure

```python
checkpoint = {
    'epoch': 49,                          # Last epoch
    'model_state_dict': {...},            # Model weights
    'optimizer_state_dict': {...},        # Optimizer state
    'train_loss': 0.234,                  # Training loss
    'val_loss': 0.345,                    # Validation loss
    'train_acc': 0.921,                   # Training accuracy
    'val_acc': 0.887,                     # Validation accuracy
    'config': {...}                       # Training config
}
```

## Key Differences

### Original ResNet50:
```
Input → Conv1 → Layer1 → Layer2 → Layer3 → Layer4 → AvgPool → FC → Output
                                                         ↑        ↑
                                                    layer -2  layer -1
```

### Pretrained Checkpoint (children()[:-2]):
```
Input → Conv1 → Layer1 → Layer2 → Layer3 → Layer4 → [END]
                                                       ↑
                                                  Output: [B, 2048, 7, 7]
```

### Our Full Model:
```
Input → [Pretrained Features] → AvgPool → [Custom Classifier] → Output
         (loaded from ckpt)                  (trainable)         [B, 6]
```

## Parameters Count

```
Total Parameters: ~25M
  ├─ Features (Pretrained): ~23M   [85%]
  │   └─ These are pretrained SSL weights
  │
  └─ Classifier (New): ~2M          [15%]
      ├─ Linear(2048 → 512): ~1M
      └─ Linear(512 → 6): ~3K
```

## Training Strategy

```
┌──────────────────────────────────────┐
│  Features (Pretrained)               │
│  ────────────────────────────        │
│  Trainable: ✓ YES                    │
│  Learning Rate: 1e-4                 │
│  Fine-tuning all layers              │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│  Classifier (New)                    │
│  ────────────────────────────────    │
│  Trainable: ✓ YES                    │
│  Learning Rate: 1e-4                 │
│  Training from scratch               │
└──────────────────────────────────────┘

Optimizer: Adam(lr=1e-4)
Loss: CrossEntropyLoss
```

---

**Summary**: The checkpoint contains only the ResNet50 feature extractor (without avgpool and fc). We load it directly and add our own avgpool + classifier head to complete the model.
