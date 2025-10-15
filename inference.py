#!/usr/bin/env python3
"""
Inference Script: Coffee Leaf Disease Classification
Load trained model and predict on new images
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys
from pathlib import Path

# Import model from train.py
from train import CoffeeLeafClassifier, CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path, num_classes=6):
    """Load trained model from checkpoint"""
    model = CoffeeLeafClassifier(num_classes=num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from {checkpoint_path}")
    if 'val_acc' in checkpoint:
        print(f"  Validation Accuracy: {checkpoint['val_acc']:.2f}%")
    if 'epoch' in checkpoint:
        print(f"  Trained Epochs: {checkpoint['epoch']}")
    
    return model

def predict_image(model, image_path, classes=None):
    """Predict class for a single image"""
    if classes is None:
        classes = CONFIG['classes']
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = classes[predicted.item()]
    confidence_score = confidence.item() * 100
    
    # Get top-3 predictions
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    top3_predictions = [
        (classes[idx.item()], prob.item() * 100) 
        for idx, prob in zip(top3_idx[0], top3_prob[0])
    ]
    
    return predicted_class, confidence_score, top3_predictions

def main():
    """Main inference function"""
    if len(sys.argv) < 3:
        print("Usage: python inference.py <checkpoint_path> <image_path>")
        print("\nExample:")
        print("  python inference.py outputs/train_20241015_120000/checkpoints/best_model.pth data/ft_data/test/Cercospora/image.jpg")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    image_path = sys.argv[2]
    
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    print("="*70)
    print("🍃 Coffee Leaf Disease Classification - Inference")
    print("="*70)
    print(f"\nImage: {image_path}")
    print(f"Model: {checkpoint_path}\n")
    
    # Load model
    model = load_model(checkpoint_path)
    
    # Predict
    predicted_class, confidence, top3 = predict_image(model, image_path)
    
    # Display results
    print("\n" + "="*70)
    print("📊 PREDICTION RESULTS")
    print("="*70)
    print(f"\n🔍 Predicted Class: {predicted_class}")
    print(f"📈 Confidence: {confidence:.2f}%\n")
    
    print("Top 3 Predictions:")
    for i, (cls, prob) in enumerate(top3, 1):
        bar = "█" * int(prob / 2)
        print(f"  {i}. {cls:12s}: {prob:6.2f}% {bar}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
