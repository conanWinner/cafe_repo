
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


###################################
#   Dataset Class
###################################
class ColorizationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")  # Load image as RGB
        image = np.array(image)

        # Convert RGB to Lab
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB) / 255.0  # Normalize to [0, 1]
        L = image_lab[..., 0:1]  # Grayscale
        ab = image_lab[..., 1:]  # Chromaticity

        # Ensure L and ab have the correct shape
        L = L.squeeze(-1)  # Remove the last singleton dimension
        ab = ab.transpose(2, 0, 1)  # Change shape to (C, H, W) for tensor conversion

        if self.transform:
            L = self.transform(Image.fromarray((L * 255).astype(np.uint8)))  # Convert to PIL for transform
            ab = self.transform(Image.fromarray((ab * 255).astype(np.uint8).transpose(1, 2, 0)))  # Revert to HWC for PIL

        return L, ab


###################################
#   Encoder and Decoder
###################################
encoder = nn.Sequential(*list(models.resnet50().children())[:-2]).to(device)

decoder = nn.Sequential(
    nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),

    nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),

    nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),

    nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),

    nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1),  # 2 channels (ab)
    nn.Tanh(),

    # Ensure the output is resized to (224, 224)
    nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)
).to(device)



###################################
#   Loss Function and Optimizer
###################################
def colorization_loss(predicted_ab, true_ab):
    return nn.MSELoss()(predicted_ab, true_ab)


def get_optimizer(models, lr=1e-4):
    params = []
    for model in models:
        params += list(model.parameters())
    return optim.AdamW(params, lr=lr)


###################################
#   Training Function
###################################
def train_colorization(encoder, decoder, dataloader, optimizer, epochs=10):
    encoder.train()
    decoder.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for L, ab in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            L = L.repeat(1, 3, 1, 1)  # Replicate the single channel to create 3 channels
            L, ab = L.to(device), ab.to(device)

            # Forward pass
            optimizer.zero_grad()
            features = encoder(L)
            predicted_ab = decoder(features)

            # Compute loss
            loss = colorization_loss(predicted_ab, ab)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


###################################
#   Visualization Function
###################################
def visualize_results(encoder, decoder, dataloader):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for L, ab in dataloader:
            L = L.repeat(1, 3, 1, 1)  # Replicate the single channel to create 3 channels
            L = L.to(device)
            ab_pred = decoder(encoder(L)).cpu().detach().numpy()
            L = L.cpu().detach().numpy()

            # Convert back to RGB
            for i in range(1):
                lab_image = np.concatenate((L[i, 0, :, :, None], ab_pred[i].transpose(1, 2, 0)), axis=2)
                lab_image = (lab_image * 255).astype(np.uint8)
                rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

                # Plot original grayscale and predicted RGB
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(L[i, 0], cmap="gray")
                axes[0].set_title("Grayscale Input (L)")
                axes[1].imshow(rgb_image)
                axes[1].set_title("Predicted Colorization")
                plt.show()
            break


###################################
#   Main Script
###################################
if __name__ == "__main__":
    # Paths
    data_dir = "data/pretrain"

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Dataset and DataLoader
    dataset = ColorizationDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Optimizer
    optimizer = get_optimizer([encoder, decoder], lr=1e-4)

    # Training Loop
    epochs = 20
    print("Starting training...")
    train_colorization(encoder, decoder, dataloader, optimizer, epochs)

    # Save Checkpoints
    torch.save(encoder.state_dict(), "./resnet50_ssl_141025.pt")
    # torch.save(decoder.state_dict(), "./decoder.pth")
    print("Training completed and checkpoints saved.")

    # Visualization
    visualize_results(encoder, decoder, dataloader)

