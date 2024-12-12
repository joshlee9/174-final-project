import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import os

class SiameseNetwork(nn.Module):
    """
    A simple Siamese Network for feature matching.
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 10 * 10, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        # Compute absolute difference
        diff = torch.abs(output1 - output2)
        out = self.fc(diff)
        return out

class SiameseDataset(Dataset):
    """
    Dataset for Siamese Network.
    """

    def __init__(self, image_pairs, labels, transform=None):
        self.image_pairs = image_pairs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        label = self.labels[idx]

        # Load images
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor([label], dtype=torch.float32)

def create_image_pairs(dataset_path, positive_ratio=0.5):
    """
    Create image pairs for training.

    Parameters:
    - dataset_path: Path to the dataset directory containing images.
    - positive_ratio: Ratio of positive pairs.

    Returns:
    - image_pairs: List of tuples (img1_path, img2_path).
    - labels: List of labels (1 for similar, 0 for dissimilar).
    """
    images = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    image_pairs = []
    labels = []

    # Create positive pairs (same image with slight variations if available)
    for img in images:
        # Here, we assume that similar images have similar filenames
        similar_imgs = [i for i in images if os.path.basename(i).split('_')[0] == os.path.basename(img).split('_')[0]]
        if len(similar_imgs) > 1:
            for sim_img in similar_imgs[1:]:
                image_pairs.append((img, sim_img))
                labels.append(1)

    # Create negative pairs
    num_positive = len(labels)
    negative_pairs = []
    for i in range(num_positive):
        img1, img2 = np.random.choice(images, 2, replace=False)
        negative_pairs.append((img1, img2))
        labels.append(0)

    image_pairs += negative_pairs

    return image_pairs, labels

def train_siamese_network(dataset_path, epochs=10, batch_size=32, learning_rate=0.001):
    """
    Train the Siamese Network.

    Parameters:
    - dataset_path: Path to the dataset directory containing images.
    - epochs: Number of training epochs.
    - batch_size: Batch size.
    - learning_rate: Learning rate.

    Returns:
    - model: Trained Siamese Network model.
    """
    # Create image pairs
    image_pairs, labels = create_image_pairs(dataset_path)

    # Split into training and validation sets
    pairs_train, pairs_val, labels_train, labels_val = train_test_split(
        image_pairs, labels, test_size=0.2, random_state=42
    )

    # Define transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create datasets
    train_dataset = SiameseDataset(pairs_train, labels_train, transform=transform)
    val_dataset = SiameseDataset(pairs_val, labels_val, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

    # Initialize model, loss function, and optimizer
    model = SiameseNetwork()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for img1, img2, label in train_loader:
            optimizer.zero_grad()
            output = model(img1, img2)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for img1, img2, label in val_loader:
                output = model(img1, img2)
                predicted = (output > 0.5).float()
                total += label.size(0)
                correct += (predicted == label).sum().item()

        accuracy = correct / total * 100
        print(f"Validation Accuracy: {accuracy:.2f}%\n")

    return model

def predict_similarity(model, img1_path, img2_path, transform=None):
    """
    Predict the similarity between two images using the trained Siamese Network.

    Parameters:
    - model: Trained Siamese Network model.
    - img1_path: Path to the first image.
    - img2_path: Path to the second image.
    - transform: Optional transformations to apply.

    Returns:
    - similarity_score: Probability score indicating similarity.
    """
    model.eval()

    # Load images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if transform:
        img1 = transform(img1).unsqueeze(0)
        img2 = transform(img2).unsqueeze(0)
    else:
        img1 = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        img2 = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

    with torch.no_grad():
        similarity_score = model(img1, img2).item()

    return similarity_score

# Example usage
if __name__ == "__main__":
    # Path to your dataset
    dataset_path = 'datasets/yaseen_panorama/'

    # Train the Siamese Network
    # Note: Training a Siamese Network requires a significant amount of data and computational resources.
    # For demonstration purposes, you might want to reduce the number of epochs or use a subset of the data.
    model = train_siamese_network(dataset_path, epochs=5, batch_size=16, learning_rate=0.001)

    # Save the trained model
    torch.save(model.state_dict(), 'siamese_model.pth')
    print("Model trained and saved.")

    # Predict similarity between two images
    img1_path = 'datasets/yaseen_panorama/sample1.jpg'
    img2_path = 'datasets/yaseen_panorama/sample2.jpg'

    # Define the same transformation used during training
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    similarity = predict_similarity(model, img1_path, img2_path, transform=transform)
    print(f"Similarity Score between images: {similarity:.4f}")
