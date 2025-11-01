import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import os
from sklearn.model_selection import train_test_split

class CarDamageDataset(Dataset):
    """Custom dataset for car damage detection"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_dataset(dataset_dir):
    """Load images from damaged and undamaged folders"""
    image_paths = []
    labels = []
    
    # Load damaged images (label = 1)
    damaged_dir = os.path.join(dataset_dir, 'damaged')
    if os.path.exists(damaged_dir):
        for img_file in os.listdir(damaged_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(damaged_dir, img_file))
                labels.append(1)  # Damaged
    
    # Load undamaged images (label = 0)
    undamaged_dir = os.path.join(dataset_dir, 'undamaged')
    if os.path.exists(undamaged_dir):
        for img_file in os.listdir(undamaged_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(undamaged_dir, img_file))
                labels.append(0)  # Undamaged
    
    return image_paths, labels

def train_model(dataset_dir='dataset', epochs=20, batch_size=8):
    """Train damage detection model"""
    
    # Check if dataset exists
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory '{dataset_dir}' not found!")
        print("Create folders: dataset/damaged/ and dataset/undamaged/")
        return
    
    # Load dataset
    image_paths, labels = load_dataset(dataset_dir)
    
    if len(image_paths) == 0:
        print("❌ No images found in dataset!")
        print("Add images to dataset/damaged/ and dataset/undamaged/")
        return
    
    print(f"✅ Found {len(image_paths)} images")
    print(f"   - Damaged: {sum(labels)}")
    print(f"   - Undamaged: {len(labels) - sum(labels)}")
    
    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CarDamageDataset(train_paths, train_labels, train_transform)
    val_dataset = CarDamageDataset(val_paths, val_labels, val_transform)
    
    # Calculate class weights to balance training (handles imbalanced dataset)
    undamaged_count = len(train_labels) - sum(train_labels)
    damaged_count = sum(train_labels)
    
    print(f"\n⚖️  Balancing dataset:")
    print(f"   Undamaged: {undamaged_count}")
    print(f"   Damaged: {damaged_count}")
    
    # Create weights - give more importance to minority class
    class_weights = [1.0 / undamaged_count, 1.0 / damaged_count]
    sample_weights = [class_weights[label] for label in train_labels]
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Use sampler instead of shuffle
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Use pre-trained ResNet18
    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: damaged/undamaged
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n🚀 Training on {device}...\n")
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for images, labels_batch in train_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels_batch).sum().item()
        
        train_acc = 100 * train_correct / len(train_dataset)
        
        # Validation phase
        model.eval()
        val_correct = 0
        
        with torch.no_grad():
            for images, labels_batch in val_loader:
                images, labels_batch = images.to(device), labels_batch.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels_batch).sum().item()
        
        val_acc = 100 * val_correct / len(val_dataset)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/damage_model.pt')
            print(f"   ✅ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    print(f"\n✅ Training complete! Best validation accuracy: {best_acc:.2f}%")
    print("Model saved to: models/damage_model.pt")

if __name__ == "__main__":
    train_model(epochs=20, batch_size=8)
