from torch.utils.data import Dataset, WeightedRandomSampler # type: ignore
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore
import numpy as np # type: ignore
from pathlib import Path
from sklearn.model_selection import train_test_split # type: ignore
import json
import warnings
warnings.filterwarnings('ignore')



class GastroVisionDataset(Dataset):
    """Custom Dataset for GastroVision-4 endoscopic images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    

def load_dataset(data_dir):
    """Load dataset from directory structure"""
    data_dir = Path(data_dir)
    classes = ['colon_polyps', 'erythema', 'normal_esophagus', 'large_bowels']
    
    image_paths = []
    labels = []
    
    for idx, class_name in enumerate(classes):
        class_dir = data_dir / class_name
        if class_dir.exists():
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_paths.append(str(img_path))
                    labels.append(idx)
    
    return np.array(image_paths), np.array(labels), classes


def prepare_data_splits(image_paths, labels, test_size=0.2, val_size=0.1, random_state=42):
    """Create stratified train/val/test splits using scikit-learn"""
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, 
        test_size=test_size, 
        stratify=labels,
        random_state=random_state
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        stratify=y_temp,
        random_state=random_state
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def get_transforms():
    """Get data augmentation transforms for training and validation"""
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_weighted_sampler(labels):
        """Create weighted sampler to handle class imbalance"""
        class_counts = np.bincount(labels)
        class_weights = 1. / class_counts
        sample_weights = class_weights[labels]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        return sampler









