import torch # type: ignore
import torch.nn as nn # type: ignore
import numpy as np # type: ignore
import random

def get_device(verbose=True):
    """
    Automatically detect and return the best available device.
    Priority: MPS (Apple Silicon) > CUDA > CPU
    
    Args:
        verbose (bool): Print device information
    
    Returns:
        torch.device: Selected device
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("✓ Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"✓ Using NVIDIA GPU (CUDA) - {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if verbose:
            print("✓ Using CPU")
    
    return device


def test_device(device):
    """
    Test if the device is working properly with a simple computation.
    
    Args:
        device (torch.device): Device to test
    
    Returns:
        bool: True if test passed, False otherwise
    """
    try:
        test_tensor = torch.randn(100, 100).to(device)
        test_result = torch.matmul(test_tensor, test_tensor)
        print(f"✓ Device {device} test passed - computations working!")
        return True
    except Exception as e:
        print(f"✗ Device {device} test failed: {e}")
        return False


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification with class imbalance.
    
    Focal Loss down-weights easy examples and focuses on hard examples,
    making it particularly effective for imbalanced datasets.
    
    Args:
        alpha (torch.Tensor, optional): Class weights of shape [num_classes].
                                       If None, no class weighting is applied.
        gamma (float): Focusing parameter. Higher values focus more on hard examples.
                      - gamma=0: Equivalent to CrossEntropyLoss
                      - gamma=1: Mild focusing
                      - gamma=2: Standard (recommended)
                      - gamma=3+: Strong focusing
        reduction (str): Specifies reduction: 'none', 'mean', 'sum'
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
        https://arxiv.org/abs/1708.02002
    
    Example:
        >>> num_classes = 4
        >>> class_weights = torch.tensor([0.5, 1.0, 1.5, 0.8])
        >>> criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        >>> outputs = torch.randn(8, num_classes)  # batch_size=8
        >>> targets = torch.randint(0, num_classes, (8,))
        >>> loss = criterion(outputs, targets)
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predicted logits of shape [batch_size, num_classes]
            targets (torch.Tensor): Ground truth labels of shape [batch_size]
        
        Returns:
            torch.Tensor: Computed focal loss
        """
        # Calculate cross entropy loss without reduction
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        
        # Calculate pt (probability of correct class)
        pt = torch.exp(-ce_loss)
        
        # Calculate focal loss: FL = -(1 - pt)^gamma * log(pt)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for multi-class classification.
    
    Prevents the model from becoming over-confident by smoothing the labels.
    Useful for preventing overfitting and improving generalization.
    
    Args:
        num_classes (int): Number of classes
        smoothing (float): Label smoothing factor (0.0 to 1.0)
                          - 0.0: No smoothing (standard CrossEntropy)
                          - 0.1: Recommended default
                          - 0.2: Higher smoothing
    
    Example:
        >>> criterion = LabelSmoothingLoss(num_classes=4, smoothing=0.1)
        >>> outputs = torch.randn(8, 4)
        >>> targets = torch.randint(0, 4, (8,))
        >>> loss = criterion(outputs, targets)
    """
    
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs, targets):
        log_probs = torch.nn.functional.log_softmax(inputs, dim=-1)
        
        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


def get_loss_function(loss_name='focal', class_weights=None, num_classes=4, **kwargs):
    """
    Factory function to get loss function by name.
    
    Args:
        loss_name (str): Name of loss function
                        - 'focal': Focal Loss (recommended for imbalance)
                        - 'cross_entropy' or 'ce': Standard CrossEntropy
                        - 'label_smoothing' or 'ls': Label Smoothing Loss
        class_weights (torch.Tensor, optional): Class weights for weighted losses
        num_classes (int): Number of classes
        **kwargs: Additional arguments for specific loss functions
    
    Returns:
        nn.Module: Loss function
    
    Example:
        >>> from sklearn.utils.class_weight import compute_class_weight
        >>> class_weights = compute_class_weight('balanced', classes=[0,1,2,3], y=train_labels)
        >>> class_weights = torch.FloatTensor(class_weights)
        >>> criterion = get_loss_function('focal', class_weights=class_weights, gamma=2.0)
    """
    loss_name = loss_name.lower()
    
    if loss_name == 'focal':
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=class_weights, gamma=gamma)
    
    elif loss_name in ['cross_entropy', 'ce']:
        return nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_name in ['label_smoothing', 'ls']:
        smoothing = kwargs.get('smoothing', 0.1)
        return LabelSmoothingLoss(num_classes=num_classes, smoothing=smoothing)
    
    else:
        raise ValueError(f"Unknown loss function: {loss_name}. "
                        f"Choose from: 'focal', 'cross_entropy', 'label_smoothing'")


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    print(f"✓ Random seed set to {seed}")


def save_checkpoint(model, optimizer, epoch, val_acc, save_path, **kwargs):
    """
    Save model checkpoint with training state.
    
    Args:
        model (nn.Module): PyTorch model
        optimizer (torch.optim.Optimizer): Optimizer
        epoch (int): Current epoch
        val_acc (float): Validation accuracy
        save_path (str): Path to save checkpoint
        **kwargs: Additional items to save (e.g., scheduler, class_names)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        **kwargs
    }
    torch.save(checkpoint, save_path)
    print(f"✓ Checkpoint saved to {save_path}")


def load_checkpoint(model, checkpoint_path, optimizer=None, device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        model (nn.Module): PyTorch model
        checkpoint_path (str): Path to checkpoint
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state
        device (str or torch.device): Device to load model to
    
    Returns:
        dict: Checkpoint dictionary with training state
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Checkpoint loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Acc: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    return checkpoint


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to config.yaml
    
    Returns:
        dict: Configuration dictionary
    """
    import yaml # type: ignore
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Configuration loaded from {config_path}")
    return config