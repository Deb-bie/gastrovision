import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import numpy as np # type: ignore
from sklearn.preprocessing import label_binarize # type: ignore
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score) # type: ignore



def plot_confusion_matrix(cm, class_names, save_path='results/plots/confusion_matrix.png'):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_normalized_confusion_matrix(cm, class_names, save_path='results/plots/confusion_matrix_normalized.png'):
    """Plot normalized confusion matrix"""
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Normalized confusion matrix saved to {save_path}")


def plot_roc_curves(y_true, y_probs, class_names, save_path='results/plots/roc_curves.png'):
    """Plot ROC curves for all classes"""
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each class
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multi-class Classification', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved to {save_path}")


def plot_precision_recall_curves(y_true, y_probs, class_names, save_path='results/plots/precision_recall_curves.png'):
    """Plot Precision-Recall curves for all classes"""
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        avg_precision = average_precision_score(y_true_bin[:, i], y_probs[:, i])
        plt.plot(recall, precision, lw=2, label=f'{class_name} (AP = {avg_precision:.3f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold')
    plt.legend(loc="best", fontsize=10)
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Precision-Recall curves saved to {save_path}")


def plot_per_class_metrics(metrics, class_names, save_path='results/plots/per_class_metrics.png'):
    """Plot per-class precision, recall, F1-score, and AUC"""
    
    precision = [metrics['per_class_precision'][c] for c in class_names]
    recall = [metrics['per_class_recall'][c] for c in class_names]
    f1 = [metrics['per_class_f1'][c] for c in class_names]
    auc_scores = [metrics['per_class_auc'][c] for c in class_names]
    
    x = np.arange(len(class_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars1 = ax.bar(x - 1.5*width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x - 0.5*width, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + 0.5*width, f1, width, label='F1-Score', color='#e74c3c')
    bars4 = ax.bar(x + 1.5*width, auc_scores, width, label='AUC-ROC', color='#f39c12')
    
    ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class metrics plot saved to {save_path}")


def plot_training_history(history, save_path='results/plots/training_history.png'):
    """Plot training and validation loss and accuracy"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2, color='#3498db')
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2, color='#e74c3c')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2, color='#3498db')
    axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2, color='#e74c3c')
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")


def plot_class_distribution(y_train, y_val, y_test, class_names, save_path='results/plots/class_distribution.png'):
    """Plot class distribution across splits"""
    
    train_counts = np.bincount(y_train, minlength=len(class_names))
    val_counts = np.bincount(y_val, minlength=len(class_names))
    test_counts = np.bincount(y_test, minlength=len(class_names))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, train_counts, width, label='Train', color='#3498db')
    ax.bar(x, val_counts, width, label='Validation', color='#2ecc71')
    ax.bar(x + width, test_counts, width, label='Test', color='#e74c3c')
    
    ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution Across Data Splits', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (train, val, test) in enumerate(zip(train_counts, val_counts, test_counts)):
        ax.text(i - width, train, str(train), ha='center', va='bottom', fontsize=9)
        ax.text(i, val, str(val), ha='center', va='bottom', fontsize=9)
        ax.text(i + width, test, str(test), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Class distribution plot saved to {save_path}")


