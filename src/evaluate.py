import torch # type: ignore
import numpy as np # type: ignore
from pathlib import Path
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score ) # type: ignore
from sklearn.preprocessing import label_binarize # type: ignore
from tqdm import tqdm # type: ignore
import json
import warnings
warnings.filterwarnings('ignore')


def get_predictions_and_probabilities(model, dataloader, device):
    """Get predictions and probabilities for evaluation"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Getting predictions", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def calculate_metrics(y_true, y_pred, y_probs, class_names):
    """Calculate all evaluation metrics"""
    
    metrics = {}
    
    # Overall accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Per-class precision, recall, F1-score
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )
    
    metrics['per_class_precision'] = dict(zip(class_names, precision))
    metrics['per_class_recall'] = dict(zip(class_names, recall))
    metrics['per_class_f1'] = dict(zip(class_names, f1))
    metrics['per_class_support'] = dict(zip(class_names, support))
    
    # Macro and weighted F1-scores
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    
    # AUC-ROC (one-vs-rest)
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    # Per-class AUC
    auc_scores = {}
    for i, class_name in enumerate(class_names):
        try:
            auc_scores[class_name] = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
        except:
            auc_scores[class_name] = 0.0
    
    metrics['per_class_auc'] = auc_scores
    
    # Macro and weighted AUC
    try:
        metrics['auc_macro'] = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
        metrics['auc_weighted'] = roc_auc_score(y_true_bin, y_probs, average='weighted', multi_class='ovr')
    except:
        metrics['auc_macro'] = 0.0
        metrics['auc_weighted'] = 0.0
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def save_metrics_to_files(metrics, class_names, save_dir='results/metrics'):
    """Save metrics to text and JSON files"""
    
    # Create metrics directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Save to text file
    with open(f'{save_dir}/evaluation_metrics.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GASTROVISION-4 CLASSIFICATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1-Score: {metrics['f1_macro']:.4f}\n")
        f.write(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}\n")
        f.write(f"Macro AUC-ROC: {metrics['auc_macro']:.4f}\n")
        f.write(f"Weighted AUC-ROC: {metrics['auc_weighted']:.4f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("PER-CLASS METRICS\n")
        f.write("-" * 80 + "\n\n")
        
        for class_name in class_names:
            f.write(f"{class_name}:\n")
            f.write(f"  Precision: {metrics['per_class_precision'][class_name]:.4f}\n")
            f.write(f"  Recall:    {metrics['per_class_recall'][class_name]:.4f}\n")
            f.write(f"  F1-Score:  {metrics['per_class_f1'][class_name]:.4f}\n")
            f.write(f"  AUC-ROC:   {metrics['per_class_auc'][class_name]:.4f}\n")
            f.write(f"  Support:   {metrics['per_class_support'][class_name]}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 80 + "\n\n")
        f.write(str(metrics['confusion_matrix']) + "\n")
    
    print(f"Metrics saved to {save_dir}/evaluation_metrics.txt")
    
    # Save to JSON (excluding confusion matrix)
    metrics_json = {
        'overall': {
            'accuracy': float(metrics['accuracy']),
            'f1_macro': float(metrics['f1_macro']),
            'f1_weighted': float(metrics['f1_weighted']),
            'auc_macro': float(metrics['auc_macro']),
            'auc_weighted': float(metrics['auc_weighted'])
        },
        'per_class': {}
    }
    
    for class_name in class_names:
        metrics_json['per_class'][class_name] = {
            'precision': float(metrics['per_class_precision'][class_name]),
            'recall': float(metrics['per_class_recall'][class_name]),
            'f1_score': float(metrics['per_class_f1'][class_name]),
            'auc_roc': float(metrics['per_class_auc'][class_name]),
            'support': int(metrics['per_class_support'][class_name])
        }
    
    with open(f'{save_dir}/evaluation_metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=4)
    
    print(f"Metrics saved to {save_dir}/evaluation_metrics.json")
