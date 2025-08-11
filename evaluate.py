#!/usr/bin/env python3
"""
Snake Image Classification Model Evaluation Script
Author: PyTorch Expert Assistant
Description: Comprehensive evaluation of the trained snake classification model
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import pandas as pd
from tqdm import tqdm
import json


class ModelEvaluator:
    """A comprehensive model evaluation class for snake classification."""
    
    def __init__(self, model_path, test_dir, validation_dir=None, device=None):
        """
        Initialize the model evaluator.
        
        Args:
            model_path (str): Path to the trained model file (.pth)
            test_dir (str): Path to test dataset directory
            validation_dir (str): Path to validation dataset directory (optional)
            device (str): Device to use for evaluation ('cuda', 'mps', 'cpu')
        """
        self.model_path = model_path
        self.test_dir = test_dir
        self.validation_dir = validation_dir
        
        # Device setup
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model and data
        self.model = None
        self.test_loader = None
        self.val_loader = None
        self.class_names = None
        self.num_classes = None
        
        self._setup_data_loaders()
        self._load_model()
    
    def _setup_data_loaders(self):
        """Setup data loaders for evaluation."""
        print("Setting up data loaders...")
        
        # ImageNet normalization values
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        
        # Test/validation transforms (no augmentation)
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
        
        # Create test dataset
        test_dataset = datasets.ImageFolder(self.test_dir, transform=test_transforms)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        # Create validation dataset if provided
        if self.validation_dir and os.path.exists(self.validation_dir):
            val_dataset = datasets.ImageFolder(self.validation_dir, transform=test_transforms)
            self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        # Get class information
        self.class_names = test_dataset.classes
        self.num_classes = len(self.class_names)
        
        print(f"Test samples: {len(test_dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(val_dataset)}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Class names: {self.class_names}")
        print()
    
    def _load_model(self):
        """Load the trained model."""
        print("Loading trained model...")
        
        # Create model architecture (same as training)
        self.model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Replace classifier
        self.model.classifier = nn.Linear(self.model.classifier[1].in_features, self.num_classes)
        
        # Load trained weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        # Move to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from: {self.model_path}")
        print()
    
    def evaluate_dataset(self, data_loader, dataset_name="Test"):
        """
        Evaluate the model on a given dataset.
        
        Args:
            data_loader: DataLoader for the dataset
            dataset_name (str): Name of the dataset for display
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        print(f"Evaluating on {dataset_name} set...")
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(tqdm(data_loader, desc=f"{dataset_name} Evaluation")):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Store results
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                total_loss += loss.item()
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        avg_loss = total_loss / len(data_loader)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average=None, labels=range(self.num_classes)
        )
        
        # Macro and weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro'
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        results = {
            'dataset_name': dataset_name,
            'accuracy': accuracy,
            'loss': avg_loss,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1,
            'per_class_support': support,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
        }
        
        return results
    
    def plot_confusion_matrix(self, results, save_path=None):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(results['labels'], results['predictions'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {results["dataset_name"]} Set\nAccuracy: {results["accuracy"]:.4f}')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def plot_per_class_metrics(self, results, save_path=None):
        """Plot per-class precision, recall, and F1-score."""
        metrics_df = pd.DataFrame({
            'Class': self.class_names,
            'Precision': results['per_class_precision'],
            'Recall': results['per_class_recall'],
            'F1-Score': results['per_class_f1'],
            'Support': results['per_class_support']
        })
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot precision, recall, F1-score
        for i, metric in enumerate(['Precision', 'Recall', 'F1-Score']):
            axes[i].bar(self.class_names, metrics_df[metric])
            axes[i].set_title(f'Per-Class {metric}')
            axes[i].set_xlabel('Class')
            axes[i].set_ylabel(metric)
            axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(metrics_df[metric]):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Per-class metrics plot saved to: {save_path}")
        
        plt.show()
        
        return metrics_df
    
    def generate_classification_report(self, results, save_path=None):
        """Generate and save detailed classification report."""
        report = classification_report(
            results['labels'], 
            results['predictions'],
            target_names=self.class_names,
            digits=4
        )
        
        print(f"\n{results['dataset_name']} Set Classification Report:")
        print("=" * 60)
        print(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(f"{results['dataset_name']} Set Classification Report\n")
                f.write("=" * 60 + "\n")
                f.write(report)
            print(f"Classification report saved to: {save_path}")
        
        return report
    
    def save_results_summary(self, results_list, save_path="evaluation_summary.json"):
        """Save evaluation results summary to JSON file."""
        summary = {}
        
        for results in results_list:
            dataset_name = results['dataset_name']
            summary[dataset_name] = {
                'accuracy': float(results['accuracy']),
                'loss': float(results['loss']),
                'macro_precision': float(results['macro_precision']),
                'macro_recall': float(results['macro_recall']),
                'macro_f1': float(results['macro_f1']),
                'weighted_precision': float(results['weighted_precision']),
                'weighted_recall': float(results['weighted_recall']),
                'weighted_f1': float(results['weighted_f1']),
                'per_class_metrics': {}
            }
            
            for i, class_name in enumerate(self.class_names):
                summary[dataset_name]['per_class_metrics'][class_name] = {
                    'precision': float(results['per_class_precision'][i]),
                    'recall': float(results['per_class_recall'][i]),
                    'f1': float(results['per_class_f1'][i]),
                    'support': int(results['per_class_support'][i])
                }
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Evaluation summary saved to: {save_path}")
        return summary
    
    def run_complete_evaluation(self, save_plots=True, save_reports=True):
        """Run complete evaluation pipeline."""
        print("Starting Complete Model Evaluation")
        print("=" * 50)
        
        results_list = []
        
        # Evaluate test set
        test_results = self.evaluate_dataset(self.test_loader, "Test")
        results_list.append(test_results)
        
        # Evaluate validation set if available
        if self.val_loader:
            val_results = self.evaluate_dataset(self.val_loader, "Validation")
            results_list.append(val_results)
        
        # Generate reports and plots
        for results in results_list:
            dataset_name = results['dataset_name'].lower()
            
            print(f"\n{results['dataset_name']} Set Results:")
            print("-" * 30)
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"Loss: {results['loss']:.4f}")
            print(f"Macro F1-Score: {results['macro_f1']:.4f}")
            print(f"Weighted F1-Score: {results['weighted_f1']:.4f}")
            
            # Generate classification report
            if save_reports:
                report_path = f"{dataset_name}_classification_report.txt"
                self.generate_classification_report(results, report_path)
            else:
                self.generate_classification_report(results)
            
            # Plot confusion matrix
            if save_plots:
                cm_path = f"{dataset_name}_confusion_matrix.png"
                self.plot_confusion_matrix(results, cm_path)
            else:
                self.plot_confusion_matrix(results)
            
            # Plot per-class metrics
            if save_plots:
                metrics_path = f"{dataset_name}_per_class_metrics.png"
                metrics_df = self.plot_per_class_metrics(results, metrics_path)
            else:
                metrics_df = self.plot_per_class_metrics(results)
            
            # Save per-class metrics to CSV
            if save_reports:
                csv_path = f"{dataset_name}_per_class_metrics.csv"
                metrics_df.to_csv(csv_path, index=False)
                print(f"Per-class metrics CSV saved to: {csv_path}")
        
        # Save overall summary
        if save_reports:
            self.save_results_summary(results_list)
        
        print("\n" + "=" * 50)
        print("Model evaluation completed successfully!")
        
        return results_list


def main():
    """Main function to run model evaluation."""
    
    # Configuration
    model_path = "snake_classifier_mobilenetv2.pth"
    test_dir = "dataset/test"
    validation_dir = "dataset/validation"
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please make sure you have trained the model using train.py first.")
        return
    
    # Check if test directory exists
    if not os.path.exists(test_dir):
        print(f"Error: Test directory '{test_dir}' not found!")
        return
    
    # Create evaluator and run evaluation
    try:
        evaluator = ModelEvaluator(
            model_path=model_path,
            test_dir=test_dir,
            validation_dir=validation_dir if os.path.exists(validation_dir) else None
        )
        
        # Run complete evaluation
        results = evaluator.run_complete_evaluation(
            save_plots=True,
            save_reports=True
        )
        
        print("\nEvaluation files generated:")
        print("- Confusion matrices (PNG)")
        print("- Per-class metrics plots (PNG)")
        print("- Classification reports (TXT)")
        print("- Per-class metrics tables (CSV)")
        print("- Evaluation summary (JSON)")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return


if __name__ == "__main__":
    main() 