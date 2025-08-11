#!/usr/bin/env python3
"""
Snake Image Classification Training Script
Author: PyTorch Expert Assistant
Description: Train a snake image classifier using MobileNetV2 transfer learning
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, models


def main():
    print("Starting Snake Image Classification Training")
    print("=" * 50)
    
    # =====================================
    # 1. Device Setup
    # =====================================
    # Automatically detect if CUDA-enabled GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # =====================================
    # 2. Data Loading and Preprocessing
    # =====================================
    print("Setting up data loaders...")
    
    # Define data directories
    train_dir = "dataset/train"
    validation_dir = "dataset/validation"
    test_dir = "dataset/test"
    
    # Check if directories exist
    for dir_path in [train_dir, validation_dir, test_dir]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} not found!")
    
    # Determine number of classes dynamically from train directory
    num_classes = len([d for d in os.listdir(train_dir) 
                      if os.path.isdir(os.path.join(train_dir, d))])
    print(f"Number of snake classes detected: {num_classes}")
    
    # ImageNet normalization values
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # Data transformation pipelines
    # Training transformations with data augmentation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    # Validation/Test transformations without augmentation
    val_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    # Create datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_dataset = datasets.ImageFolder(validation_dir, transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print()
    
    # =====================================
    # 3. Model Architecture (Transfer Learning)
    # =====================================
    print("Setting up MobileNetV2 model with transfer learning...")
    
    # Load pre-trained MobileNetV2
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    
    # Freeze all parameters in the convolutional base
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final classifier layer
    # MobileNetV2 classifier is a Linear layer with 1280 input features
    model.classifier = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # Move model to device
    model = model.to(device)
    
    # Verify only classifier parameters are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print()
    
    # =====================================
    # 4. Training Setup
    # =====================================
    print("Setting up training configuration...")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer - only optimize the new classifier layer parameters
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    print(f"Loss function: {criterion.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__} (lr=0.001)")
    print(f"Training epochs: 15")
    print()
    
    # =====================================
    # 5. Training & Validation Loop
    # =====================================
    print("Starting training...")
    print("=" * 70)
    
    num_epochs = 15
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 20)
        
        # Training phase
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} - "
                      f"Loss: {loss.item():.4f}")
        
        # Calculate training metrics
        train_loss = running_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in validation_loader:
                # Move data to device
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        val_loss = running_val_loss / len(validation_loader)
        val_accuracy = 100 * correct_val / total_val
        
        # Print epoch summary
        print(f"  Training   - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        print()
    
    print("Training completed!")
    print("=" * 50)
    
    # =====================================
    # 6. Model Saving
    # =====================================
    print("Saving trained model...")
    
    model_save_path = "snake_classifier_mobilenetv2.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    # Display class names for reference
    print(f"\nClass names: {train_dataset.classes}")
    
    print("\nTraining script completed successfully!")


if __name__ == "__main__":
    main() 