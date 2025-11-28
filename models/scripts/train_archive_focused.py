#!/usr/bin/env python3
"""
Archive-Focused Training Script
Specifically trains on the archive dataset to achieve 97% authentic accuracy for real images
and high deepfake accuracy for fake images.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import yaml
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from model import ResNetLSTMClassifier, ResNeXtLSTMClassifier, EarlyStopping, ModelUtils
from preprocessing import (
    DatasetBuilder, 
    DeepfakeDataset, 
    DataAugmentation
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArchiveFocusedTrainer:
    """Trainer focused exclusively on archive dataset performance."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=25,  # Increased patience for focused training
            min_delta=0.0001
        )
        
        # Tracking
        self.best_archive_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Setup logging
        self._setup_logging()
    
    def _build_model(self) -> nn.Module:
        """Build the model based on configuration."""
        model_config = self.config['model']
        model_type = model_config.get('type', 'resnet_lstm')
        
        if model_type == 'resnet_lstm':
            return ResNetLSTMClassifier(
                resnet_type=model_config['resnet_type'],
                lstm_hidden_size=model_config['lstm_hidden_size'],
                lstm_num_layers=model_config['lstm_num_layers'],
                num_classes=model_config['num_classes'],
                dropout_rate=0.3,  # Reduced dropout for focused training
                freeze_resnet=False
            )
        elif model_type == 'resnext_lstm':
            return ResNeXtLSTMClassifier(
                resnext_type=model_config['resnext_type'],
                lstm_hidden_size=model_config['lstm_hidden_size'],
                lstm_num_layers=model_config['lstm_num_layers'],
                num_classes=model_config['num_classes'],
                dropout_rate=0.3,  # Reduced dropout for focused training
                freeze_resnext=False
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer with lower learning rate for focused training."""
        return optim.Adam(
            self.model.parameters(),
            lr=0.00005,  # Lower learning rate for focused training
            weight_decay=1e-5
        )
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        return optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=30,
            gamma=0.5
        )
    
    def _setup_logging(self):
        """Setup TensorBoard logging."""
        log_dir = Path("./logs/archive_focused")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))
    
    def _prepare_archive_datasets(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare training, validation, and test datasets exclusively from archive."""
        # Build archive dataset
        dataset_builder = DatasetBuilder("./data")
        archive_train_data = dataset_builder.build_archive_dataset('train')
        archive_val_data = dataset_builder.build_archive_dataset('val')
        archive_test_data = dataset_builder.build_archive_dataset('test')
        
        # Create datasets
        train_transforms = DataAugmentation.get_training_transforms()
        val_transforms = DataAugmentation.get_validation_transforms()
        
        train_dataset = DeepfakeDataset(
            archive_train_data,
            transform=train_transforms,
            sequence_length=16,
            is_training=True
        )
        
        val_dataset = DeepfakeDataset(
            archive_val_data,
            transform=val_transforms,
            sequence_length=16,
            is_training=False
        )
        
        test_dataset = DeepfakeDataset(
            archive_test_data,
            transform=val_transforms,
            sequence_length=16,
            is_training=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Archive Dataset sizes - Train: {len(train_dataset)}, "
                   f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            frames = batch['frames'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, probabilities, attention_weights = self.model(frames)
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            if batch_idx % 5 == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
                })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Tuple[float, float, dict]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        real_predictions = []
        real_labels = []
        fake_predictions = []
        fake_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]")
            
            for batch_idx, batch in enumerate(pbar):
                frames = batch['frames'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits, probabilities, attention_weights = self.model(frames)
                
                # Compute loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Update metrics
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy()[:, 1])
                
                # Separate real and fake predictions for detailed analysis
                for i in range(len(labels)):
                    if labels[i].item() == 0:  # Real
                        real_predictions.append(predictions[i].item())
                        real_labels.append(labels[i].item())
                    else:  # Fake
                        fake_predictions.append(predictions[i].item())
                        fake_labels.append(labels[i].item())
                
                # Update progress bar
                if batch_idx % 5 == 0:
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
                    })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Calculate real and fake specific accuracies
        real_accuracy = accuracy_score(real_labels, real_predictions) if real_labels else 0.0
        fake_accuracy = accuracy_score(fake_labels, fake_predictions) if fake_labels else 0.0
        
        # Calculate additional metrics
        if len(np.unique(all_labels)) > 1:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='binary'
            )
            try:
                auc = roc_auc_score(all_labels, all_probabilities)
            except:
                auc = 0.0
        else:
            precision = recall = f1 = auc = 0.0
        
        # Detailed metrics
        metrics = {
            'real_accuracy': real_accuracy,
            'fake_accuracy': fake_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        # Log metrics
        logger.info(f"Validation - Loss: {avg_loss:.4f}, "
                   f"Accuracy: {accuracy:.4f}, "
                   f"Real Acc: {real_accuracy:.4f}, "
                   f"Fake Acc: {fake_accuracy:.4f}")
        
        # TensorBoard logging
        self.writer.add_scalar('Loss/Validation', avg_loss, epoch)
        self.writer.add_scalar('Accuracy/Validation', accuracy, epoch)
        self.writer.add_scalar('Accuracy/Real', real_accuracy, epoch)
        self.writer.add_scalar('Accuracy/Fake', fake_accuracy, epoch)
        self.writer.add_scalar('Precision/Validation', precision, epoch)
        self.writer.add_scalar('Recall/Validation', recall, epoch)
        self.writer.add_scalar('F1/Validation', f1, epoch)
        self.writer.add_scalar('AUC/Validation', auc, epoch)
        
        return avg_loss, accuracy, metrics
    
    def train(self):
        """Main training loop focused on archive dataset."""
        # Prepare datasets
        train_loader, val_loader, test_loader = self._prepare_archive_datasets()
        
        # Training loop
        num_epochs = 200  # Extended training for focused performance
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc, metrics = self.validate_epoch(val_loader, epoch)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Log epoch metrics
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Learning_Rate', 
                                 self.optimizer.param_groups[0]['lr'], epoch)
            
            # Early stopping based on real accuracy targeting 97%
            real_acc = metrics['real_accuracy']
            fake_acc = metrics['fake_accuracy']
            
            # Save best model when we achieve target accuracy
            if (real_acc >= 0.96 and fake_acc >= 0.90) or \
               (real_acc >= 0.95 and fake_acc >= 0.95) or \
               (real_acc >= 0.97):
                if val_acc > self.best_archive_acc:
                    self.best_archive_acc = val_acc
                    checkpoint_path = Path("./checkpoints/archive_focused")
                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                    best_model_path = checkpoint_path / "best_archive_model.pth"
                    
                    ModelUtils.save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        val_loss,
                        val_acc,
                        str(best_model_path),
                        {
                            'config': self.config,
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'metrics': metrics
                        }
                    )
                    
                    logger.info(f"Best archive model saved with validation accuracy: {val_acc:.4f}")
                    logger.info(f"Real accuracy: {real_acc:.4f}, Fake accuracy: {fake_acc:.4f}")
            
            # Also save if it's just better overall
            if val_acc > self.best_archive_acc:
                self.best_archive_acc = val_acc
            
            # Update tracking
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Check for early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Test the best model
        logger.info("Training completed. Testing best model...")
        self._test_best_model(test_loader)
        
        # Close TensorBoard writer
        self.writer.close()
    
    def _test_best_model(self, test_loader: DataLoader):
        """Test the best model on the test set."""
        # Load best model
        checkpoint_path = Path("./checkpoints/archive_focused/best_archive_model.pth")
        if checkpoint_path.exists():
            checkpoint = ModelUtils.load_checkpoint(
                self.model, 
                str(checkpoint_path), 
                device=self.device
            )
            logger.info("Loaded best model for testing")
        
        # Test
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        real_predictions = []
        real_labels = []
        fake_predictions = []
        fake_labels = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Testing")
            
            for batch in pbar:
                frames = batch['frames'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits, probabilities, attention_weights = self.model(frames)
                
                # Update metrics
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy()[:, 1])
                
                # Separate real and fake predictions for detailed analysis
                for i in range(len(labels)):
                    if labels[i].item() == 0:  # Real
                        real_predictions.append(predictions[i].item())
                        real_labels.append(labels[i].item())
                    else:  # Fake
                        fake_predictions.append(predictions[i].item())
                        fake_labels.append(labels[i].item())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        real_accuracy = accuracy_score(real_labels, real_predictions) if real_labels else 0.0
        fake_accuracy = accuracy_score(fake_labels, fake_predictions) if fake_labels else 0.0
        
        if len(np.unique(all_labels)) > 1:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='binary'
            )
            try:
                auc = roc_auc_score(all_labels, all_probabilities)
            except:
                auc = 0.0
        else:
            precision = recall = f1 = auc = 0.0
        
        # Log results
        logger.info("Final Test Results:")
        logger.info(f"  Overall Accuracy: {accuracy:.4f}")
        logger.info(f"  Real Accuracy: {real_accuracy:.4f}")
        logger.info(f"  Fake Accuracy: {fake_accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  AUC: {auc:.4f}")
        
        # Save test results
        results_path = Path("./results")
        results_path.mkdir(parents=True, exist_ok=True)
        results_file = results_path / "archive_focused_test_results.txt"
        
        with open(results_file, 'w') as f:
            f.write(f"Archive-Focused Test Results:\n")
            f.write(f"  Overall Accuracy: {accuracy:.4f}\n")
            f.write(f"  Real Accuracy: {real_accuracy:.4f}\n")
            f.write(f"  Fake Accuracy: {fake_accuracy:.4f}\n")
            f.write(f"  Precision: {precision:.4f}\n")
            f.write(f"  Recall: {recall:.4f}\n")
            f.write(f"  F1 Score: {f1:.4f}\n")
            f.write(f"  AUC: {auc:.4f}\n")
            f.write(f"\nTarget Achieved:\n")
            f.write(f"  97% Authentic for Real Dataset: {'YES' if real_accuracy >= 0.97 else 'NO'}\n")
            f.write(f"  High Deepfake for Fake Dataset: {'YES' if fake_accuracy >= 0.90 else 'NO'}\n")
        
        logger.info(f"Test results saved to {results_file}")


def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train deepfake detection model focused on archive dataset')
    parser.add_argument('--config', type=str, default='./configs/train_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create trainer
    trainer = ArchiveFocusedTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = ModelUtils.load_checkpoint(
            trainer.model, 
            args.resume, 
            trainer.optimizer, 
            trainer.device
        )
        logger.info(f"Resumed from epoch {checkpoint['epoch']}")
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()