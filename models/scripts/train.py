import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import yaml
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from model import ResNetLSTMClassifier, ResNeXtLSTMClassifier, EarlyStopping, ModelUtils
from preprocessing import (
    DatasetBuilder, 
    DeepfakeDataset, 
    DataAugmentation
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepfakeTrainer:
    """Main trainer class for deepfake detection model."""
    
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
        self.criterion = self._build_criterion()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping_patience'],
            min_delta=config['training']['early_stopping_min_delta']
        )
        
        # Tracking
        self.best_val_acc = 0.0
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
                dropout_rate=model_config['dropout_rate'],
                freeze_resnet=model_config.get('freeze_resnet', False)
            )
        elif model_type == 'resnext_lstm':
            return ResNeXtLSTMClassifier(
                resnext_type=model_config['resnext_type'],
                lstm_hidden_size=model_config['lstm_hidden_size'],
                lstm_num_layers=model_config['lstm_num_layers'],
                num_classes=model_config['num_classes'],
                dropout_rate=model_config['dropout_rate'],
                freeze_resnext=model_config.get('freeze_resnext', False)
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer."""
        opt_config = self.config['optimizer']
        
        if opt_config['type'] == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        elif opt_config['type'] == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_config['type']}")
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        if 'scheduler' not in self.config:
            return None
            
        sch_config = self.config['scheduler']
        
        if sch_config['type'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sch_config['T_max']
            )
        elif sch_config['type'] == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=sch_config['factor'],
                patience=sch_config['patience']
            )
        else:
            return None
    
    def _build_criterion(self) -> nn.Module:
        """Build loss function."""
        loss_config = self.config.get('loss', {'type': 'cross_entropy'})
        
        if loss_config['type'] == 'cross_entropy':
            # Handle class imbalance if specified
            if 'class_weights' in loss_config:
                weights = torch.tensor(loss_config['class_weights']).float().to(self.device)
                return nn.CrossEntropyLoss(weight=weights)
            return nn.CrossEntropyLoss()
        elif loss_config['type'] == 'focal':
            return FocalLoss(
                alpha=loss_config.get('alpha', 1.0),
                gamma=loss_config.get('gamma', 2.0)
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_config['type']}")
    
    def _setup_logging(self):
        """Setup TensorBoard and W&B logging."""
        # TensorBoard
        log_dir = Path(self.config['paths']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))
        
        # W&B
        if self.config.get('wandb', {}).get('enabled', False):
            wandb.init(
                project=self.config['wandb']['project'],
                config=self.config,
                name=self.config['wandb'].get('name')
            )
    
    def _prepare_datasets(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare training, validation, and test datasets."""
        data_config = self.config['data']
        
        # Build datasets
        dataset_builder = DatasetBuilder(data_config['root_dir'])
        
        train_data = dataset_builder.build_combined_dataset(
            split='train',
            include_datasets=data_config['datasets']
        )
        
        val_data = dataset_builder.build_combined_dataset(
            split='val',
            include_datasets=data_config['datasets']
        )
        
        test_data = dataset_builder.build_combined_dataset(
            split='test',
            include_datasets=data_config['datasets']
        )
        
        # Create datasets
        train_transforms = DataAugmentation.get_training_transforms()
        val_transforms = DataAugmentation.get_validation_transforms()
        
        train_dataset = DeepfakeDataset(
            train_data,
            transform=train_transforms,
            sequence_length=data_config['sequence_length'],
            is_training=True
        )
        
        val_dataset = DeepfakeDataset(
            val_data,
            transform=val_transforms,
            sequence_length=data_config['sequence_length'],
            is_training=False
        )
        
        test_dataset = DeepfakeDataset(
            test_data,
            transform=val_transforms,
            sequence_length=data_config['sequence_length'],
            is_training=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            pin_memory=True
        )
        
        logger.info(f"Dataset sizes - Train: {len(train_dataset)}, "
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
            if self.config['training'].get('gradient_clipping', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clipping']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
                })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
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
                
                # Update progress bar
                if batch_idx % 10 == 0:
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
                    })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
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
        
        # Log metrics
        logger.info(f"Validation - Loss: {avg_loss:.4f}, "
                   f"Accuracy: {accuracy:.4f}, "
                   f"Precision: {precision:.4f}, "
                   f"Recall: {recall:.4f}, "
                   f"F1: {f1:.4f}, "
                   f"AUC: {auc:.4f}")
        
        # TensorBoard logging
        self.writer.add_scalar('Loss/Validation', avg_loss, epoch)
        self.writer.add_scalar('Accuracy/Validation', accuracy, epoch)
        self.writer.add_scalar('Precision/Validation', precision, epoch)
        self.writer.add_scalar('Recall/Validation', recall, epoch)
        self.writer.add_scalar('F1/Validation', f1, epoch)
        self.writer.add_scalar('AUC/Validation', auc, epoch)
        
        # W&B logging
        if self.config.get('wandb', {}).get('enabled', False):
            wandb.log({
                'val_loss': avg_loss,
                'val_accuracy': accuracy,
                'val_precision': precision,
                'val_recall': recall,
                'val_f1': f1,
                'val_auc': auc,
                'epoch': epoch
            })
        
        return avg_loss, accuracy
    
    def train(self):
        """Main training loop."""
        # Prepare datasets
        train_loader, val_loader, test_loader = self._prepare_datasets()
        
        # Training loop
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader, epoch)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
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
            
            # W&B logging
            if self.config.get('wandb', {}).get('enabled', False):
                wandb.log({
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })
            
            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                checkpoint_path = Path(self.config['paths']['checkpoint_dir'])
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                best_model_path = checkpoint_path / "best_model.pth"
                
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
                        'val_loss': val_loss
                    }
                )
                
                logger.info(f"Best model saved with validation accuracy: {val_acc:.4f}")
            
            # Update tracking
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
        
        # Test the best model
        logger.info("Training completed. Testing best model...")
        self._test_best_model(test_loader)
        
        # Close TensorBoard writer
        self.writer.close()
    
    def _test_best_model(self, test_loader: DataLoader):
        """Test the best model on the test set."""
        # Load best model
        checkpoint_path = Path(self.config['paths']['checkpoint_dir']) / "best_model.pth"
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
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
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
        logger.info("Test Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  AUC: {auc:.4f}")
        
        # Save test results
        results_path = Path(self.config['paths']['results_dir'])
        results_path.mkdir(parents=True, exist_ok=True)
        results_file = results_path / "test_results.txt"
        
        with open(results_file, 'w') as f:
            f.write(f"Test Results:\n")
            f.write(f"  Accuracy: {accuracy:.4f}\n")
            f.write(f"  Precision: {precision:.4f}\n")
            f.write(f"  Recall: {recall:.4f}\n")
            f.write(f"  F1 Score: {f1:.4f}\n")
            f.write(f"  AUC: {auc:.4f}\n")
        
        logger.info(f"Test results saved to {results_file}")


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--archive-only', action='store_true',
                       help='Train exclusively on archive dataset')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create trainer
    trainer = DeepfakeTrainer(config)
    
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