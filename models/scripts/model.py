import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ResNetLSTMClassifier(nn.Module):
    """
    ResNet + LSTM architecture for deepfake detection.
    
    Architecture:
    1. ResNet feature extractor (pretrained on ImageNet)
    2. LSTM for temporal modeling
    3. Classification head for authentic vs deepfake
    """
    
    def __init__(
        self,
        resnet_type: str = "resnet50",
        lstm_hidden_size: int = 512,
        lstm_num_layers: int = 2,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        freeze_resnet: bool = False
    ):
        super(ResNetLSTMClassifier, self).__init__()
        
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        # ResNet backbone for feature extraction
        if resnet_type == "resnet18":
            self.resnet = models.resnet18(pretrained=True)
            resnet_output_size = 512
        elif resnet_type == "resnet34":
            self.resnet = models.resnet34(pretrained=True)
            resnet_output_size = 512
        elif resnet_type == "resnet50":
            self.resnet = models.resnet50(pretrained=True)
            resnet_output_size = 2048
        elif resnet_type == "resnet101":
            self.resnet = models.resnet101(pretrained=True)
            resnet_output_size = 2048
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")
        
        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Freeze ResNet parameters if specified
        if freeze_resnet:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature reduction layer
        self.feature_reduction = nn.Linear(resnet_output_size, lstm_hidden_size)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_size, num_classes)
        )
        
        # Attention mechanism for sequence aggregation
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.Tanh(),
            nn.Linear(lstm_hidden_size, 1)
        )
        
        self.softmax = nn.Softmax(dim=1)
        
    def extract_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract features from individual frames using ResNet.
        
        Args:
            frames: Tensor of shape (batch_size, sequence_length, channels, height, width)
            
        Returns:
            Frame features of shape (batch_size, sequence_length, feature_dim)
        """
        batch_size, seq_len, channels, height, width = frames.shape
        
        # Reshape to process all frames at once
        frames_flat = frames.view(batch_size * seq_len, channels, height, width)
        
        # Extract features using ResNet
        with torch.set_grad_enabled(self.training):
            features = self.resnet(frames_flat)
            features = self.adaptive_pool(features)
            features = features.view(features.size(0), -1)
            features = self.feature_reduction(features)
        
        # Reshape back to sequence format
        features = features.view(batch_size, seq_len, -1)
        
        return features
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
            
        Returns:
            Tuple of (logits, probabilities, attention_weights)
        """
        # Extract frame-level features
        frame_features = self.extract_frame_features(x)
        
        # Process through LSTM
        lstm_out, (hidden, cell) = self.lstm(frame_features)
        
        # Apply attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum of LSTM outputs
        weighted_features = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Final classification
        logits = self.classifier(weighted_features)
        probabilities = self.softmax(logits)
        
        return logits, probabilities, attention_weights.squeeze(-1)
    
    def predict_frame_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get per-frame deepfake scores.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
            
        Returns:
            Frame scores of shape (batch_size, sequence_length)
        """
        frame_features = self.extract_frame_features(x)
        
        # Process each frame independently through classifier
        batch_size, seq_len, feature_dim = frame_features.shape
        frame_features_flat = frame_features.view(batch_size * seq_len, feature_dim)
        
        # Use the classifier to get frame-level predictions
        frame_logits = self.classifier(frame_features_flat)
        frame_probs = self.softmax(frame_logits)
        
        # Reshape and return deepfake probabilities (class 1)
        frame_scores = frame_probs[:, 1].view(batch_size, seq_len)
        
        return frame_scores


class ResNeXtLSTMClassifier(nn.Module):
    """
    ResNeXt + LSTM architecture for deepfake detection.
    
    Architecture:
    1. ResNeXt feature extractor (pretrained on ImageNet)
    2. LSTM for temporal modeling
    3. Classification head for authentic vs deepfake
    """
    
    def __init__(
        self,
        resnext_type: str = "resnext50_32x4d",
        lstm_hidden_size: int = 512,
        lstm_num_layers: int = 2,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        freeze_resnext: bool = False
    ):
        super(ResNeXtLSTMClassifier, self).__init__()
        
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        # ResNeXt backbone for feature extraction
        if resnext_type == "resnext50_32x4d":
            self.resnext = models.resnext50_32x4d(pretrained=True)
            resnext_output_size = 2048
        elif resnext_type == "resnext101_32x8d":
            self.resnext = models.resnext101_32x8d(pretrained=True)
            resnext_output_size = 2048
        else:
            raise ValueError(f"Unsupported ResNeXt type: {resnext_type}")
        
        # Remove the final classification layer
        self.resnext = nn.Sequential(*list(self.resnext.children())[:-1])
        
        # Freeze ResNeXt parameters if specified
        if freeze_resnext:
            for param in self.resnext.parameters():
                param.requires_grad = False
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature reduction layer
        self.feature_reduction = nn.Linear(resnext_output_size, lstm_hidden_size)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_size, num_classes)
        )
        
        # Attention mechanism for sequence aggregation
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.Tanh(),
            nn.Linear(lstm_hidden_size, 1)
        )
        
        self.softmax = nn.Softmax(dim=1)
        
    def extract_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract features from individual frames using ResNeXt.
        
        Args:
            frames: Tensor of shape (batch_size, sequence_length, channels, height, width)
            
        Returns:
            Frame features of shape (batch_size, sequence_length, feature_dim)
        """
        batch_size, seq_len, channels, height, width = frames.shape
        
        # Reshape to process all frames at once
        frames_flat = frames.view(batch_size * seq_len, channels, height, width)
        
        # Extract features using ResNeXt
        with torch.set_grad_enabled(self.training):
            features = self.resnext(frames_flat)
            features = self.adaptive_pool(features)
            features = features.view(features.size(0), -1)
            features = self.feature_reduction(features)
        
        # Reshape back to sequence format
        features = features.view(batch_size, seq_len, -1)
        
        return features
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
            
        Returns:
            Tuple of (logits, probabilities, attention_weights)
        """
        # Extract frame-level features
        frame_features = self.extract_frame_features(x)
        
        # Process through LSTM
        lstm_out, (hidden, cell) = self.lstm(frame_features)
        
        # Apply attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum of LSTM outputs
        weighted_features = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Final classification
        logits = self.classifier(weighted_features)
        probabilities = self.softmax(logits)
        
        return logits, probabilities, attention_weights.squeeze(-1)
    
    def predict_frame_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get per-frame deepfake scores.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
            
        Returns:
            Frame scores of shape (batch_size, sequence_length)
        """
        frame_features = self.extract_frame_features(x)
        
        # Process each frame independently through classifier
        batch_size, seq_len, feature_dim = frame_features.shape
        frame_features_flat = frame_features.view(batch_size * seq_len, feature_dim)
        
        # Use the classifier to get frame-level predictions
        frame_logits = self.classifier(frame_features_flat)
        frame_probs = self.softmax(frame_logits)
        
        # Reshape and return deepfake probabilities (class 1)
        frame_scores = frame_probs[:, 1].view(batch_size, seq_len)
        
        return frame_scores


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class ModelUtils:
    """Utility functions for model operations."""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def freeze_layers(model: nn.Module, layer_names: list):
        """Freeze specific layers in the model."""
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
    
    @staticmethod
    def unfreeze_layers(model: nn.Module, layer_names: list):
        """Unfreeze specific layers in the model."""
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
    
    @staticmethod
    def save_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        accuracy: float,
        filepath: str,
        metadata: Optional[dict] = None
    ):
        """Save model checkpoint with metadata."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            'metadata': metadata or {}
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    @staticmethod
    def load_checkpoint(
        model: nn.Module,
        filepath: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu'
    ) -> dict:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded: {filepath}")
        return checkpoint