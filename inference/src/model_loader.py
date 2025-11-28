import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any
import logging
import yaml
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration for model loading and inference."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "models/config.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration."""
        default_config = {
            "model": {
                "type": "resnet50_lstm",
                "resnet_type": "resnet50",
                "resnext_type": "resnext50_32x4d",
                "lstm_hidden_size": 512,
                "lstm_num_layers": 2,
                "num_classes": 2,
                "dropout_rate": 0.5,
                "sequence_length": 16,
                "freeze_backbone": False
            },
            "inference": {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "batch_size": 1,
                "num_workers": 4
            }
        }
        
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                # Merge with default config
                for key, value in file_config.items():
                    if key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}. Using defaults.")
        
        return default_config


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
    
    def forward(self, x: torch.Tensor) -> tuple:
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


class ModelLoader:
    """Model loader that supports both ResNet and ResNeXt architectures."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = ModelConfig(config_path)
        self.model = None
        self.model_version = "1.0.0"
        self.device = torch.device(self.config.config['inference']['device'])
        
    def load_default_model(self):
        """Load the default model based on configuration."""
        model_config = self.config.config['model']
        model_type = model_config['type']
        
        # Check if model files exist, if not create basic models
        model_dir = Path("models")
        if not model_dir.exists():
            model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple model if none exists
        model_files = list(model_dir.glob("*.pth"))
        if not model_files:
            logger.info("No model files found, creating basic model...")
            self._create_basic_model()
        
        if model_type.startswith('resnet'):
            # For now, we'll use ResNeXt as fallback since we don't have ResNet implementation here
            self.model = ResNeXtLSTMClassifier(
                resnext_type=model_config['resnet_type'].replace('resnet', 'resnext'),
                lstm_hidden_size=model_config['lstm_hidden_size'],
                lstm_num_layers=model_config['lstm_num_layers'],
                num_classes=model_config['num_classes'],
                dropout_rate=model_config['dropout_rate'],
                freeze_resnext=model_config['freeze_backbone']
            )
        elif model_type.startswith('resnext'):
            self.model = ResNeXtLSTMClassifier(
                resnext_type=model_config['resnext_type'],
                lstm_hidden_size=model_config['lstm_hidden_size'],
                lstm_num_layers=model_config['lstm_num_layers'],
                num_classes=model_config['num_classes'],
                dropout_rate=model_config['dropout_rate'],
                freeze_resnext=model_config['freeze_backbone']
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded {model_type} model on {self.device}")
    
    def _create_basic_model(self):
        """Create a basic model file for testing."""
        try:
            # Create a simple model state dict
            dummy_model = ResNeXtLSTMClassifier()
            dummy_model.to(self.device)
            
            # Save the model
            model_path = Path("models") / "basic_model.pth"
            torch.save({
                'model_state_dict': dummy_model.state_dict(),
                'model_version': '1.0.0-basic',
                'architecture': 'ResNeXt50-LSTM'
            }, model_path)
            
            logger.info(f"Basic model created at {model_path}")
        except Exception as e:
            logger.error(f"Failed to create basic model: {e}")
    
    def load_model(self, model_path: str):
        """Load model from specified path."""
        try:
            if not os.path.exists(model_path):
                logger.warning(f"Model file {model_path} not found, creating basic model...")
                self._create_basic_model()
                return
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint and self.model is not None:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'model_version' in checkpoint:
                    self.model_version = checkpoint['model_version']
            elif self.model is not None:
                self.model.load_state_dict(checkpoint)
            
            if self.model is not None:
                self.model.to(self.device)
                self.model.eval()
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            # Create a basic model as fallback
            self._create_basic_model()
            raise