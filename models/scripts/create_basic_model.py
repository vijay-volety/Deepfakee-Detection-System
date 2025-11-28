import torch
import torch.nn as nn
import torchvision.models as models
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleResNeXtLSTMClassifier(nn.Module):
    """
    Simplified ResNeXt + LSTM architecture for deepfake detection.
    """
    
    def __init__(
        self,
        resnext_type: str = "resnext50_32x4d",
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 1,
        num_classes: int = 2,
        dropout_rate: float = 0.3
    ):
        super(SimpleResNeXtLSTMClassifier, self).__init__()
        
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        # ResNeXt backbone for feature extraction
        if resnext_type == "resnext50_32x4d":
            self.resnext = models.resnext50_32x4d(pretrained=True)
            resnext_output_size = 2048
        else:
            raise ValueError(f"Unsupported ResNeXt type: {resnext_type}")
        
        # Remove the final classification layer
        self.resnext = nn.Sequential(*list(self.resnext.children())[:-1])
        
        # Freeze ResNeXt parameters
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
            bidirectional=False
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_size // 2, num_classes)
        )
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
            
        Returns:
            Tuple of (logits, probabilities)
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Reshape to process all frames at once
        frames_flat = x.view(batch_size * seq_len, channels, height, width)
        
        # Extract features using ResNeXt
        with torch.no_grad():
            features = self.resnext(frames_flat)
            features = self.adaptive_pool(features)
            features = features.view(features.size(0), -1)
            features = self.feature_reduction(features)
        
        # Reshape back to sequence format
        features = features.view(batch_size, seq_len, -1)
        
        # Process through LSTM
        lstm_out, (hidden, cell) = self.lstm(features)
        
        # Use the last output for classification
        final_features = lstm_out[:, -1, :]
        
        # Final classification
        logits = self.classifier(final_features)
        probabilities = self.softmax(logits)
        
        return logits, probabilities, None  # Return None for attention weights to match interface

def create_and_save_model():
    """Create a basic model and save it."""
    logger.info("Creating basic ResNeXt + LSTM model...")
    
    # Create model
    model = SimpleResNeXtLSTMClassifier(
        resnext_type="resnext50_32x4d",
        lstm_hidden_size=256,
        lstm_num_layers=1,
        num_classes=2,
        dropout_rate=0.3
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path("../checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save model
    checkpoint_path = checkpoint_dir / "basic_resnext_lstm.pth"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_version': '1.0.0-basic',
        'architecture': 'ResNeXt50-LSTM'
    }, checkpoint_path)
    
    logger.info(f"Model saved to {checkpoint_path}")
    logger.info("Basic model creation completed successfully!")

if __name__ == "__main__":
    create_and_save_model()