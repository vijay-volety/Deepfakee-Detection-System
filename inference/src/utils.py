import logging
import sys
from typing import Dict, Any
import yaml
from pathlib import Path

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent adding multiple handlers if logger already exists
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


class ModelConfig:
    """Configuration for model loading and inference."""
    
    def __init__(self, config_path: str = "models/config.yaml"):
        self.config_path = config_path
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
                "device": "cuda",
                "batch_size": 1,
                "num_workers": 4
            }
        }
        
        # Try to import torch to check CUDA availability
        try:
            import torch
            default_config["inference"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            default_config["inference"]["device"] = "cpu"
        
        config_file = Path(self.config_path)
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                # Merge with default config
                for key, value in file_config.items():
                    if key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
            except Exception as e:
                logger = setup_logger(__name__)
                logger.warning(f"Failed to load config file: {e}. Using defaults.")
        
        return default_config