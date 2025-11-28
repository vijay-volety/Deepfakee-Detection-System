# Archive Dataset Training Guide

This guide explains how to train the deepfake detection model specifically on the archive dataset to achieve the target accuracy requirements:

- 97% authentic percentage for real dataset files
- High deepfake percentage for fake dataset files

## Directory Structure

The archive dataset should be organized as follows:
```
archive/
└── Dataset/
    ├── Train/
    │   ├── Real/
    │   └── Fake/
    ├── Validation/
    │   ├── Real/
    │   └── Fake/
    └── Test/
        ├── Real/
        └── Fake/
```

## Training Scripts

We provide three specialized training scripts:

### 1. Standard Training with Archive Dataset
Uses the existing [train.py](scripts/train.py) script with archive dataset included:

```bash
python train.py --config configs/train_config.yaml
```

### 2. Archive-Specialized Training
Uses the enhanced training script that focuses on archive dataset performance:

```bash
python train_archive_specialized.py --config configs/train_config.yaml
```

### 3. Archive-Focused Training (Recommended)
Trains exclusively on the archive dataset for maximum accuracy on this specific dataset:

```bash
python train_archive_focused.py --config configs/train_config.yaml
```

## Configuration

The training configuration has been updated to include the archive dataset by default in [train_config.yaml](configs/train_config.yaml):

```yaml
data:
  datasets: ["DFDC", "FaceForensics++", "Celeb-DF", "Archive"]
```

## Expected Results

When training with the archive-focused approach, you should expect:

- **Real/Real Dataset Accuracy**: ≥ 97%
- **Fake/Fake Dataset Accuracy**: ≥ 90%
- **Overall Validation Accuracy**: ≥ 95%

## Model Outputs

Trained models are saved in the `checkpoints/` directory:
- `best_archive_model.pth` - Best model based on archive dataset performance
- `best_general_model.pth` - Best model based on overall performance

## Testing

To verify the trained model performance:

```bash
python verify_archive_dataset.py
```

This script will check that the archive dataset is properly integrated and can be loaded correctly.

## Integration with Existing System

All modifications preserve existing functionality:
- The original training scripts continue to work as before
- The archive dataset is added as an additional data source
- No breaking changes to existing APIs or data structures