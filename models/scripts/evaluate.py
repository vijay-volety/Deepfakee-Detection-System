import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

from model import ResNetLSTMClassifier, ModelUtils
from preprocessing import DatasetBuilder, DeepfakeDataset, DataAugmentation
from train import load_config

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation utility."""
    
    def __init__(self, model_path: str, config_path: str, device: str = 'auto'):
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else device)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        logger.info(f"Model loaded on device: {self.device}")
        logger.info(f"Model parameters: {ModelUtils.count_parameters(self.model):,}")
    
    def _load_model(self, model_path: str) -> ResNetLSTMClassifier:
        """Load trained model from checkpoint."""
        model = ResNetLSTMClassifier(
            resnet_type=self.config['model']['resnet_type'],
            lstm_hidden_size=self.config['model']['lstm_hidden_size'],
            lstm_num_layers=self.config['model']['lstm_num_layers'],
            num_classes=self.config['model']['num_classes'],
            dropout_rate=self.config['model']['dropout_rate']
        )
        
        checkpoint = ModelUtils.load_checkpoint(model, model_path, device=self.device)
        model.to(self.device)
        
        return model
    
    def evaluate_dataset(self, dataset_split: str = 'test') -> Dict:
        """Evaluate model on specified dataset split."""
        # Build dataset
        dataset_builder = DatasetBuilder(self.config['data']['root_dir'])
        
        if dataset_split == 'test':
            data_list = dataset_builder.build_combined_dataset(
                split='test',
                include_datasets=self.config['data']['datasets']
            )
        elif dataset_split == 'val':
            data_list = dataset_builder.build_combined_dataset(
                split='val',
                include_datasets=self.config['data']['datasets']
            )
        else:
            raise ValueError(f"Unsupported dataset split: {dataset_split}")
        
        # Create dataset and dataloader
        transforms = DataAugmentation.get_validation_transforms()
        dataset = DeepfakeDataset(
            data_list,
            transform=transforms,
            sequence_length=self.config['data']['sequence_length'],
            is_training=False
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers']
        )
        
        # Evaluation
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_frame_scores = []
        all_attention_weights = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {dataset_split}"):
                frames = batch['frames'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits, probabilities, attention_weights = self.model(frames)
                frame_scores = self.model.predict_frame_scores(frames)
                
                # Collect results
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_frame_scores.extend(frame_scores.cpu().numpy())
                all_attention_weights.extend(attention_weights.cpu().numpy())
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            all_labels, 
            all_predictions, 
            all_probabilities
        )
        
        # Add frame-level analysis
        metrics['frame_analysis'] = self._analyze_frame_scores(
            all_frame_scores, 
            all_labels
        )
        
        # Add attention analysis
        metrics['attention_analysis'] = self._analyze_attention_weights(
            all_attention_weights,
            all_labels
        )
        
        return metrics
    
    def _calculate_metrics(
        self, 
        labels: List[int], 
        predictions: List[int], 
        probabilities: List[List[float]]
    ) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        
        # AUC
        probs_positive = [prob[1] for prob in probabilities]
        auc = roc_auc_score(labels, probs_positive)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = \
            precision_recall_fscore_support(labels, predictions, average=None)
        
        # Calibration
        calibration_results = self._evaluate_calibration(labels, probs_positive)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': {
                'authentic': {
                    'precision': precision_per_class[0],
                    'recall': recall_per_class[0],
                    'f1_score': f1_per_class[0],
                    'support': support[0]
                },
                'deepfake': {
                    'precision': precision_per_class[1],
                    'recall': recall_per_class[1],
                    'f1_score': f1_per_class[1],
                    'support': support[1]
                }
            },
            'calibration': calibration_results
        }
    
    def _evaluate_calibration(
        self, 
        labels: List[int], 
        probabilities: List[float]
    ) -> Dict:
        """Evaluate model calibration."""
        # Reliability diagram
        fraction_of_positives, mean_predicted_value = calibration_curve(
            labels, probabilities, n_bins=10
        )
        
        # Calibration error (ECE - Expected Calibration Error)
        bin_boundaries = np.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities >= bin_lower) & (probabilities < bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.array(labels)[in_bin].mean()
                avg_confidence_in_bin = np.array(probabilities)[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return {
            'expected_calibration_error': ece,
            'reliability_curve': {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            }
        }
    
    def _analyze_frame_scores(
        self, 
        frame_scores: List[List[float]], 
        labels: List[int]
    ) -> Dict:
        """Analyze per-frame deepfake scores."""
        authentic_scores = []
        deepfake_scores = []
        
        for scores, label in zip(frame_scores, labels):
            if label == 0:  # Authentic
                authentic_scores.extend(scores)
            else:  # Deepfake
                deepfake_scores.extend(scores)
        
        return {
            'authentic_frame_scores': {
                'mean': np.mean(authentic_scores),
                'std': np.std(authentic_scores),
                'median': np.median(authentic_scores),
                'min': np.min(authentic_scores),
                'max': np.max(authentic_scores)
            },
            'deepfake_frame_scores': {
                'mean': np.mean(deepfake_scores),
                'std': np.std(deepfake_scores),
                'median': np.median(deepfake_scores),
                'min': np.min(deepfake_scores),
                'max': np.max(deepfake_scores)
            }
        }
    
    def _analyze_attention_weights(
        self, 
        attention_weights: List[List[float]], 
        labels: List[int]
    ) -> Dict:
        """Analyze attention weight patterns."""
        authentic_attention = []
        deepfake_attention = []
        
        for weights, label in zip(attention_weights, labels):
            if label == 0:  # Authentic
                authentic_attention.append(weights)
            else:  # Deepfake
                deepfake_attention.append(weights)
        
        # Calculate statistics
        authentic_mean_attention = np.mean(authentic_attention, axis=0) if authentic_attention else []
        deepfake_mean_attention = np.mean(deepfake_attention, axis=0) if deepfake_attention else []
        
        return {
            'authentic_attention_pattern': authentic_mean_attention.tolist() if len(authentic_mean_attention) > 0 else [],
            'deepfake_attention_pattern': deepfake_mean_attention.tolist() if len(deepfake_mean_attention) > 0 else []
        }
    
    def evaluate_per_dataset(self) -> Dict:
        """Evaluate model performance on each dataset separately."""
        results = {}
        
        dataset_builder = DatasetBuilder(self.config['data']['root_dir'])
        
        for dataset_name in self.config['data']['datasets']:
            logger.info(f"Evaluating on {dataset_name}")
            
            # Build dataset
            if dataset_name == 'DFDC':
                data_list = dataset_builder.build_dfdc_dataset('test')
            elif dataset_name == 'FaceForensics++':
                data_list = dataset_builder.build_faceforensics_dataset('test')
            elif dataset_name == 'Celeb-DF':
                data_list = dataset_builder.build_celebdf_dataset('test')
            else:
                continue
            
            if not data_list:
                logger.warning(f"No test data found for {dataset_name}")
                continue
            
            # Create dataset and evaluate
            transforms = DataAugmentation.get_validation_transforms()
            dataset = DeepfakeDataset(
                data_list,
                transform=transforms,
                sequence_length=self.config['data']['sequence_length'],
                is_training=False
            )
            
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config['data']['batch_size'],
                shuffle=False,
                num_workers=self.config['data']['num_workers']
            )
            
            # Evaluate
            predictions = []
            probabilities = []
            labels = []
            
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
                    frames = batch['frames'].to(self.device)
                    batch_labels = batch['label'].to(self.device)
                    
                    logits, probs, _ = self.model(frames)
                    
                    pred = torch.argmax(logits, dim=1)
                    predictions.extend(pred.cpu().numpy())
                    probabilities.extend(probs.cpu().numpy())
                    labels.extend(batch_labels.cpu().numpy())
            
            # Calculate metrics
            metrics = self._calculate_metrics(labels, predictions, probabilities)
            metrics['num_samples'] = len(data_list)
            
            results[dataset_name] = metrics
        
        return results
    
    def generate_visualization_report(self, output_dir: str):
        """Generate visualization report with plots and analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Evaluate on test set
        test_metrics = self.evaluate_dataset('test')
        
        # Plot confusion matrix
        self._plot_confusion_matrix(
            test_metrics['confusion_matrix'],
            str(output_path / 'confusion_matrix.png')
        )
        
        # Plot calibration curve
        self._plot_calibration_curve(
            test_metrics['calibration'],
            str(output_path / 'calibration_curve.png')
        )
        
        # Plot frame score distributions
        self._plot_frame_score_distributions(
            test_metrics['frame_analysis'],
            str(output_path / 'frame_score_distributions.png')
        )
        
        # Save metrics as JSON
        with open(output_path / 'evaluation_metrics.json', 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        logger.info(f"Evaluation report saved to {output_path}")
    
    def _plot_confusion_matrix(self, cm: List[List[int]], save_path: str):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Authentic', 'Deepfake'],
            yticklabels=['Authentic', 'Deepfake']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration_curve(self, calibration_data: Dict, save_path: str):
        """Plot calibration curve."""
        plt.figure(figsize=(8, 6))
        
        fraction_of_positives = calibration_data['reliability_curve']['fraction_of_positives']
        mean_predicted_value = calibration_data['reliability_curve']['mean_predicted_value']
        
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Calibration Plot (ECE: {calibration_data["expected_calibration_error"]:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_frame_score_distributions(self, frame_analysis: Dict, save_path: str):
        """Plot frame score distributions."""
        plt.figure(figsize=(12, 6))
        
        # Extract data
        authentic_stats = frame_analysis['authentic_frame_scores']
        deepfake_stats = frame_analysis['deepfake_frame_scores']
        
        # Create violin plot
        data_to_plot = [
            [authentic_stats['mean']] * 100,  # Placeholder for visualization
            [deepfake_stats['mean']] * 100    # Placeholder for visualization
        ]
        
        plt.violinplot(data_to_plot, positions=[1, 2], showmeans=True)
        plt.xticks([1, 2], ['Authentic', 'Deepfake'])
        plt.ylabel('Frame Deepfake Score')
        plt.title('Distribution of Frame-level Deepfake Scores')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate deepfake detection model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training configuration file')
    parser.add_argument('--output', type=str, default='./evaluation_results',
                       help='Output directory for evaluation results')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'],
                       help='Dataset split to evaluate on')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model, args.config)
    
    # Run evaluation
    logger.info("Running comprehensive evaluation...")
    
    # Overall evaluation
    test_results = evaluator.evaluate_dataset(args.split)
    
    # Per-dataset evaluation
    per_dataset_results = evaluator.evaluate_per_dataset()
    
    # Generate visualization report
    evaluator.generate_visualization_report(args.output)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Overall Accuracy: {test_results['accuracy']:.4f}")
    print(f"Overall AUC: {test_results['auc']:.4f}")
    print(f"Overall F1 Score: {test_results['f1_score']:.4f}")
    print(f"Calibration Error: {test_results['calibration']['expected_calibration_error']:.4f}")
    
    print("\nPer-Dataset Results:")
    for dataset_name, results in per_dataset_results.items():
        print(f"{dataset_name}: Acc={results['accuracy']:.4f}, AUC={results['auc']:.4f}")
    
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == '__main__':
    main()