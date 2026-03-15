"""
LSTM Model Evaluation Script
============================
Evaluates the LSTM (Deep Learning) phishing detection model individually
using a shuffled subset of the dataset.

Usage:
    python -m eval.evaluate_lstm
    or
    cd final_build && uv run python eval/evaluate_lstm.py
"""

import pandas as pd
import numpy as np
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from phishing_detector.classifiers import DLClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(__file__).parent / "evaluation_lstm.log",
            encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger("LSTMEvaluation")


class LSTMEvaluator:
    """
    Evaluates the LSTM (DL) model for phishing detection.
    """
    
    def __init__(
        self,
        dataset_path: str,
        sample_size: int = 600,
        random_seed: int = 42,
    ):
        """
        Initialize the evaluator.
        
        Args:
            dataset_path: Path to the CSV dataset
            sample_size: Number of samples to evaluate
            random_seed: Seed for reproducibility
        """
        self.dataset_path = Path(dataset_path)
        self.sample_size = sample_size
        self.random_seed = random_seed
        
        # Results storage
        self.results: List[Dict] = []
        self.start_time = None
        self.end_time = None
        
        logger.info(f"Initializing LSTM evaluator with {sample_size} samples")
        
    def load_and_sample_dataset(self) -> pd.DataFrame:
        """Load dataset, shuffle, and sample specified number of rows."""
        logger.info(f"Loading dataset from {self.dataset_path}")
        
        # Load the dataset
        df = pd.read_csv(self.dataset_path)
        logger.info(f"Dataset loaded: {len(df)} total rows")
        
        # Shuffle the dataset
        df_shuffled = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        logger.info("Dataset shuffled")
        
        # Sample the specified number of rows
        df_sample = df_shuffled.head(self.sample_size)
        logger.info(f"Sampled {len(df_sample)} rows")
        
        # Show class distribution in sample
        class_dist = df_sample['label'].value_counts()
        logger.info(f"Class distribution in sample: {class_dist.to_dict()}")
        
        return df_sample
    
    def evaluate(self, save_results: bool = True) -> Dict:
        """
        Run the evaluation.
        
        Args:
            save_results: Whether to save results to CSV and JSON files
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Load and sample data
        df_sample = self.load_and_sample_dataset()
        
        # Initialize LSTM classifier
        logger.info("Initializing LSTM (DL) Classifier...")
        
        # Get model paths
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        models_dir = project_root / "models"
        
        dl_model_path = str(models_dir / "lstm_phishing_best.pt")
        dl_metadata_path = str(models_dir / "lstm_model_metadata.json")
        
        try:
            classifier = DLClassifier(dl_model_path, dl_metadata_path)
            logger.info("LSTM Classifier initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LSTM classifier: {e}")
            return {}
        
        self.start_time = datetime.now()
        logger.info(f"Starting LSTM evaluation at {self.start_time}")
        
        # Process each URL
        for idx, row in df_sample.iterrows():
            url = row['url']
            true_label = row['label']  # 0 = legitimate, 1 = phishing
            
            try:
                # Classify the URL
                result = classifier.predict(url)
                
                # Map prediction to label (0 = legitimate, 1 = phishing)
                predicted_label = 1 if result.prediction.lower() == "phishing" else 0
                
                # Store result
                self.results.append({
                    'url': url,
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'prediction': result.prediction,
                    'confidence': result.confidence,
                    'inference_time': result.inference_time,
                    'error': result.error,
                })
                
                # Log progress
                correct = "✓" if predicted_label == true_label else "✗"
                logger.info(
                    f"[{idx + 1}/{self.sample_size}] {correct} "
                    f"URL: {url[:50]}... | "
                    f"True: {'phishing' if true_label else 'legit'} | "
                    f"Pred: {result.prediction} ({result.confidence:.1%})"
                )
                
            except Exception as e:
                logger.error(f"[{idx + 1}/{self.sample_size}] Error processing {url}: {e}")
                self.results.append({
                    'url': url,
                    'true_label': true_label,
                    'predicted_label': None,
                    'error': str(e),
                })
        
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"LSTM evaluation completed in {duration:.1f} seconds")
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Save results
        if save_results:
            self.save_results(metrics)
        
        return metrics
    
    def calculate_metrics(self) -> Dict:
        """Calculate evaluation metrics."""
        # Filter out errors
        valid_results = [r for r in self.results if r.get('predicted_label') is not None]
        
        if not valid_results:
            logger.error("No valid results to calculate metrics")
            return {}
        
        true_labels = np.array([r['true_label'] for r in valid_results])
        pred_labels = np.array([r['predicted_label'] for r in valid_results])
        
        # Basic metrics
        total = len(valid_results)
        correct = np.sum(true_labels == pred_labels)
        accuracy = correct / total
        
        # True Positives, False Positives, True Negatives, False Negatives
        tp = np.sum((true_labels == 1) & (pred_labels == 1))
        fp = np.sum((true_labels == 0) & (pred_labels == 1))
        tn = np.sum((true_labels == 0) & (pred_labels == 0))
        fn = np.sum((true_labels == 1) & (pred_labels == 0))
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Average inference time
        avg_inference_time = np.mean([
            r['inference_time'] for r in valid_results 
            if r.get('inference_time') is not None
        ])
        
        metrics = {
            'model_name': 'LSTM (BiLSTM-DL)',
            'total_samples': total,
            'valid_samples': len(valid_results),
            'error_samples': len(self.results) - len(valid_results),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'avg_inference_time_seconds': avg_inference_time,
            'total_evaluation_time_seconds': (self.end_time - self.start_time).total_seconds(),
            'evaluation_timestamp': self.start_time.isoformat(),
        }
        
        logger.info("=" * 60)
        logger.info("LSTM EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Model: LSTM (BiLSTM-DL)")
        logger.info(f"Total Samples: {total}")
        logger.info(f"Accuracy: {accuracy:.2%}")
        logger.info(f"Precision: {precision:.2%}")
        logger.info(f"Recall: {recall:.2%}")
        logger.info(f"F1 Score: {f1:.2%}")
        logger.info("-" * 60)
        logger.info(f"True Positives: {tp} | False Positives: {fp}")
        logger.info(f"True Negatives: {tn} | False Negatives: {fn}")
        logger.info("-" * 60)
        logger.info(f"Avg Inference Time: {avg_inference_time:.4f}s per URL")
        logger.info("=" * 60)
        
        return metrics
    
    def save_results(self, metrics: Dict):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent
        
        # Save detailed results to CSV
        results_df = pd.DataFrame(self.results)
        results_path = output_dir / f"lstm_evaluation_results_{timestamp}.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Detailed results saved to: {results_path}")
        
        # Save metrics to JSON
        metrics_path = output_dir / f"lstm_evaluation_metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to: {metrics_path}")
        
        # Also save a summary report
        report_path = output_dir / f"lstm_evaluation_report_{timestamp}.md"
        with open(report_path, 'w') as f:
            f.write("# LSTM Model Evaluation Report\n\n")
            f.write(f"**Model:** LSTM (BiLSTM-DL)\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Sample Size:** {metrics['total_samples']}\n\n")
            f.write("## Performance Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Accuracy | {metrics['accuracy']:.2%} |\n")
            f.write(f"| Precision | {metrics['precision']:.2%} |\n")
            f.write(f"| Recall | {metrics['recall']:.2%} |\n")
            f.write(f"| F1 Score | {metrics['f1_score']:.2%} |\n\n")
            f.write("## Confusion Matrix\n\n")
            f.write("| | Predicted Legitimate | Predicted Phishing |\n")
            f.write("|---|---|---|\n")
            f.write(f"| **Actual Legitimate** | {metrics['true_negatives']} (TN) | {metrics['false_positives']} (FP) |\n")
            f.write(f"| **Actual Phishing** | {metrics['false_negatives']} (FN) | {metrics['true_positives']} (TP) |\n\n")
            f.write("## Performance Statistics\n\n")
            f.write(f"- Average Inference Time: {metrics['avg_inference_time_seconds']:.4f}s per URL\n")
            f.write(f"- Total Evaluation Time: {metrics['total_evaluation_time_seconds']:.1f}s\n")
        logger.info(f"Report saved to: {report_path}")


def main():
    """Main entry point for LSTM evaluation."""
    # Get the dataset path
    script_dir = Path(__file__).parent
    dataset_path = script_dir / "Dataset.csv"
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        sys.exit(1)
    
    # Create evaluator
    evaluator = LSTMEvaluator(
        dataset_path=str(dataset_path),
        sample_size=600,
        random_seed=42,  # For reproducibility
    )
    
    # Run evaluation
    metrics = evaluator.evaluate(save_results=True)
    
    print("\n" + "=" * 60)
    print("LSTM EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"F1 Score: {metrics['f1_score']:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
