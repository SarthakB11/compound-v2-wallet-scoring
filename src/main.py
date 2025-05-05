"""
Main module for the Compound V2 wallet scoring project.
Orchestrates the entire pipeline from data loading to scoring.
"""
import os
import logging
import argparse
import time
from datetime import datetime

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.anomaly_detector import AnomalyDetector
from src.models.heuristic_scorer import HeuristicScorer
from src.scoring.scorer import WalletScorer
import src.config as config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", f"wallet_scoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def setup():
    """
    Set up the project environment.
    """
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    
    # Log system information
    logger.info(f"Starting Compound V2 wallet scoring pipeline")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Data directory: {config.DATA_DIR}")
    logger.info(f"Results directory: {config.RESULTS_DIR}")

def run_pipeline(skip_to=None):
    """
    Run the complete wallet scoring pipeline.
    
    Args:
        skip_to (str, optional): Stage to skip to (for resuming pipeline)
    """
    start_time = time.time()
    
    # Setup environment
    setup()
    
    try:
        # 1. Load and process raw data
        if skip_to is None or skip_to == 'load':
            logger.info("STAGE 1: Loading Data")
            data_loader = DataLoader()
            transaction_df = data_loader.load_and_process_data()
            logger.info(f"Loaded {len(transaction_df)} transactions")
        
        # 2. Preprocess data
        if skip_to is None or skip_to in ('load', 'preprocess'):
            logger.info("STAGE 2: Preprocessing Data")
            preprocessor = DataPreprocessor()
            transaction_df, wallet_df = preprocessor.process()
            logger.info(f"Preprocessed {len(transaction_df)} transactions for {len(wallet_df)} wallets")
        
        # 3. Generate features
        if skip_to is None or skip_to in ('load', 'preprocess', 'features'):
            logger.info("STAGE 3: Feature Engineering")
            feature_engineer = FeatureEngineer()
            features_df = feature_engineer.generate_all_features()
            logger.info(f"Generated features for {len(features_df)} wallets")
        
        # 4. Anomaly detection
        if skip_to is None or skip_to in ('load', 'preprocess', 'features', 'anomaly'):
            logger.info("STAGE 4: Anomaly Detection")
            anomaly_detector = AnomalyDetector()
            anomaly_df = anomaly_detector.detect_anomalies()
            logger.info(f"Detected anomalies in {anomaly_df['is_anomaly'].sum()} of {len(anomaly_df)} wallets")
        
        # 5. Heuristic scoring
        if skip_to is None or skip_to in ('load', 'preprocess', 'features', 'anomaly', 'heuristic'):
            logger.info("STAGE 5: Heuristic Scoring")
            heuristic_scorer = HeuristicScorer()
            heuristic_df = heuristic_scorer.score_wallets()
            logger.info(f"Scored {len(heuristic_df)} wallets using heuristic model")
        
        # 6. Final scoring and output
        if skip_to is None or skip_to in ('load', 'preprocess', 'features', 'anomaly', 'heuristic', 'score'):
            logger.info("STAGE 6: Final Scoring and Output")
            wallet_scorer = WalletScorer()
            output_df = wallet_scorer.generate_scores()
            logger.info(f"Generated final scores for top {len(output_df)} wallets")
        
        # Calculate runtime
        end_time = time.time()
        runtime = end_time - start_time
        logger.info(f"Pipeline completed successfully in {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Compound V2 Wallet Scoring Pipeline")
    
    parser.add_argument('--skip-to', type=str, choices=['load', 'preprocess', 'features', 'anomaly', 'heuristic', 'score'],
                        help='Skip to a specific stage in the pipeline (for resuming)')
    
    return parser.parse_args()

def main():
    """
    Main function.
    """
    args = parse_arguments()
    run_pipeline(skip_to=args.skip_to)

if __name__ == "__main__":
    main() 