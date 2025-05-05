"""
Main module for the Compound V2 wallet scoring project.
Orchestrates the entire pipeline from data loading to scoring.
"""
import os
import sys
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.anomaly_detector import AnomalyDetector
from src.models.heuristic_scorer import HeuristicScorer
from src.scoring.scorer import WalletScorer
from src.data.download import download_dataset, extract_largest_files
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
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    
    # Log system information
    logger.info(f"Starting Compound V2 wallet scoring pipeline")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Data directory: {config.DATA_DIR}")
    logger.info(f"Results directory: {config.RESULTS_DIR}")
    
    # Check if raw data exists
    if not os.path.exists(config.RAW_DATA_DIR) or not os.listdir(config.RAW_DATA_DIR):
        logger.error("No data found in raw directory. Please run use_real_data.sh first.")
        sys.exit(1)
    
    # Log the real data files being used
    files = os.listdir(config.RAW_DATA_DIR)
    logger.info(f"Using {len(files)} real Compound V2 data files for analysis:")
    for file in files:
        file_path = os.path.join(config.RAW_DATA_DIR, file)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"  - {file} ({file_size_mb:.2f} MB)")

def run_pipeline(skip_to=None, optimize=True):
    """
    Run the complete wallet scoring pipeline.
    
    Args:
        skip_to (str, optional): Stage to skip to (for resuming pipeline)
        optimize (bool): Whether to use optimized algorithms
    """
    start_time = time.time()
    
    # Setup environment
    setup()
    
    try:
        # 1. Load and process raw data
        if skip_to is None or skip_to == 'load':
            logger.info("STAGE 1: Loading Data")
            data_loader = DataLoader()
            transaction_df = data_loader.load_and_process_data(optimize=optimize)
            logger.info(f"Loaded {len(transaction_df)} transactions")
        
        # 2. Preprocess data
        if skip_to is None or skip_to in ('load', 'preprocess'):
            logger.info("STAGE 2: Preprocessing Data")
            preprocessor = DataPreprocessor()
            transaction_df, wallet_df = preprocessor.process(optimize=optimize)
            logger.info(f"Preprocessed {len(transaction_df)} transactions for {len(wallet_df)} wallets")
        
        # 3. Generate features
        if skip_to is None or skip_to in ('load', 'preprocess', 'features'):
            logger.info("STAGE 3: Feature Engineering")
            feature_engineer = FeatureEngineer()
            features_df = feature_engineer.generate_all_features(optimize=optimize)
            logger.info(f"Generated features for {len(features_df)} wallets")
        
        # 4. Anomaly detection
        if skip_to is None or skip_to in ('load', 'preprocess', 'features', 'anomaly'):
            logger.info("STAGE 4: Anomaly Detection")
            anomaly_detector = AnomalyDetector()
            anomaly_df = anomaly_detector.detect_anomalies(optimize=optimize)
            logger.info(f"Detected anomalies in {anomaly_df['is_anomaly'].sum()} of {len(anomaly_df)} wallets")
        
        # 5. Heuristic scoring
        if skip_to is None or skip_to in ('load', 'preprocess', 'features', 'anomaly', 'heuristic'):
            logger.info("STAGE 5: Heuristic Scoring")
            heuristic_scorer = HeuristicScorer()
            heuristic_df = heuristic_scorer.score_wallets(optimize=optimize)
            logger.info(f"Scored {len(heuristic_df)} wallets using heuristic model")
        
        # 6. Final scoring and output
        if skip_to is None or skip_to in ('load', 'preprocess', 'features', 'anomaly', 'heuristic', 'score'):
            logger.info("STAGE 6: Final Scoring and Output")
            wallet_scorer = WalletScorer()
            output_df = wallet_scorer.generate_scores(optimize=optimize)
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
    parser.add_argument('--no-optimize', action='store_true', help='Disable algorithmic optimizations')
    
    return parser.parse_args()

def main():
    """
    Main function.
    """
    args = parse_arguments()
    run_pipeline(skip_to=args.skip_to, optimize=not args.no_optimize)

if __name__ == "__main__":
    main() 