"""
Wallet scoring pipeline for Compound V2.
"""
import os
import argparse
import logging
import time
from pathlib import Path
import pandas as pd

from src.data.loader import DataLoader
from src.preprocessing.preprocessor import Preprocessor
from src.features.feature_engineering import FeatureEngineer
from src.scoring.heuristic_scorer import HeuristicScorer
from src.scoring.final_scorer import FinalScorer
from src.advanced.feature_selection import FeatureSelector
from src.advanced.hyperparameter_tuning import HyperparameterTuner
from src.advanced.deep_learning import DeepLearningModel
from src.advanced.model_integration import ModelIntegrator
import src.config as config
from src.utils.caching import clear_cache, get_cache_size

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.LOG_DIR, "pipeline.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_pipeline(data_dir=None, output_dir=None, clear_cache_first=False,
                steps=None, use_advanced=False):
    """
    Run the wallet scoring pipeline.
    
    Args:
        data_dir (str, optional): Directory containing input data
        output_dir (str, optional): Directory to save results
        clear_cache_first (bool, optional): Whether to clear cache before running
        steps (list, optional): Specific pipeline steps to run
        use_advanced (bool, optional): Whether to use advanced optimization components
    """
    start_time = time.time()
    
    # Use default directories if not provided
    data_dir = data_dir or config.DATA_DIR
    output_dir = output_dir or config.RESULTS_DIR
    
    logger.info("Starting wallet scoring pipeline")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create necessary directories
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear cache if requested
    if clear_cache_first:
        logger.info("Clearing cache")
        clear_cache()
    
    # Initialize components
    data_loader = DataLoader(data_dir)
    preprocessor = Preprocessor()
    feature_engineer = FeatureEngineer()
    heuristic_scorer = HeuristicScorer()
    final_scorer = FinalScorer()
    
    # Advanced components
    if use_advanced:
        feature_selector = FeatureSelector()
        hyperparameter_tuner = HyperparameterTuner()
        deep_learning_model = DeepLearningModel()
        model_integrator = ModelIntegrator()
    
    # Define pipeline steps
    default_steps = ['load', 'preprocess', 'engineer', 'score', 'finalize']
    advanced_steps = ['select_features', 'tune_hyperparameters', 'train_deep_learning', 'integrate_models']
    
    # Use specified steps or all steps
    if steps:
        steps_to_run = steps
    else:
        steps_to_run = default_steps
        if use_advanced:
            steps_to_run += advanced_steps
    
    logger.info(f"Pipeline steps to run: {steps_to_run}")
    
    # Run the pipeline steps
    
    # 1. Load Data
    if 'load' in steps_to_run:
        logger.info("Step 1: Loading data")
        transactions_df = data_loader.load_transactions()
        users_df = data_loader.load_users()
        
        # Save intermediate results
        transactions_df.to_parquet(os.path.join(config.PROCESSED_DATA_DIR, "transactions.parquet"))
        users_df.to_parquet(os.path.join(config.PROCESSED_DATA_DIR, "users.parquet"))
        
        logger.info(f"Loaded {len(transactions_df)} transactions and {len(users_df)} users")
    else:
        logger.info("Skipping data loading step")
        transactions_df = pd.read_parquet(os.path.join(config.PROCESSED_DATA_DIR, "transactions.parquet"))
        users_df = pd.read_parquet(os.path.join(config.PROCESSED_DATA_DIR, "users.parquet"))
    
    # 2. Preprocess Data
    if 'preprocess' in steps_to_run:
        logger.info("Step 2: Preprocessing data")
        processed_df = preprocessor.preprocess(transactions_df, users_df)
        
        # Save intermediate results
        processed_df.to_parquet(os.path.join(config.PROCESSED_DATA_DIR, "processed_data.parquet"))
        
        logger.info(f"Preprocessed data with {len(processed_df)} records")
    else:
        logger.info("Skipping preprocessing step")
        processed_df = pd.read_parquet(os.path.join(config.PROCESSED_DATA_DIR, "processed_data.parquet"))
    
    # 3. Engineer Features
    if 'engineer' in steps_to_run:
        logger.info("Step 3: Engineering features")
        wallet_features = feature_engineer.engineer_features(processed_df)
        
        # Save intermediate results
        wallet_features.to_parquet(os.path.join(config.PROCESSED_DATA_DIR, "wallet_features.parquet"))
        
        logger.info(f"Engineered features for {len(wallet_features)} wallets")
    else:
        logger.info("Skipping feature engineering step")
        wallet_features = pd.read_parquet(os.path.join(config.PROCESSED_DATA_DIR, "wallet_features.parquet"))
    
    # 4. Score Wallets with Heuristics
    if 'score' in steps_to_run:
        logger.info("Step 4: Scoring wallets with heuristics")
        wallet_scores = heuristic_scorer.score_wallets(wallet_features)
        
        # Save intermediate results
        wallet_scores.to_parquet(os.path.join(config.PROCESSED_DATA_DIR, "wallet_scores.parquet"))
        
        logger.info(f"Scored {len(wallet_scores)} wallets")
    else:
        logger.info("Skipping heuristic scoring step")
        wallet_scores = pd.read_parquet(os.path.join(config.PROCESSED_DATA_DIR, "wallet_scores.parquet"))
    
    # 5. Finalize Scores
    if 'finalize' in steps_to_run:
        logger.info("Step 5: Finalizing scores")
        final_scores = final_scorer.finalize_scores(wallet_scores, wallet_features)
        
        # Save final results
        final_scores.to_csv(os.path.join(output_dir, "wallet_scores.csv"), index=False)
        
        logger.info(f"Finalized scores for {len(final_scores)} wallets")
    else:
        logger.info("Skipping score finalization step")
    
    # Advanced optimization steps
    if use_advanced:
        # 6. Select Features
        if 'select_features' in steps_to_run:
            logger.info("Step 6: Selecting optimal features")
            selected_features = feature_selector.select_optimal_features(method='combined', n_features=15)
            logger.info(f"Selected {len(selected_features)} features")
        
        # 7. Tune Hyperparameters
        if 'tune_hyperparameters' in steps_to_run:
            logger.info("Step 7: Tuning hyperparameters")
            for model_name in ['random_forest', 'gradient_boosting', 'ridge']:
                logger.info(f"Tuning {model_name}")
                hyperparameter_tuner.run_tuning(model_name, method='optuna', use_selected_features=True)
        
        # 8. Train Deep Learning Models
        if 'train_deep_learning' in steps_to_run:
            logger.info("Step 8: Training deep learning models")
            # Train standard architecture
            deep_learning_model.run_experiment(architecture='standard', use_selected_features=True)
            
            # Try sequence model if available
            try:
                deep_learning_model.run_sequence_experiment(use_selected_features=True)
            except Exception as e:
                logger.warning(f"Error running sequence experiment: {e}")
        
        # 9. Integrate Models
        if 'integrate_models' in steps_to_run:
            logger.info("Step 9: Integrating models")
            integrated_scores = model_integrator.run_integration_pipeline(use_selected_features=True)
            
            # Save final integrated results
            integrated_scores.to_csv(os.path.join(output_dir, "integrated_wallet_scores.csv"), index=False)
            
            logger.info(f"Integrated scores for {len(integrated_scores)} wallets")
    
    # Calculate and log execution time
    execution_time = time.time() - start_time
    logger.info(f"Pipeline execution completed in {execution_time:.2f} seconds")
    
    # Log cache size
    cache_size = get_cache_size()
    logger.info(f"Current cache size: {cache_size / (1024*1024):.2f} MB")
    
    return "Pipeline execution completed successfully"

def main():
    """
    Main function to run the wallet scoring pipeline from command line.
    """
    parser = argparse.ArgumentParser(description="Run the wallet scoring pipeline")
    parser.add_argument("--data-dir", help="Directory containing input data")
    parser.add_argument("--output-dir", help="Directory to save results")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache before running")
    parser.add_argument("--steps", help="Comma-separated list of pipeline steps to run")
    parser.add_argument("--advanced", action="store_true", help="Use advanced optimization components")
    
    args = parser.parse_args()
    
    # Parse steps if provided
    steps = args.steps.split(",") if args.steps else None
    
    # Run the pipeline
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        clear_cache_first=args.clear_cache,
        steps=steps,
        use_advanced=args.advanced
    )

if __name__ == "__main__":
    main() 