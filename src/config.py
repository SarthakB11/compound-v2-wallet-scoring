"""
Configuration file for the Compound V2 Wallet Scoring project.
"""
import os
import multiprocessing
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CACHE_DIR = os.path.join(PROCESSED_DATA_DIR, "cache")

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Performance optimization settings
PERFORMANCE = {
    "default_n_jobs": max(1, multiprocessing.cpu_count() - 1),  # Default to N-1 cores
    "batch_size": 10000,  # Process data in batches of this size
    "use_caching": True,  # Enable caching of intermediate results
    "cache_expiry_days": 7,  # Cache files older than this will be regenerated
}

# Data processing parameters
NUM_FILES_TO_PROCESS = 3  # Process the 3 largest files from the dataset

# Feature engineering parameters
TIME_WINDOWS = {
    "lifetime": None,  # All data
    "90d": 90,         # Last 90 days
    "30d": 30,         # Last 30 days
}

# Anomaly detection parameters
ANOMALY_DETECTION = {
    "model": "isolation_forest",  # Options: isolation_forest, one_class_svm
    "contamination": 0.05,        # Estimated proportion of anomalies
    "random_state": 42,
}

# Scoring parameters
SCORING = {
    "method": "percentile",       # Options: percentile, sigmoid, min_max
    "reverse": True,              # Lower raw scores are better (e.g., for anomaly scores)
    "sigmoid_params": {
        "k": 1.0,                 # Sigmoid steepness
        "x0": 0.0,                # Sigmoid midpoint
    },
}

# Feature weights for heuristic scoring
# Higher weights indicate stronger influence on the final score
# Positive weights contribute to higher scores (better behavior)
# Negative weights contribute to lower scores (worse behavior)
FEATURE_WEIGHTS = {
    # Transaction activity features
    "tx_count_total": 0.2,              # More activity is generally positive
    "borrow_to_supply_ratio_vol": -0.3, # High leverage is risky
    "repay_ratio": 0.4,                 # Higher repayment ratio is good
    
    # Temporal features
    "account_age_days": 0.3,            # Longer account age is good
    "days_since_last_tx": -0.2,         # Recent activity is good
    "activity_consistency_stddev": -0.2, # Consistent activity is good
    
    # Financial health features
    "avg_ltv_90d": -0.5,                # High LTV is risky
    "min_ltv_alltime": -0.3,            # Very low minimum LTV indicates close calls
    "liquidation_count_borrower": -0.8, # Being liquidated is bad
    "liquidation_count_liquidator": 0.0, # Being a liquidator is neutral
    "time_near_liquidation_pct": -0.6,  # Time spent near liquidation is risky
    
    # Protocol interaction features
    "market_diversity_count": 0.3,      # Diverse market participation is good
    "comp_earned_estimate_ratio": -0.1, # Heavy COMP farming could be risky
    "avg_gas_price_relative": 0.1,      # Higher gas prices might indicate urgency
    "complex_action_freq": 0.1,         # Complex strategies could indicate sophistication
    
    # Anomaly score
    "anomaly_score": -0.7,              # High anomaly scores indicate unusual behavior
}

# Output settings
OUTPUT_SETTINGS = {
    "top_n_wallets": 1000,              # Number of top wallets to include in the final output
    "output_filename": "wallet_scores.csv",
} 