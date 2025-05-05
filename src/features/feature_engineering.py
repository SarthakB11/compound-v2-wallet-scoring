"""
Feature engineering for wallet behavior analysis.
"""
import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
import pickle

from src.utils.helpers import calculate_time_diff_days
import src.config as config

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Class for generating features from wallet transaction data.
    """
    
    def __init__(self, processed_data_dir=None):
        """
        Initialize the FeatureEngineer.
        
        Args:
            processed_data_dir (str, optional): Directory containing processed data
        """
        self.processed_data_dir = processed_data_dir or config.PROCESSED_DATA_DIR
        
        # Time windows for feature calculation
        self.time_windows = config.TIME_WINDOWS
        
        # Cache for feature data
        self._cache = {}
        
        # Ensure directory exists
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(config.CACHE_DIR, exist_ok=True)
    
    def load_data(self):
        """
        Load the processed transaction and wallet data.
        
        Returns:
            tuple: (transaction_df, wallet_df)
        """
        # Check cache first
        if 'transaction_df' in self._cache and 'wallet_df' in self._cache:
            logger.info("Using cached transaction and wallet data")
            return self._cache['transaction_df'], self._cache['wallet_df']
        
        # Load transaction data
        tx_file_path = os.path.join(self.processed_data_dir, "cleaned_transactions.parquet")
        if not os.path.exists(tx_file_path):
            raise FileNotFoundError(f"Transaction data file not found: {tx_file_path}")
        
        transaction_df = pd.read_parquet(tx_file_path)
        logger.info(f"Loaded {len(transaction_df)} transactions from {tx_file_path}")
        
        # Load wallet summary data
        wallet_file_path = os.path.join(self.processed_data_dir, "wallet_summary.parquet")
        if not os.path.exists(wallet_file_path):
            raise FileNotFoundError(f"Wallet data file not found: {wallet_file_path}")
        
        wallet_df = pd.read_parquet(wallet_file_path)
        logger.info(f"Loaded summary data for {len(wallet_df)} wallets from {wallet_file_path}")
        
        # Store in cache
        self._cache['transaction_df'] = transaction_df
        self._cache['wallet_df'] = wallet_df
        
        return transaction_df, wallet_df
    
    def generate_transaction_activity_features(self, transaction_df, wallet_df, optimize=True):
        """
        Generate features related to transaction activity.
        
        Args:
            transaction_df (pd.DataFrame): Transaction data
            wallet_df (pd.DataFrame): Wallet summary data
            optimize (bool): Whether to use optimized algorithms
            
        Returns:
            pd.DataFrame: DataFrame with transaction activity features
        """
        logger.info("Generating transaction activity features...")
        
        # Create a copy of the wallet DataFrame
        wallet_features = wallet_df.copy()
        
        # Check for cached features if optimization is enabled
        cache_file = os.path.join(config.CACHE_DIR, "tx_activity_features.pkl")
        if optimize and os.path.exists(cache_file):
            try:
                # Check if cache is newer than source data
                tx_file_path = os.path.join(self.processed_data_dir, "cleaned_transactions.parquet")
                wallet_file_path = os.path.join(self.processed_data_dir, "wallet_summary.parquet")
                
                if (os.path.getmtime(cache_file) > os.path.getmtime(tx_file_path) and
                    os.path.getmtime(cache_file) > os.path.getmtime(wallet_file_path)):
                    logger.info(f"Loading transaction activity features from cache: {cache_file}")
                    with open(cache_file, 'rb') as f:
                        activity_features = pickle.load(f)
                    
                    # Merge with wallet_features
                    if not activity_features.empty:
                        new_features = [col for col in activity_features.columns 
                                       if col != 'wallet' and col not in wallet_features.columns]
                        if new_features:
                            wallet_features = wallet_features.merge(
                                activity_features[['wallet'] + new_features], 
                                on='wallet', 
                                how='left'
                            )
                    
                    logger.info(f"Using cached transaction activity features for {len(wallet_features)} wallets")
                    return wallet_features
            except Exception as e:
                logger.warning(f"Error loading from cache, will regenerate: {str(e)}")
        
        # Optimized batch calculation
        if optimize:
            # Pre-compute event type filters
            mint_mask = transaction_df['event_type'] == 'Mint'
            borrow_mask = transaction_df['event_type'] == 'Borrow'
            repay_mask = transaction_df['event_type'] == 'RepayBorrow'
            liquidation_mask = transaction_df['event_type'] == 'LiquidateBorrow'
            
            # Prepare features DataFrame
            features = []
            
            # Process in batches by wallet
            wallet_groups = list(transaction_df.groupby('account'))
            
            for wallet, wallet_data in tqdm(wallet_groups, desc="Calculating transaction features"):
                wallet_row = {'wallet': wallet}
                
                # Calculate supply and borrow volumes
                wallet_mint = wallet_data[wallet_data['event_type'] == 'Mint']
                wallet_borrow = wallet_data[wallet_data['event_type'] == 'Borrow']
                wallet_repay = wallet_data[wallet_data['event_type'] == 'RepayBorrow']
                
                # Calculate volumes efficiently
                if 'amount' in wallet_data.columns:
                    supply_volume = wallet_mint['amount'].sum() if not wallet_mint.empty else 0
                    borrow_volume = wallet_borrow['amount'].sum() if not wallet_borrow.empty else 0
                    repay_volume = wallet_repay['amount'].sum() if not wallet_repay.empty else 0
                else:
                    supply_volume = borrow_volume = repay_volume = 0
                
                # Calculate ratios
                wallet_row['borrow_to_supply_ratio_count'] = (
                    len(wallet_borrow) / max(1, len(wallet_mint))
                )
                
                wallet_row['borrow_to_supply_ratio_vol'] = (
                    borrow_volume / max(1, supply_volume)
                )
                
                wallet_row['repay_ratio'] = (
                    repay_volume / max(1, borrow_volume)
                )
                
                # Process liquidations
                wallet_liquidations = wallet_data[wallet_data['event_type'] == 'LiquidateBorrow']
                liquidation_as_borrower = 0
                liquidation_as_liquidator = 0
                
                if 'liquidator' in wallet_liquidations.columns and not wallet_liquidations.empty:
                    # Count liquidations where this wallet was the borrower
                    liquidation_as_borrower = wallet_liquidations[
                        wallet_liquidations['account'] == wallet
                    ].shape[0]
                    
                    # Count liquidations where this wallet was the liquidator
                    liquidation_as_liquidator = wallet_liquidations[
                        wallet_liquidations['liquidator'] == wallet
                    ].shape[0]
                
                wallet_row['liquidation_count_borrower'] = liquidation_as_borrower
                wallet_row['liquidation_count_liquidator'] = liquidation_as_liquidator
                
                features.append(wallet_row)
        else:
            # Original non-optimized implementation
            features = []
            
            for wallet, wallet_data in tqdm(transaction_df.groupby('account'), desc="Calculating transaction features"):
                wallet_row = {}
                wallet_row['wallet'] = wallet
                
                # Calculate supply and borrow volumes
                supply_volume = 0
                borrow_volume = 0
                repay_volume = 0
                
                # Filter by event type
                mint_events = wallet_data[wallet_data['event_type'] == 'Mint']
                borrow_events = wallet_data[wallet_data['event_type'] == 'Borrow']
                repay_events = wallet_data[wallet_data['event_type'] == 'RepayBorrow']
                
                # Calculate volumes
                if 'amount' in wallet_data.columns:
                    supply_volume = mint_events['amount'].sum() if not mint_events.empty else 0
                    borrow_volume = borrow_events['amount'].sum() if not borrow_events.empty else 0
                    repay_volume = repay_events['amount'].sum() if not repay_events.empty else 0
                
                # Calculate ratios
                wallet_row['borrow_to_supply_ratio_count'] = (
                    wallet_data[wallet_data['event_type'] == 'Borrow'].shape[0] / 
                    max(1, wallet_data[wallet_data['event_type'] == 'Mint'].shape[0])
                )
                
                wallet_row['borrow_to_supply_ratio_vol'] = (
                    borrow_volume / max(1, supply_volume)
                )
                
                wallet_row['repay_ratio'] = (
                    repay_volume / max(1, borrow_volume)
                )
                
                # Liquidations (as borrower vs. as liquidator)
                liquidation_events = wallet_data[wallet_data['event_type'] == 'LiquidateBorrow']
                liquidation_as_borrower = 0
                liquidation_as_liquidator = 0
                
                if 'liquidator' in liquidation_events.columns:
                    # Count liquidations where this wallet was the borrower
                    liquidation_as_borrower = liquidation_events[
                        liquidation_events['account'] == wallet
                    ].shape[0]
                    
                    # Count liquidations where this wallet was the liquidator
                    liquidation_as_liquidator = liquidation_events[
                        liquidation_events['liquidator'] == wallet
                    ].shape[0]
                
                wallet_row['liquidation_count_borrower'] = liquidation_as_borrower
                wallet_row['liquidation_count_liquidator'] = liquidation_as_liquidator
                
                features.append(wallet_row)
        
        # Create DataFrame from features
        activity_features = pd.DataFrame(features)
        
        # Save to cache if optimization is enabled
        if optimize:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(activity_features, f)
                logger.info(f"Cached transaction activity features to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache transaction activity features: {str(e)}")
        
        # Merge with wallet_features if there are new features
        if not activity_features.empty:
            # Keep only the new features (not already in wallet_features)
            new_features = [col for col in activity_features.columns if col != 'wallet' and col not in wallet_features.columns]
            if new_features:
                wallet_features = wallet_features.merge(
                    activity_features[['wallet'] + new_features], 
                    on='wallet', 
                    how='left'
                )
        
        logger.info(f"Generated transaction activity features for {len(wallet_features)} wallets")
        
        return wallet_features
    
    def generate_temporal_features(self, transaction_df, wallet_features, optimize=True):
        """
        Generate features related to temporal patterns.
        
        Args:
            transaction_df (pd.DataFrame): Transaction data
            wallet_features (pd.DataFrame): Wallet features
            optimize (bool): Whether to use optimized algorithms
            
        Returns:
            pd.DataFrame: DataFrame with temporal features added
        """
        logger.info("Generating temporal pattern features...")
        
        # Create a copy of the input DataFrame
        wallet_features = wallet_features.copy()
        
        # Check for cached features if optimization is enabled
        cache_file = os.path.join(config.CACHE_DIR, "temporal_features.pkl")
        if optimize and os.path.exists(cache_file):
            try:
                # Check if cache is newer than source data
                tx_file_path = os.path.join(self.processed_data_dir, "cleaned_transactions.parquet")
                
                if os.path.getmtime(cache_file) > os.path.getmtime(tx_file_path):
                    logger.info(f"Loading temporal features from cache: {cache_file}")
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    # Add days since last transaction (this needs to be recalculated every time)
                    current_time = datetime.now()
                    wallet_features['days_since_last_tx'] = wallet_features['last_tx_timestamp'].apply(
                        lambda x: calculate_time_diff_days(x, current_time)
                    )
                    
                    # Merge cached temporal features with wallet_features
                    new_features = [col for col in cached_data.columns 
                                   if col != 'wallet' and col not in wallet_features.columns]
                    if new_features:
                        wallet_features = wallet_features.merge(
                            cached_data[['wallet'] + new_features], 
                            on='wallet', 
                            how='left'
                        )
                    
                    logger.info(f"Using cached temporal features for {len(wallet_features)} wallets")
                    return wallet_features
            except Exception as e:
                logger.warning(f"Error loading from cache, will regenerate: {str(e)}")
        
        # Calculate days since last transaction (always calculated fresh)
        current_time = datetime.now()
        wallet_features['days_since_last_tx'] = wallet_features['last_tx_timestamp'].apply(
            lambda x: calculate_time_diff_days(x, current_time)
        )
        
        if optimize:
            # Optimized temporal feature calculation
            # Pre-sort transaction data by timestamp for each wallet
            features = []
            
            # Group by wallet and process each wallet's transactions
            wallet_groups = list(transaction_df.groupby('account'))
            
            for wallet, data in tqdm(wallet_groups, desc="Calculating temporal features"):
                wallet_row = {'wallet': wallet}
                
                # Sort transactions by timestamp (more efficient than in loop)
                sorted_tx = data.sort_values('timestamp')
                
                if len(sorted_tx) > 1:
                    # Vectorized time interval calculation
                    timestamps = sorted_tx['timestamp'].values
                    next_timestamps = np.append(timestamps[1:], [pd.NaT])
                    
                    # Convert to seconds, then days
                    time_intervals = []
                    for i in range(len(timestamps) - 1):
                        interval = (timestamps[i+1] - timestamps[i]).total_seconds() / (24 * 3600)
                        time_intervals.append(interval)
                    
                    if time_intervals:
                        time_intervals = np.array(time_intervals)
                        wallet_row['avg_time_between_tx_days'] = np.mean(time_intervals)
                        wallet_row['max_time_between_tx_days'] = np.max(time_intervals)
                        wallet_row['min_time_between_tx_days'] = np.min(time_intervals)
                        wallet_row['activity_consistency_stddev'] = np.std(time_intervals)
                        
                        # Coefficient of variation (CV)
                        if np.mean(time_intervals) > 0:
                            wallet_row['activity_consistency_cv'] = np.std(time_intervals) / np.mean(time_intervals)
                
                features.append(wallet_row)
        else:
            # Original implementation
            features = []
            
            for wallet, wallet_data in tqdm(transaction_df.groupby('account'), desc="Calculating temporal features"):
                wallet_row = {}
                wallet_row['wallet'] = wallet
                
                # Sort transactions by timestamp
                sorted_tx = wallet_data.sort_values('timestamp')
                
                if len(sorted_tx) > 1:
                    # Calculate time intervals between consecutive transactions
                    sorted_tx['next_timestamp'] = sorted_tx['timestamp'].shift(-1)
                    sorted_tx['time_interval'] = sorted_tx.apply(
                        lambda row: calculate_time_diff_days(row['timestamp'], row['next_timestamp'])
                        if pd.notna(row['next_timestamp']) else np.nan,
                        axis=1
                    )
                    
                    # Calculate statistics of time intervals
                    time_intervals = sorted_tx['time_interval'].dropna()
                    if len(time_intervals) > 0:
                        wallet_row['avg_time_between_tx_days'] = time_intervals.mean()
                        wallet_row['max_time_between_tx_days'] = time_intervals.max()
                        wallet_row['min_time_between_tx_days'] = time_intervals.min()
                        wallet_row['activity_consistency_stddev'] = time_intervals.std()
                
                features.append(wallet_row)
        
        # Create DataFrame from features
        temporal_features = pd.DataFrame(features)
        
        # Save to cache if optimization is enabled
        if optimize and not temporal_features.empty:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(temporal_features, f)
                logger.info(f"Cached temporal features to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache temporal features: {str(e)}")
        
        # Merge with wallet_features if there are new features
        if not temporal_features.empty:
            # Keep only the new features (not already in wallet_features)
            new_features = [col for col in temporal_features.columns if col != 'wallet' and col not in wallet_features.columns]
            if new_features:
                wallet_features = wallet_features.merge(
                    temporal_features[['wallet'] + new_features], 
                    on='wallet', 
                    how='left'
                )
        
        logger.info(f"Generated temporal features for {len(wallet_features)} wallets")
        
        return wallet_features
    
    def handle_missing_values(self, features_df):
        """
        Handle missing values in features.
        
        Args:
            features_df (pd.DataFrame): Features DataFrame with potential missing values
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        logger.info("Handling missing values in features...")
        
        # Create a copy of the input DataFrame
        cleaned_df = features_df.copy()
        
        # Get numerical columns (excluding wallet and other non-numeric columns)
        numeric_cols = cleaned_df.select_dtypes(include=['int', 'float']).columns.tolist()
        
        # More efficient missing value handling
        # Vectorized operations instead of looping through columns
        
        # For ratio features (fill with 0)
        ratio_cols = [col for col in numeric_cols if 'ratio' in col or 'pct' in col]
        if ratio_cols:
            cleaned_df[ratio_cols] = cleaned_df[ratio_cols].fillna(0)
        
        # For count features (fill with 0)
        count_cols = [col for col in numeric_cols if 'count' in col]
        if count_cols:
            cleaned_df[count_cols] = cleaned_df[count_cols].fillna(0)
        
        # For time-related features (fill with median)
        time_cols = [col for col in numeric_cols if 'time' in col or 'days' in col]
        for col in time_cols:
            if cleaned_df[col].isna().any():
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        
        # For remaining features (fill with median)
        remaining_cols = [col for col in numeric_cols 
                         if col not in ratio_cols and col not in count_cols and col not in time_cols]
        for col in remaining_cols:
            if cleaned_df[col].isna().any():
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        
        # Final check for any remaining NaN values
        for col in numeric_cols:
            if cleaned_df[col].isna().any():
                logger.warning(f"Some NaN values remain in {col}, filling with 0")
                cleaned_df[col] = cleaned_df[col].fillna(0)
        
        logger.info("Missing values handling complete")
        
        return cleaned_df

    def generate_all_features(self, optimize=True):
        """
        Generate all features for wallet scoring.
        
        Args:
            optimize (bool): Whether to use optimized algorithms
            
        Returns:
            pd.DataFrame: DataFrame with all features
        """
        logger.info(f"Generating all features for wallets with optimization={'enabled' if optimize else 'disabled'}...")
        
        # Check for cached full feature set if optimization is enabled
        if optimize:
            features_cache = os.path.join(config.CACHE_DIR, "all_features.pkl")
            if os.path.exists(features_cache):
                try:
                    # Check if source data has been modified
                    tx_file_path = os.path.join(self.processed_data_dir, "cleaned_transactions.parquet")
                    wallet_file_path = os.path.join(self.processed_data_dir, "wallet_summary.parquet")
                    
                    features_mtime = os.path.getmtime(features_cache)
                    if (features_mtime > os.path.getmtime(tx_file_path) and 
                        features_mtime > os.path.getmtime(wallet_file_path)):
                        logger.info(f"Loading all features from cache: {features_cache}")
                        with open(features_cache, 'rb') as f:
                            wallet_features = pickle.load(f)
                        logger.info(f"Loaded cached features for {len(wallet_features)} wallets")
                        return wallet_features
                except Exception as e:
                    logger.warning(f"Error loading from all-features cache, will regenerate: {str(e)}")
        
        # Load data
        transaction_df, wallet_df = self.load_data()
        
        # Generate features
        wallet_features = self.generate_transaction_activity_features(transaction_df, wallet_df, optimize=optimize)
        wallet_features = self.generate_temporal_features(transaction_df, wallet_features, optimize=optimize)
        
        # Handle missing values
        wallet_features = self.handle_missing_values(wallet_features)
        
        # Save generated features
        features_path = os.path.join(self.processed_data_dir, "wallet_features.parquet")
        wallet_features.to_parquet(features_path, index=False)
        logger.info(f"Saved generated features for {len(wallet_features)} wallets to {features_path}")
        
        # Cache all features if optimization is enabled
        if optimize:
            try:
                with open(os.path.join(config.CACHE_DIR, "all_features.pkl"), 'wb') as f:
                    pickle.dump(wallet_features, f)
                logger.info(f"Cached all features to {os.path.join(config.CACHE_DIR, 'all_features.pkl')}")
            except Exception as e:
                logger.warning(f"Failed to cache all features: {str(e)}")
        
        return wallet_features
        
def main():
    """
    Run feature engineering as a standalone script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate features for wallet scoring")
    parser.add_argument('--no-optimize', action='store_true', help='Disable algorithmic optimizations')
    args = parser.parse_args()
    
    feature_engineer = FeatureEngineer()
    feature_engineer.generate_all_features(optimize=not args.no_optimize)

if __name__ == "__main__":
    main() 