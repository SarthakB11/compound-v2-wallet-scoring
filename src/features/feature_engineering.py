"""
Feature engineering for wallet behavior analysis.
"""
import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

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
        
        # Ensure directory exists
        os.makedirs(self.processed_data_dir, exist_ok=True)
    
    def load_data(self):
        """
        Load the processed transaction and wallet data.
        
        Returns:
            tuple: (transaction_df, wallet_df)
        """
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
        
        return transaction_df, wallet_df
    
    def generate_transaction_activity_features(self, transaction_df, wallet_df):
        """
        Generate features related to transaction activity.
        
        Args:
            transaction_df (pd.DataFrame): Transaction data
            wallet_df (pd.DataFrame): Wallet summary data
            
        Returns:
            pd.DataFrame: DataFrame with transaction activity features
        """
        logger.info("Generating transaction activity features...")
        
        # Create a copy of the wallet DataFrame
        wallet_features = wallet_df.copy()
        
        # Some features are already calculated during aggregation
        # tx_count, mint_count, redeem_count, borrow_count, repay_count, liquidation_count
        
        # Calculate additional transaction activity features
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
    
    def generate_temporal_features(self, transaction_df, wallet_features):
        """
        Generate features related to temporal patterns.
        
        Args:
            transaction_df (pd.DataFrame): Transaction data
            wallet_features (pd.DataFrame): Wallet features
            
        Returns:
            pd.DataFrame: DataFrame with temporal features added
        """
        logger.info("Generating temporal pattern features...")
        
        # Create a copy of the input DataFrame
        wallet_features = wallet_features.copy()
        
        # Calculate days since last transaction
        current_time = datetime.now()
        wallet_features['days_since_last_tx'] = wallet_features['last_tx_timestamp'].apply(
            lambda x: calculate_time_diff_days(x, current_time)
        )
        
        # Calculate activity consistency
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
                    wallet_row['median_time_between_tx_days'] = time_intervals.median()
                    wallet_row['max_time_between_tx_days'] = time_intervals.max()
                    wallet_row['activity_consistency_stddev'] = time_intervals.std()
                
                # Identify periods of inactivity (gaps > 30 days)
                long_gaps = time_intervals[time_intervals > 30].count()
                wallet_row['long_inactivity_periods'] = long_gaps
                
                # Calculate activity burstiness
                if len(time_intervals) > 1:
                    mean_interval = time_intervals.mean()
                    std_interval = time_intervals.std()
                    if mean_interval > 0:
                        wallet_row['activity_burstiness'] = std_interval / mean_interval
                
                # Calculate session-based metrics
                # Group transactions occurring within 1 hour as a session
                SESSION_THRESHOLD = 1  # hours
                sessions = []
                current_session = [sorted_tx.iloc[0]]
                
                for i in range(1, len(sorted_tx)):
                    time_diff = (sorted_tx.iloc[i]['timestamp'] - sorted_tx.iloc[i-1]['timestamp']).total_seconds() / 3600
                    if time_diff <= SESSION_THRESHOLD:
                        current_session.append(sorted_tx.iloc[i])
                    else:
                        sessions.append(current_session)
                        current_session = [sorted_tx.iloc[i]]
                
                if current_session:
                    sessions.append(current_session)
                
                wallet_row['session_count'] = len(sessions)
                wallet_row['avg_actions_per_session'] = np.mean([len(session) for session in sessions])
            
            features.append(wallet_row)
        
        # Create DataFrame from features
        temporal_features = pd.DataFrame(features)
        
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
        
        logger.info(f"Generated temporal pattern features for {len(wallet_features)} wallets")
        
        return wallet_features
    
    def generate_financial_health_features(self, transaction_df, wallet_features):
        """
        Generate features related to financial health and risk.
        
        Args:
            transaction_df (pd.DataFrame): Transaction data
            wallet_features (pd.DataFrame): Wallet features
            
        Returns:
            pd.DataFrame: DataFrame with financial health features added
        """
        logger.info("Generating financial health and risk features...")
        
        # Create a copy of the input DataFrame
        wallet_features = wallet_features.copy()
        
        # Track historical LTV (Loan-to-Value) for each wallet
        features = []
        
        for wallet, wallet_data in tqdm(transaction_df.groupby('account'), desc="Calculating financial features"):
            wallet_row = {}
            wallet_row['wallet'] = wallet
            
            # Sort transactions by timestamp
            sorted_tx = wallet_data.sort_values('timestamp')
            
            # Initialize wallet state
            collateral_balance = 0
            borrow_balance = 0
            ltv_history = []
            
            # Process transactions sequentially to track wallet state
            for _, tx in sorted_tx.iterrows():
                event_type = tx['event_type']
                amount = tx.get('amount', 0)
                
                # Update wallet state based on event type
                if event_type == 'Mint':
                    collateral_balance += amount
                elif event_type == 'Redeem':
                    collateral_balance = max(0, collateral_balance - amount)
                elif event_type == 'Borrow':
                    borrow_balance += amount
                elif event_type == 'RepayBorrow':
                    borrow_balance = max(0, borrow_balance - amount)
                elif event_type == 'LiquidateBorrow':
                    # If this wallet was liquidated
                    if 'liquidator' in tx and tx.get('account') == wallet:
                        # Rough estimation: liquidation typically clears some debt and takes some collateral
                        # In Compound V2, liquidation factor is typically around 50%
                        borrow_balance = max(0, borrow_balance - amount)
                        collateral_amount = tx.get('collateral_amount', amount * 1.1)  # Estimate if not available
                        collateral_balance = max(0, collateral_balance - collateral_amount)
                
                # Calculate LTV (if collateral is present)
                if collateral_balance > 0:
                    ltv = min(1.0, borrow_balance / collateral_balance)  # Cap at 1.0 for realism
                    ltv_history.append((tx['timestamp'], ltv))
            
            # Calculate LTV-based features
            if ltv_history:
                ltv_values = [ltv for _, ltv in ltv_history]
                wallet_row['avg_ltv_alltime'] = np.mean(ltv_values)
                wallet_row['max_ltv_alltime'] = np.max(ltv_values)
                wallet_row['min_ltv_alltime'] = np.min(ltv_values)
                wallet_row['std_ltv_alltime'] = np.std(ltv_values)
                
                # Calculate time spent near liquidation threshold (LTV > 0.75)
                high_ltv_count = sum(1 for ltv in ltv_values if ltv > 0.75)
                wallet_row['time_near_liquidation_pct'] = high_ltv_count / len(ltv_values) if ltv_values else 0
                
                # Calculate recent LTV (last 90 days)
                if 'timestamp' in sorted_tx.columns:
                    last_timestamp = sorted_tx['timestamp'].max()
                    cutoff_90d = last_timestamp - timedelta(days=90)
                    recent_ltv = [ltv for time, ltv in ltv_history if time >= cutoff_90d]
                    if recent_ltv:
                        wallet_row['avg_ltv_90d'] = np.mean(recent_ltv)
                        wallet_row['max_ltv_90d'] = np.max(recent_ltv)
            
            features.append(wallet_row)
        
        # Create DataFrame from features
        financial_features = pd.DataFrame(features)
        
        # Merge with wallet_features if there are new features
        if not financial_features.empty:
            # Keep only the new features (not already in wallet_features)
            new_features = [col for col in financial_features.columns if col != 'wallet' and col not in wallet_features.columns]
            if new_features:
                wallet_features = wallet_features.merge(
                    financial_features[['wallet'] + new_features], 
                    on='wallet', 
                    how='left'
                )
        
        logger.info(f"Generated financial health features for {len(wallet_features)} wallets")
        
        return wallet_features
    
    def generate_protocol_interaction_features(self, transaction_df, wallet_features):
        """
        Generate features related to protocol interaction patterns.
        
        Args:
            transaction_df (pd.DataFrame): Transaction data
            wallet_features (pd.DataFrame): Wallet features
            
        Returns:
            pd.DataFrame: DataFrame with protocol interaction features added
        """
        logger.info("Generating protocol interaction features...")
        
        # Create a copy of the input DataFrame
        wallet_features = wallet_features.copy()
        
        # Calculate protocol interaction features
        features = []
        
        for wallet, wallet_data in tqdm(transaction_df.groupby('account'), desc="Calculating interaction features"):
            wallet_row = {}
            wallet_row['wallet'] = wallet
            
            # Market diversity is already calculated in the wallet summary (market_count)
            
            # Calculate gas price metrics if available
            if 'gas_price' in wallet_data.columns:
                wallet_row['avg_gas_price'] = wallet_data['gas_price'].mean()
                wallet_row['median_gas_price'] = wallet_data['gas_price'].median()
                wallet_row['max_gas_price'] = wallet_data['gas_price'].max()
            
            # Calculate complex action frequency
            # Complex actions are defined as multiple different actions (event types) 
            # within a short time window (e.g., 1 hour)
            if 'timestamp' in wallet_data.columns:
                # Sort by timestamp
                sorted_tx = wallet_data.sort_values('timestamp')
                
                # Group transactions into 1-hour windows
                sorted_tx['hour'] = sorted_tx['timestamp'].dt.floor('H')
                
                # Count unique event types per hour
                action_complexity = sorted_tx.groupby('hour')['event_type'].nunique()
                
                # Calculate metrics
                complex_actions = action_complexity[action_complexity > 1]
                wallet_row['complex_action_count'] = len(complex_actions)
                wallet_row['complex_action_freq'] = len(complex_actions) / len(action_complexity) if len(action_complexity) > 0 else 0
                wallet_row['max_action_complexity'] = action_complexity.max() if len(action_complexity) > 0 else 0
            
            # Approximate COMP farming behavior
            # In Compound V2, COMP rewards were based on borrowing and supplying
            # A simple proxy: ratio of borrow events that are followed by supply events within a short time
            if 'timestamp' in wallet_data.columns and len(wallet_data) > 1:
                borrow_events = wallet_data[wallet_data['event_type'] == 'Borrow']
                farming_count = 0
                
                for _, borrow_tx in borrow_events.iterrows():
                    borrow_time = borrow_tx['timestamp']
                    # Check if there's a supply event within 30 minutes
                    supply_after_borrow = wallet_data[
                        (wallet_data['event_type'] == 'Mint') & 
                        (wallet_data['timestamp'] > borrow_time) & 
                        (wallet_data['timestamp'] <= borrow_time + timedelta(minutes=30))
                    ]
                    
                    if len(supply_after_borrow) > 0:
                        farming_count += 1
                
                wallet_row['comp_farming_events'] = farming_count
                wallet_row['comp_earned_estimate_ratio'] = farming_count / len(borrow_events) if len(borrow_events) > 0 else 0
            
            features.append(wallet_row)
        
        # Create DataFrame from features
        protocol_features = pd.DataFrame(features)
        
        # Merge with wallet_features if there are new features
        if not protocol_features.empty:
            # Keep only the new features (not already in wallet_features)
            new_features = [col for col in protocol_features.columns if col != 'wallet' and col not in wallet_features.columns]
            if new_features:
                wallet_features = wallet_features.merge(
                    protocol_features[['wallet'] + new_features], 
                    on='wallet', 
                    how='left'
                )
        
        logger.info(f"Generated protocol interaction features for {len(wallet_features)} wallets")
        
        return wallet_features
    
    def handle_missing_values(self, features_df):
        """
        Handle missing values in the features DataFrame.
        
        Args:
            features_df (pd.DataFrame): Features DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        logger.info("Handling missing values in features...")
        
        # Create a copy of the input DataFrame
        features_df = features_df.copy()
        
        # Fill missing values with appropriate defaults
        for column in features_df.columns:
            if column == 'wallet':
                continue
            
            missing_count = features_df[column].isna().sum()
            if missing_count > 0:
                logger.info(f"Filling {missing_count} missing values in column '{column}'")
                
                # Different strategies based on feature type
                if 'ratio' in column or 'pct' in column:
                    # For ratio/percentage features, use 0
                    features_df[column] = features_df[column].fillna(0)
                elif 'count' in column:
                    # For count features, use 0
                    features_df[column] = features_df[column].fillna(0)
                elif column in ['avg_ltv_90d', 'max_ltv_90d']:
                    # For recent LTV features, use all-time values if available
                    # or 0 if not available
                    all_time_col = column.replace('90d', 'alltime')
                    if all_time_col in features_df.columns:
                        features_df[column] = features_df[column].fillna(features_df[all_time_col]).fillna(0)
                    else:
                        features_df[column] = features_df[column].fillna(0)
                else:
                    # For other numeric features, use median
                    features_df[column] = features_df[column].fillna(features_df[column].median())
        
        return features_df
    
    def generate_all_features(self):
        """
        Generate all features for wallet scoring.
        
        Returns:
            pd.DataFrame: DataFrame with all features
        """
        # Load data
        transaction_df, wallet_df = self.load_data()
        
        # Generate transaction activity features
        features_df = self.generate_transaction_activity_features(transaction_df, wallet_df)
        
        # Generate temporal pattern features
        features_df = self.generate_temporal_features(transaction_df, features_df)
        
        # Generate financial health features
        features_df = self.generate_financial_health_features(transaction_df, features_df)
        
        # Generate protocol interaction features
        features_df = self.generate_protocol_interaction_features(transaction_df, features_df)
        
        # Handle missing values
        features_df = self.handle_missing_values(features_df)
        
        # Save features
        features_file_path = os.path.join(self.processed_data_dir, "wallet_features.parquet")
        features_df.to_parquet(features_file_path, index=False)
        logger.info(f"Saved features for {len(features_df)} wallets to {features_file_path}")
        
        return features_df

def main():
    """
    Main function to generate features.
    """
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.generate_all_features()
    
    print(f"Generated features for {len(features_df)} wallets")
    print(f"Feature columns: {features_df.columns.tolist()}")
    print("\nSample features:")
    print(features_df.head())
    
if __name__ == "__main__":
    main() 