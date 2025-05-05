"""
Preprocessor for cleaning and preparing Compound V2 transaction data.
"""
import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.utils.helpers import standardize_address, save_dataframe
import src.config as config

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Class for preprocessing Compound V2 transaction data.
    """
    
    def __init__(self, processed_data_dir=None):
        """
        Initialize the DataPreprocessor.
        
        Args:
            processed_data_dir (str, optional): Directory for processed data
        """
        self.processed_data_dir = processed_data_dir or config.PROCESSED_DATA_DIR
        
        # Ensure directory exists
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Event types we're interested in
        self.event_types = ['Mint', 'Redeem', 'Borrow', 'RepayBorrow', 'LiquidateBorrow']
        
    def load_data(self, file_path=None):
        """
        Load the transaction data.
        
        Args:
            file_path (str, optional): Path to the data file
            
        Returns:
            pd.DataFrame: Loaded transaction data
        """
        if file_path is None:
            file_path = os.path.join(self.processed_data_dir, "combined_transactions.parquet")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load data
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(df)} transactions from {file_path}")
        
        return df
    
    def clean_data(self, df):
        """
        Clean the transaction data.
        
        Args:
            df (pd.DataFrame): Raw transaction data
            
        Returns:
            pd.DataFrame: Cleaned transaction data
        """
        logger.info("Cleaning transaction data...")
        
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Check for and handle missing values in critical columns
        critical_columns = ['timestamp', 'account', 'event_type']
        for col in critical_columns:
            if col in cleaned_df.columns:
                missing_count = cleaned_df[col].isna().sum()
                if missing_count > 0:
                    logger.warning(f"Found {missing_count} missing values in column '{col}'")
                    
                    # Drop rows with missing critical values
                    cleaned_df = cleaned_df.dropna(subset=[col])
        
        # Standardize event types
        if 'event_type' in cleaned_df.columns:
            # Ensure consistent capitalization
            cleaned_df['event_type'] = cleaned_df['event_type'].str.strip().str.title()
            
            # Map similar event types to our standard ones
            event_type_mapping = {
                'Mint': 'Mint',
                'Supply': 'Mint',
                'Deposit': 'Mint',
                
                'Redeem': 'Redeem',
                'Withdraw': 'Redeem',
                
                'Borrow': 'Borrow',
                
                'Repayborrow': 'RepayBorrow',
                'Repay': 'RepayBorrow',
                'Repayment': 'RepayBorrow',
                
                'Liquidateborrow': 'LiquidateBorrow',
                'Liquidate': 'LiquidateBorrow',
            }
            
            # Apply mapping
            cleaned_df['event_type'] = cleaned_df['event_type'].map(
                lambda x: event_type_mapping.get(x, x) if isinstance(x, str) else x
            )
            
            # Filter to only include our event types of interest
            valid_events = cleaned_df['event_type'].isin(self.event_types)
            cleaned_df = cleaned_df[valid_events]
            
            logger.info(f"Kept {len(cleaned_df)} transactions with valid event types")
        
        # Handle amount fields
        amount_cols = [col for col in cleaned_df.columns if 'amount' in col.lower()]
        for col in amount_cols:
            # Try to convert to numeric
            if cleaned_df[col].dtype == 'object':
                try:
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                except:
                    logger.warning(f"Could not convert column '{col}' to numeric")
        
        # Ensure timestamp is datetime type
        if 'timestamp' in cleaned_df.columns and cleaned_df['timestamp'].dtype != 'datetime64[ns]':
            try:
                cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'])
            except:
                logger.warning("Could not convert timestamp to datetime")
        
        # Sort by timestamp
        if 'timestamp' in cleaned_df.columns:
            cleaned_df = cleaned_df.sort_values('timestamp')
            
        # Reset index
        cleaned_df = cleaned_df.reset_index(drop=True)
        
        logger.info(f"Cleaned data contains {len(cleaned_df)} transactions")
        
        return cleaned_df
    
    def aggregate_by_wallet(self, df):
        """
        Aggregate transactions by wallet.
        
        Args:
            df (pd.DataFrame): Cleaned transaction data
            
        Returns:
            pd.DataFrame: Wallet-aggregated data
        """
        logger.info("Aggregating transactions by wallet...")
        
        # Group by wallet address (account)
        wallet_groups = df.groupby('account')
        
        # Create a list to store wallet summaries
        wallet_summaries = []
        
        # Process each wallet
        for wallet, wallet_df in tqdm(wallet_groups, desc="Aggregating wallets"):
            # Basic stats
            first_tx = wallet_df['timestamp'].min()
            last_tx = wallet_df['timestamp'].max()
            tx_count = len(wallet_df)
            
            # Count event types
            event_counts = wallet_df['event_type'].value_counts().to_dict()
            mint_count = event_counts.get('Mint', 0)
            redeem_count = event_counts.get('Redeem', 0)
            borrow_count = event_counts.get('Borrow', 0)
            repay_count = event_counts.get('RepayBorrow', 0)
            liquidation_count = event_counts.get('LiquidateBorrow', 0)
            
            # Calculate wallet age in days
            wallet_age_days = (last_tx - first_tx).total_seconds() / (60 * 60 * 24)
            
            # Get unique tokens/markets
            unique_tokens = set()
            if 'token' in wallet_df.columns:
                unique_tokens.update(wallet_df['token'].dropna().unique())
            if 'ctoken' in wallet_df.columns:
                unique_tokens.update(wallet_df['ctoken'].dropna().unique())
            market_count = len(unique_tokens)
            
            # Create wallet summary
            wallet_summary = {
                'wallet': wallet,
                'first_tx_timestamp': first_tx,
                'last_tx_timestamp': last_tx,
                'tx_count': tx_count,
                'wallet_age_days': wallet_age_days,
                'mint_count': mint_count,
                'redeem_count': redeem_count,
                'borrow_count': borrow_count,
                'repay_count': repay_count,
                'liquidation_count': liquidation_count,
                'market_count': market_count,
            }
            
            wallet_summaries.append(wallet_summary)
        
        # Create DataFrame from summaries
        wallet_df = pd.DataFrame(wallet_summaries)
        
        logger.info(f"Created aggregated data for {len(wallet_df)} wallets")
        
        return wallet_df
    
    def save_processed_data(self, transaction_df, wallet_df):
        """
        Save the processed data.
        
        Args:
            transaction_df (pd.DataFrame): Cleaned transaction data
            wallet_df (pd.DataFrame): Wallet-aggregated data
        """
        # Save transaction data
        tx_file_path = os.path.join(self.processed_data_dir, "cleaned_transactions.parquet")
        transaction_df.to_parquet(tx_file_path, index=False)
        logger.info(f"Saved cleaned transaction data to {tx_file_path}")
        
        # Save wallet data
        wallet_file_path = os.path.join(self.processed_data_dir, "wallet_summary.parquet")
        wallet_df.to_parquet(wallet_file_path, index=False)
        logger.info(f"Saved wallet summary data to {wallet_file_path}")
    
    def process(self, input_file=None):
        """
        Run the full preprocessing pipeline.
        
        Args:
            input_file (str, optional): Path to the input file
            
        Returns:
            tuple: (transaction_df, wallet_df)
        """
        # Load data
        transaction_df = self.load_data(input_file)
        
        # Clean data
        cleaned_df = self.clean_data(transaction_df)
        
        # Aggregate by wallet
        wallet_df = self.aggregate_by_wallet(cleaned_df)
        
        # Save processed data
        self.save_processed_data(cleaned_df, wallet_df)
        
        return cleaned_df, wallet_df

def main():
    """
    Main function to preprocess data.
    """
    preprocessor = DataPreprocessor()
    transaction_df, wallet_df = preprocessor.process()
    
    print(f"Processed {len(transaction_df)} transactions for {len(wallet_df)} wallets")
    print("\nTransaction data sample:")
    print(transaction_df.head())
    print("\nWallet summary sample:")
    print(wallet_df.head())
    
if __name__ == "__main__":
    main() 