"""
Data loader for Compound V2 transaction data.
"""
import os
import json
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.utils.helpers import get_largest_files, standardize_address
import src.config as config

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Class for loading and parsing Compound V2 transaction data.
    """
    
    def __init__(self, raw_data_dir=None, processed_data_dir=None, num_files=None):
        """
        Initialize the DataLoader.
        
        Args:
            raw_data_dir (str, optional): Directory containing raw data files
            processed_data_dir (str, optional): Directory to save processed data
            num_files (int, optional): Number of largest files to process
        """
        self.raw_data_dir = raw_data_dir or config.RAW_DATA_DIR
        self.processed_data_dir = processed_data_dir or config.PROCESSED_DATA_DIR
        self.num_files = num_files or config.NUM_FILES_TO_PROCESS
        
        # Ensure directories exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Compound V2 event types we're interested in
        self.event_types = ['Mint', 'Redeem', 'Borrow', 'RepayBorrow', 'LiquidateBorrow']
        
    def get_input_files(self):
        """
        Get the list of input files to process.
        
        Returns:
            list: List of file paths
        """
        if not os.path.exists(self.raw_data_dir) or not os.listdir(self.raw_data_dir):
            raise FileNotFoundError(f"No files found in {self.raw_data_dir}. Please download the dataset first.")
        
        # Get the largest files
        files = get_largest_files(self.raw_data_dir, self.num_files)
        
        if not files:
            raise FileNotFoundError(f"No files found in {self.raw_data_dir}")
        
        logger.info(f"Selected {len(files)} files for processing:")
        for f in files:
            logger.info(f"  - {os.path.basename(f)} ({os.path.getsize(f) / (1024 * 1024):.2f} MB)")
            
        return files
    
    def infer_file_format(self, file_path):
        """
        Infer the format of a file based on its extension and content.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            str: File format ('csv', 'json', or 'unknown')
        """
        # Check extension first
        if file_path.endswith('.csv'):
            return 'csv'
        elif file_path.endswith('.json'):
            return 'json'
        
        # If extension doesn't reveal format, check content
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                first_line = f.readline().strip()
                # Check if it looks like JSON
                if first_line.startswith('{') or first_line.startswith('['):
                    return 'json'
                # Check if it looks like CSV
                elif ',' in first_line:
                    return 'csv'
            except:
                pass
        
        return 'unknown'
    
    def parse_file(self, file_path):
        """
        Parse a Compound V2 transaction file.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            pd.DataFrame: Parsed transactions
        """
        file_format = self.infer_file_format(file_path)
        
        if file_format == 'csv':
            return self._parse_csv(file_path)
        elif file_format == 'json':
            return self._parse_json(file_path)
        else:
            raise ValueError(f"Unsupported file format for {file_path}")
    
    def _parse_csv(self, file_path):
        """
        Parse a CSV file containing Compound V2 transactions.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Parsed transactions
        """
        try:
            # Try to read with standard parameters
            df = pd.read_csv(file_path)
            
            # Check if this looks like a valid transaction log
            required_columns = self._get_required_columns()
            
            # Check if we have enough required columns
            found_columns = [col for col in required_columns if col in df.columns]
            if len(found_columns) < len(required_columns) / 2:
                logger.warning(f"File {file_path} is missing many required columns. Trying alternative parsing.")
                
                # Try with different parameters
                df = pd.read_csv(file_path, sep=None, engine='python')
            
            # Standardize column names (lowercase)
            df.columns = [col.lower() for col in df.columns]
            
            return self._standardize_dataframe(df)
            
        except Exception as e:
            logger.error(f"Error parsing CSV file {file_path}: {str(e)}")
            raise
    
    def _parse_json(self, file_path):
        """
        Parse a JSON file containing Compound V2 transactions.
        
        Args:
            file_path (str): Path to the JSON file
            
        Returns:
            pd.DataFrame: Parsed transactions
        """
        try:
            # Read the JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle different JSON structures
            if isinstance(data, list):
                # List of transaction objects
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Real Compound V2 data format has keys like 'deposits', 'borrows', etc.
                # Check if it's a dictionary with transaction arrays
                compound_v2_event_types = {
                    'deposits': 'Mint',
                    'withdraws': 'Redeem',
                    'borrows': 'Borrow',
                    'repays': 'RepayBorrow',
                    'liquidations': 'LiquidateBorrow'
                }
                
                if any(key in data for key in compound_v2_event_types.keys()):
                    # Combine different event types into one DataFrame
                    dfs = []
                    for data_key, event_type in compound_v2_event_types.items():
                        if data_key in data and isinstance(data[data_key], list):
                            event_data = data[data_key]
                            # Check if there's any data
                            if event_data:
                                # Transform nested structures
                                transformed_data = []
                                for item in event_data:
                                    # Create a flattened version with standardized fields
                                    transformed_item = {}
                                    
                                    # Extract account ID
                                    if 'account' in item and isinstance(item['account'], dict) and 'id' in item['account']:
                                        transformed_item['account'] = item['account']['id']
                                    else:
                                        transformed_item['account'] = None
                                    
                                    # Extract asset/token info
                                    if 'asset' in item and isinstance(item['asset'], dict):
                                        if 'id' in item['asset']:
                                            transformed_item['token'] = item['asset']['id']
                                        if 'symbol' in item['asset']:
                                            transformed_item['token_symbol'] = item['asset']['symbol']
                                    
                                    # Copy basic fields
                                    for field in ['amount', 'amountUSD', 'hash', 'timestamp']:
                                        if field in item:
                                            if field == 'hash':
                                                transformed_item['transaction_hash'] = item[field]
                                            else:
                                                transformed_item[field] = item[field]
                                    
                                    # Add event type
                                    transformed_item['event_type'] = event_type
                                    
                                    transformed_data.append(transformed_item)
                                
                                if transformed_data:
                                    event_df = pd.DataFrame(transformed_data)
                                    dfs.append(event_df)
                    
                    if dfs:
                        df = pd.concat(dfs, ignore_index=True)
                    else:
                        # If no Compound V2 data found, try the standard approach
                        if any(key in data for key in self.event_types):
                            # Combine different event types into one DataFrame
                            dfs = []
                            for event_type in self.event_types:
                                if event_type in data:
                                    event_df = pd.DataFrame(data[event_type])
                                    event_df['event_type'] = event_type
                                    dfs.append(event_df)
                            
                            if dfs:
                                df = pd.concat(dfs, ignore_index=True)
                            else:
                                raise ValueError(f"No recognized event types found in {file_path}")
                        else:
                            # Single transaction object or unknown structure
                            df = pd.DataFrame([data])
                else:
                    # Handle original format
                    if any(key in data for key in self.event_types):
                        # Combine different event types into one DataFrame
                        dfs = []
                        for event_type in self.event_types:
                            if event_type in data:
                                event_df = pd.DataFrame(data[event_type])
                                event_df['event_type'] = event_type
                                dfs.append(event_df)
                        
                        if dfs:
                            df = pd.concat(dfs, ignore_index=True)
                        else:
                            raise ValueError(f"No recognized event types found in {file_path}")
                    else:
                        # Single transaction object or unknown structure
                        df = pd.DataFrame([data])
            else:
                raise ValueError(f"Unexpected JSON structure in {file_path}")
            
            # Standardize column names (lowercase)
            df.columns = [col.lower() for col in df.columns]
            
            return self._standardize_dataframe(df)
            
        except Exception as e:
            logger.error(f"Error parsing JSON file {file_path}: {str(e)}")
            raise
    
    def _get_required_columns(self):
        """
        Get the list of required columns for transaction data.
        
        Returns:
            list: List of required column names
        """
        # These are typical fields we expect in Compound V2 transaction data
        # based on the Compound V2 subgraph schema and event logs
        return [
            'event_type',           # Type of event (Mint, Redeem, Borrow, etc.)
            'timestamp',            # Transaction timestamp
            'block_number',         # Block number
            'transaction_hash',     # Transaction hash
            'account',              # User wallet address
            'amount',               # Amount of tokens
            'token',                # Token address or symbol
        ]
    
    def _standardize_dataframe(self, df):
        """
        Standardize a DataFrame containing transaction data.
        
        Args:
            df (pd.DataFrame): Raw transaction DataFrame
            
        Returns:
            pd.DataFrame: Standardized DataFrame
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Map column names to standard format
        column_mapping = {
            # Timestamps
            'timestamp': 'timestamp',
            'block_timestamp': 'timestamp',
            'time': 'timestamp',
            'date': 'timestamp',
            
            # Block info
            'block_number': 'block_number',
            'block': 'block_number',
            'blocknumber': 'block_number',
            
            # Transaction info
            'transaction_hash': 'transaction_hash',
            'tx_hash': 'transaction_hash',
            'txhash': 'transaction_hash',
            'hash': 'transaction_hash',
            
            # Event type
            'event_type': 'event_type',
            'event': 'event_type',
            'type': 'event_type',
            
            # Account/wallet addresses
            'account': 'account',
            'user': 'account',
            'wallet': 'account',
            'from': 'account',
            'minter': 'account',
            'borrower': 'account',
            'payer': 'account',
            
            # Secondary account (for liquidations)
            'liquidator': 'liquidator',
            'to': 'liquidator',
            
            # Amounts
            'amount': 'amount',
            'amount_raw': 'amount_raw',
            'amount_decimal': 'amount',
            'asset_amount': 'amount',
            'collateral_amount': 'collateral_amount',
            
            # Token/market info
            'token': 'token',
            'c_token': 'ctoken',
            'ctoken': 'ctoken',
            'asset': 'token',
            'underlying': 'token',
            'collateral': 'collateral_token',
        }
        
        # Standardize column names
        for old_col, new_col in column_mapping.items():
            if old_col in result.columns and new_col not in result.columns:
                result[new_col] = result[old_col]
        
        # Ensure all required columns exist
        required_columns = ['timestamp', 'transaction_hash', 'account', 'event_type']
        for col in required_columns:
            if col not in result.columns:
                logger.warning(f"Required column '{col}' not found. Creating empty column.")
                result[col] = None
        
        # Infer event type if missing
        if 'event_type' in result.columns and result['event_type'].isna().all():
            for event_type in self.event_types:
                # Check if event-specific column exists
                event_columns = {
                    'Mint': ['minter', 'mint_amount'],
                    'Redeem': ['redeemer', 'redeem_amount'],
                    'Borrow': ['borrower', 'borrow_amount'],
                    'RepayBorrow': ['payer', 'repay_amount'],
                    'LiquidateBorrow': ['liquidator', 'seized_amount'],
                }
                
                if any(col in result.columns for col in event_columns.get(event_type, [])):
                    result['event_type'] = event_type
                    break
        
        # Standardize addresses
        address_columns = ['account', 'liquidator', 'token', 'ctoken', 'collateral_token']
        for col in address_columns:
            if col in result.columns:
                result[col] = result[col].apply(standardize_address)
        
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in result.columns:
            if result['timestamp'].dtype != 'datetime64[ns]':
                try:
                    # Try to convert assuming Unix timestamp (seconds since epoch)
                    result['timestamp'] = pd.to_datetime(result['timestamp'], unit='s')
                except:
                    try:
                        # Try to convert assuming string format
                        result['timestamp'] = pd.to_datetime(result['timestamp'])
                    except:
                        logger.warning("Could not convert timestamp to datetime")
        
        return result
    
    def load_and_process_data(self, optimize=True):
        """
        Load and process transaction data.
        
        Args:
            optimize (bool): Whether to use optimized algorithms
        
        Returns:
            pd.DataFrame: Processed transaction data
        """
        # Get input files
        input_files = self.get_input_files()
        
        # Parse each file
        dfs = []
        for file_path in tqdm(input_files, desc="Processing files"):
            logger.info(f"Processing file: {os.path.basename(file_path)}")
            try:
                df = self.parse_file(file_path)
                dfs.append(df)
                logger.info(f"Successfully processed file with {len(df)} records")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
        
        if not dfs:
            raise ValueError("No data could be processed from the input files")
        
        # Combine all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Save to processed directory
        processed_file = os.path.join(self.processed_data_dir, "combined_transactions.parquet")
        combined_df.to_parquet(processed_file, index=False)
        logger.info(f"Saved processed data to {processed_file} ({len(combined_df)} records)")
        
        return combined_df

def main():
    """
    Main function to load and process data.
    """
    loader = DataLoader()
    df = loader.load_and_process_data()
    print(f"Processed {len(df)} transactions")
    print(f"Sample data:\n{df.head()}")
    
if __name__ == "__main__":
    main() 