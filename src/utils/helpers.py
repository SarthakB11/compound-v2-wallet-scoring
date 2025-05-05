"""
Helper utilities for the Compound V2 Wallet Scoring project.
"""
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_largest_files(directory, n=3, pattern=None):
    """
    Get the n largest files from a directory, optionally filtering by pattern.
    
    Args:
        directory (str): Path to the directory
        n (int): Number of files to return
        pattern (str, optional): Glob pattern to filter files (e.g., "*.json")
        
    Returns:
        list: List of file paths, sorted by size (largest first)
    """
    # Get files matching the pattern if provided
    if pattern:
        file_paths = glob.glob(os.path.join(directory, pattern))
        files = [(f, os.path.getsize(f)) for f in file_paths if os.path.isfile(f)]
    else:
        files = [(os.path.join(directory, f), os.path.getsize(os.path.join(directory, f))) 
                for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Sort by file size (descending)
    files.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top n files
    return [f[0] for f in files[:n]]

def standardize_address(address):
    """
    Standardize Ethereum address format.
    
    Args:
        address (str): Ethereum address
        
    Returns:
        str: Standardized address in lowercase
    """
    if isinstance(address, str):
        # Remove '0x' prefix if present and convert to lowercase
        if address.startswith('0x'):
            address = address[2:]
        return address.lower()
    return address

def plot_distribution(data, column_name, title=None, bins=50, figsize=(10, 6), save_path=None):
    """
    Plot the distribution of a column in a DataFrame.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data
        column_name (str): Column to plot
        title (str, optional): Plot title
        bins (int, optional): Number of bins for histogram
        figsize (tuple, optional): Figure size
        save_path (str, optional): Path to save the figure
        
    Returns:
        None
    """
    plt.figure(figsize=figsize)
    sns.histplot(data[column_name], bins=bins, kde=True)
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Distribution of {column_name}')
    
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
    plt.show()

def calculate_time_diff_days(start_time, end_time):
    """
    Calculate the difference in days between two timestamps.
    
    Args:
        start_time (pd.Timestamp or datetime): Start time
        end_time (pd.Timestamp or datetime): End time
        
    Returns:
        float: Difference in days
    """
    if isinstance(start_time, str):
        start_time = pd.to_datetime(start_time)
    if isinstance(end_time, str):
        end_time = pd.to_datetime(end_time)
        
    delta = end_time - start_time
    return delta.total_seconds() / (60 * 60 * 24)  # Convert to days

def save_dataframe(df, filepath, index=False):
    """
    Save DataFrame to a file with proper directory creation.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filepath (str): Path to save the DataFrame
        index (bool, optional): Whether to include index
        
    Returns:
        None
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save DataFrame based on file extension
    if filepath.endswith('.csv'):
        df.to_csv(filepath, index=index)
    elif filepath.endswith('.parquet'):
        df.to_parquet(filepath, index=index)
    elif filepath.endswith('.pkl'):
        df.to_pickle(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    logger.info(f"Saved DataFrame with {len(df)} rows to {filepath}")

def load_dataframe(filepath):
    """
    Load DataFrame from a file.
    
    Args:
        filepath (str): Path to the DataFrame file
        
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Load DataFrame based on file extension
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    elif filepath.endswith('.pkl'):
        df = pd.read_pickle(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    logger.info(f"Loaded DataFrame with {len(df)} rows from {filepath}")
    return df 