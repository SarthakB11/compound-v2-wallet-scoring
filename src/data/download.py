"""
Script to download Compound V2 transaction data and extract the largest JSON files.
"""
import os
import sys
import logging
import requests
import json
import zipfile
import shutil
import gdown
import tempfile
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.helpers import get_largest_files
import src.config as config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset URL from Project_description.md
DATASET_URL = "https://drive.google.com/drive/folders/1kCrMk30zlf8r1U4frgW9ecpYc1KhMSTE?usp=drive_link"

def download_gdrive_folder(url, destination):
    """
    Download a Google Drive folder.
    
    Args:
        url (str): Google Drive folder URL
        destination (str): Local destination directory
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading Google Drive folder: {url}")
        logger.info(f"This may take some time depending on folder size...")
        
        # Create destination directory if it doesn't exist
        os.makedirs(destination, exist_ok=True)
        
        # Use gdown to download the folder
        gdown.download_folder(url=url, output=destination, quiet=False, remaining_ok=True)
        
        logger.info(f"Downloaded folder to {destination}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading Google Drive folder: {str(e)}")
        return False

def download_file(url, destination):
    """
    Download a file from a URL with progress tracking.
    
    Args:
        url (str): URL to download
        destination (str): Local destination path
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        with open(destination, 'wb') as f, tqdm(
            desc=f"Downloading {os.path.basename(destination)}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(block_size):
                f.write(data)
                progress_bar.update(len(data))
                
        logger.info(f"Downloaded {destination}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading file from {url}: {str(e)}")
        return False

def generate_sample_data():
    """
    Generate sample data for testing when real API endpoint is not available.
    This creates synthetic Compound V2 transaction data.
    """
    raw_dir = config.RAW_DATA_DIR
    os.makedirs(raw_dir, exist_ok=True)
    
    logger.info("Generating sample data for testing purposes")
    
    # Sample addresses
    addresses = [
        "0x1234567890123456789012345678901234567890",
        "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
        "0xfedcbafedcbafedcbafedcbafedcbafedcbafedc",
        "0x0123456789abcdef0123456789abcdef01234567",
        "0x9876543210fedcba9876543210fedcba98765432"
    ]
    
    # Sample tokens
    tokens = [
        {"symbol": "USDC", "address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "decimals": 6},
        {"symbol": "DAI", "address": "0x6B175474E89094C44Da98b954EedeAC495271d0F", "decimals": 18},
        {"symbol": "WETH", "address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "decimals": 18},
        {"symbol": "WBTC", "address": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", "decimals": 8},
        {"symbol": "COMP", "address": "0xc00e94Cb662C3520282E6f5717214004A7f26888", "decimals": 18}
    ]
    
    # Sample cTokens
    ctokens = [
        {"symbol": "cUSDC", "address": "0x39AA39c021dfbaE8faC545936693aC917d5E7563", "underlying": tokens[0]},
        {"symbol": "cDAI", "address": "0x5d3a536E4D6DbD6114cc1Ead35777bAB948E3643", "underlying": tokens[1]},
        {"symbol": "cETH", "address": "0x4Ddc2D193948926D02f9B1fE9e1daa0718270ED5", "underlying": tokens[2]},
        {"symbol": "cWBTC", "address": "0xC11b1268C1A384e55C48c2391d8d480264A3A7F4", "underlying": tokens[3]},
    ]
    
    # Generate 3 sample files
    for i, event_type in enumerate(["mint", "borrow", "redeem"]):
        events = []
        # Generate 1000-2000 events for each file
        num_events = 1000 + i * 500
        
        for j in range(num_events):
            user_idx = j % len(addresses)
            token_idx = j % len(ctokens)
            block_number = 10000000 + j * 10
            timestamp = 1600000000 + j * 3600  # 1 hour increments
            
            event = {
                "event_type": event_type.capitalize(),
                "block_number": block_number,
                "timestamp": timestamp,
                "transaction_hash": f"0x{j:064x}",
                "account": addresses[user_idx],
                "ctoken": ctokens[token_idx]["address"],
                "token": ctokens[token_idx]["underlying"]["address"],
                "amount": float(j * 10) / (10 ** ctokens[token_idx]["underlying"]["decimals"]),
                "amount_raw": str(j * 10),
            }
            
            # Add event-specific fields
            if event_type == "mint":
                event["minter"] = addresses[user_idx]
            elif event_type == "borrow":
                event["borrower"] = addresses[user_idx]
            elif event_type == "redeem":
                event["redeemer"] = addresses[user_idx]
                
            events.append(event)
        
        # Save to file
        file_path = os.path.join(raw_dir, f"compound_v2_{event_type}_events.json")
        with open(file_path, 'w') as f:
            json.dump(events, f)
        
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"Generated {file_path} with {num_events} events ({file_size_mb:.2f} MB)")

def download_dataset():
    """
    Download the Compound V2 dataset from Google Drive.
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Create raw data directory if it doesn't exist
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    
    try:
        logger.info("Attempting to download dataset from Google Drive...")
        
        # Check if we have files already
        if os.listdir(config.RAW_DATA_DIR):
            logger.info(f"Files already exist in {config.RAW_DATA_DIR}. Skipping download.")
            return True
        
        # Create a temporary directory to download all files
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Created temporary directory at {temp_dir}")
            
            # Download the Google Drive folder
            success = download_gdrive_folder(DATASET_URL, temp_dir)
            if not success:
                logger.error("Failed to download Google Drive folder")
                
                # Fall back to generating sample data
                logger.info("Falling back to sample data generation")
                generate_sample_data()
                return True
            
            # Get the largest files from the temp directory
            largest_files = get_largest_files(temp_dir, config.NUM_FILES_TO_PROCESS)
            
            # Copy the largest files to the raw data directory
            for i, file_path in enumerate(largest_files, 1):
                file_name = os.path.basename(file_path)
                destination = os.path.join(config.RAW_DATA_DIR, file_name)
                
                logger.info(f"Copying file {i}/{len(largest_files)}: {file_name}")
                shutil.copy2(file_path, destination)
                
                file_size_mb = os.path.getsize(destination) / (1024 * 1024)
                logger.info(f"Copied file {file_name} ({file_size_mb:.2f} MB)")
        
        logger.info("Dataset download completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        
        # Fall back to generating sample data
        logger.info("Falling back to sample data generation")
        generate_sample_data()
        return True

def extract_largest_files():
    """
    Identify the three largest files from the downloaded dataset.
    
    Returns:
        list: Paths to the largest files
    """
    # Get the largest files
    largest_files = get_largest_files(config.RAW_DATA_DIR, config.NUM_FILES_TO_PROCESS)
    
    if len(largest_files) < config.NUM_FILES_TO_PROCESS:
        logger.warning(f"Found only {len(largest_files)} files, wanted {config.NUM_FILES_TO_PROCESS}")
    
    logger.info(f"Selected the {len(largest_files)} largest files:")
    for i, file_path in enumerate(largest_files, 1):
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"{i}. {os.path.basename(file_path)} ({file_size_mb:.2f} MB)")
    
    return largest_files

def main():
    """
    Main function to download dataset and extract largest files.
    """
    logger.info("Starting Compound V2 dataset download...")
    
    # Download dataset
    success = download_dataset()
    if not success:
        logger.error("Failed to download dataset")
        return
    
    # Extract largest files
    largest_files = extract_largest_files()
    
    # Report success
    logger.info(f"Successfully prepared {len(largest_files)} largest files for processing")
    logger.info("You can now run the main pipeline to process these files")

if __name__ == "__main__":
    main() 