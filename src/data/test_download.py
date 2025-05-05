"""
Test script for the download functionality.
This script verifies that the download.py script works correctly.
"""
import os
import sys
from pathlib import Path
import logging

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.download import download_dataset, extract_largest_files
import src.config as config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_download():
    """
    Test the download functionality.
    """
    logger.info("Testing dataset download...")
    
    # Clear raw directory for testing
    raw_dir = config.RAW_DATA_DIR
    
    # Create directory if it doesn't exist
    os.makedirs(raw_dir, exist_ok=True)
    
    # Check if directory is empty
    if os.listdir(raw_dir):
        logger.info(f"Raw directory {raw_dir} is not empty. Skipping download test.")
        return True
    
    # Download dataset
    success = download_dataset()
    if not success:
        logger.error("Dataset download failed!")
        return False
    
    # Check if files were downloaded
    if not os.listdir(raw_dir):
        logger.error("No files were downloaded!")
        return False
    
    # Extract largest files
    largest_files = extract_largest_files()
    if not largest_files:
        logger.error("No largest files were extracted!")
        return False
    
    # Print file info
    logger.info(f"Successfully extracted {len(largest_files)} largest files:")
    for i, file_path in enumerate(largest_files, 1):
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"{i}. {os.path.basename(file_path)} ({file_size_mb:.2f} MB)")
    
    logger.info("Download test completed successfully!")
    return True

if __name__ == "__main__":
    test_download() 