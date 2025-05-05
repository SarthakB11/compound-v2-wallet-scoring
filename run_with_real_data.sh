#!/bin/bash
# Script to run the Compound V2 wallet scoring pipeline with real data

# Ensure we're in the project root directory
cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Clear any existing data to ensure we download fresh data
echo "Clearing any existing data..."
rm -rf data/raw/* data/processed/* data/models/* results/*

# Download and prepare real dataset
echo "Downloading real Compound V2 dataset..."
python src/data/download.py

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to download real Compound V2 data. Exiting."
    exit 1
fi

# Run the complete pipeline with real data
echo "Running the complete pipeline with real Compound V2 data..."
python src/main.py

if [ $? -ne 0 ]; then
    echo "Error: Pipeline failed. Check logs for details."
    exit 1
fi

echo "Pipeline completed successfully!"
echo "Results are available in the results/ directory."
echo "To view the top-scoring wallets: cat results/wallet_scores.csv" 