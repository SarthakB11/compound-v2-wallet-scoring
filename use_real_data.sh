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

# Clear any existing data in raw directory
echo "Clearing any existing data in raw directory..."
rm -rf data/raw/* data/processed/* data/models/* results/*

# Copy the three largest files from the "Compound V2" folder to raw directory
echo "Copying real Compound V2 data files..."
mkdir -p data/raw

# The folder name has spaces, so we need to handle that properly
COMPOUND_DIR="data/Compound V2"
if [ -d "$COMPOUND_DIR" ]; then
    # Find the 3 largest files in the Compound V2 directory and copy them to raw
    find "$COMPOUND_DIR" -name "*.json" -type f -exec du -h {} \; | sort -hr | head -n 3 | awk '{print $2}' | xargs -I{} cp "{}" data/raw/
    
    # Verify the files were copied
    echo "Verifying copied files:"
    ls -lh data/raw/
    
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
else
    echo "Error: Compound V2 directory not found at $COMPOUND_DIR"
    exit 1
fi 