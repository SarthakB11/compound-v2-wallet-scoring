#!/bin/bash
# Script to run the complete Compound V2 wallet scoring pipeline with algorithmic optimizations

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

# Download and prepare dataset
echo "Downloading and preparing dataset..."
python src/data/download.py

# Run the complete pipeline with algorithmic optimizations
echo "Running the complete pipeline with optimized algorithms..."
python src/main.py

echo "Pipeline completed! Results are available in the results/ directory."
echo "To view the top-scoring wallets: cat results/wallet_scores.csv" 