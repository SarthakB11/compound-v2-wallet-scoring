#!/bin/bash
# Simple script to copy Compound V2 data and run the pipeline

# Ensure we're in the project root directory
cd "$(dirname "$0")"

# Clear existing data in raw directory
echo "Clearing raw directory..."
rm -rf data/raw/*

# Create raw directory if it doesn't exist
mkdir -p data/raw

# Copy the three largest files from 'Compound V2' directory manually
echo "Copying the largest files from 'Compound V2' directory..."
COMPOUND_DIR="data/Compound V2"

# List the files
ls -l "$COMPOUND_DIR"

# Copy the three largest files (chunks 0, 1, and 2) to raw directory
cp "$COMPOUND_DIR/compoundV2_transactions_ethereum_chunk_0.json" data/raw/
cp "$COMPOUND_DIR/compoundV2_transactions_ethereum_chunk_1.json" data/raw/
cp "$COMPOUND_DIR/compoundV2_transactions_ethereum_chunk_2.json" data/raw/

# Verify the copied files
echo "Verifying copied files..."
ls -lh data/raw/

# Run the pipeline
echo "Running the pipeline with real data..."
python src/main.py

if [ $? -ne 0 ]; then
    echo "Error: Pipeline failed. Check logs for details."
    exit 1
fi

echo "Pipeline completed successfully!"
echo "Results are available in the results/ directory." 