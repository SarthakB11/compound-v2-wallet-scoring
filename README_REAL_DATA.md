# Running the Compound V2 Wallet Scoring Project with Real Data

This guide explains how to run the Compound V2 wallet scoring pipeline using real Compound V2 transaction data.

## Prerequisites

- Python 3.6+
- Required dependencies (see `requirements.txt`)

## Real Data Location

The real Compound V2 data is located in the `data/Compound V2/` directory. This data consists of multiple JSON files containing real Compound V2 transaction data.

## How to Run the Pipeline

We've created a simple script that will run the pipeline with the real Compound V2 data. Just follow these steps:

1. Clone the repository (if you haven't already)
2. Navigate to the project root directory
3. Run the script:

```bash
./copy_and_run.sh
```

This script will:
1. Copy the three largest files from the `data/Compound V2/` directory to the `data/raw/` directory
2. Run the complete wallet scoring pipeline on this real data

## Understanding the Data Format

The real Compound V2 data is structured differently than the mock data:

- The JSON files contain a top-level dictionary with keys like `deposits`, `borrows`, `withdraws`, `repays`, and `liquidations`
- Each of these keys contains an array of transaction objects
- Each transaction object has nested structures for `account` and `asset` information
- Transaction timestamps are Unix timestamps (in seconds)

The pipeline has been updated to handle this format correctly.

## Results

After running the pipeline, you can find the results in the `results/` directory:

- `wallet_scores.csv`: Contains the top 1000 wallets with their scores
- `all_wallet_scores.parquet`: Contains scores for all wallets (Parquet format)
- `wallet_analysis.csv`: Detailed analysis of the top and bottom wallets
- `wallet_analysis_summary.md`: A human-readable summary of the most interesting wallets

## Troubleshooting

If you encounter any issues:

1. Make sure you have all the required dependencies installed
2. Check the logs in the `logs/` directory for detailed error messages
3. Verify that the real data files are accessible in the `data/Compound V2/` directory

For any other issues, please open an issue on the project repository. 