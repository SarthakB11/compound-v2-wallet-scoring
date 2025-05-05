# Compound V2 Wallet Credit Scoring

## Overview
This project implements a machine learning model to score wallets interacting with the Compound V2 protocol based on their transaction behavior. The scoring system assigns values between 0-100 to each wallet, with higher scores indicating more reliable and responsible usage, and lower scores reflecting risky or exploitative behavior.

## Features
- **Data processing pipeline** for Compound V2 protocol transaction data
- **Comprehensive feature engineering** for wallet behavior analysis:
  - Transaction activity patterns
  - Temporal usage patterns
  - Financial health metrics
  - Protocol interaction patterns
- **Hybrid scoring model** combining:
  - Heuristic rule-based scoring with feature weights
  - Anomaly detection for identifying unusual behavior patterns
- **Wallet behavior analysis** with insights on high and low scoring wallets

## Project Structure
```
├── data/               # Data files
│   ├── raw/            # Original data files
│   └── processed/      # Processed data
├── logs/               # Log files
├── models/             # Saved models
├── notebooks/          # Jupyter notebooks for exploration
├── results/            # Output scores and analysis
├── src/                # Source code
│   ├── data/           # Data loading and processing
│   ├── features/       # Feature engineering
│   ├── models/         # Model implementation
│   ├── scoring/        # Scoring system
│   └── utils/          # Utility functions
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
└── PROJECT_CONTEXT.md  # Implementation context
```

## Requirements
- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter, tqdm, polars, pyarrow, joblib, plotly

## Setup and Installation

### Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd compound-v2-wallet-scoring

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation
1. Download the Compound V2 dataset from the provided Google Drive link
2. Select the 3 largest files from the dataset
3. Place the files in the `data/raw` directory

## Running the Pipeline

### Full Pipeline
To run the complete pipeline:

```bash
python src/main.py
```

This will execute all steps:
1. Load and process raw data
2. Preprocess data
3. Generate features
4. Detect anomalies
5. Calculate heuristic scores
6. Generate final scores and output

### Resume Pipeline
If you want to resume from a specific stage:

```bash
python src/main.py --skip-to [stage]
```

Available stages:
- `load`: Start from loading data
- `preprocess`: Start from preprocessing
- `features`: Start from feature engineering
- `anomaly`: Start from anomaly detection
- `heuristic`: Start from heuristic scoring
- `score`: Start from final scoring

### Individual Components
You can also run individual components:

```bash
# Data loading
python src/data/loader.py

# Data preprocessing
python src/data/preprocessor.py

# Feature engineering
python src/features/feature_engineering.py

# Anomaly detection
python src/models/anomaly_detector.py

# Heuristic scoring
python src/models/heuristic_scorer.py

# Final scoring
python src/scoring/scorer.py
```

## Data Exploration
A Jupyter notebook is provided for exploring the Compound V2 data. The notebook template is available in `notebooks/README.md`. You can create a new notebook with these cells to analyze the transaction data.

## Output
After running the pipeline, the following outputs will be generated:

- `results/wallet_scores.csv`: CSV file with the top 1,000 wallets, sorted by score
- `results/all_wallet_scores.parquet`: Parquet file with scores for all wallets
- `results/wallet_analysis.csv`: CSV file with detailed analysis of top and bottom wallets
- `results/wallet_analysis_summary.md`: Markdown document with behavioral patterns analysis

## Methodology
The scoring methodology is based on:

1. **Feature Engineering**: Extracting meaningful patterns from transaction data
2. **Anomaly Detection**: Identifying unusual wallet behavior
3. **Heuristic Scoring**: Assigning weights to different behavioral features
4. **Score Normalization**: Transforming raw scores to the 0-100 range

Key scoring factors:
- Consistency of participation
- Prudent collateralization maintenance
- Timely repayments
- Market diversification
- Liquidation avoidance
- Reasonable leverage

## Customization
You can customize the model by modifying parameters in `src/config.py`:

- `FEATURE_WEIGHTS`: Adjust the importance of different features
- `ANOMALY_DETECTION`: Configure anomaly detection parameters
- `SCORING`: Change the method for transforming raw scores to final scores
- `OUTPUT_SETTINGS`: Modify output file settings

## License
This project is provided for educational and research purposes.

## Acknowledgments
- Compound V2 Protocol
- Zeru Finance for the problem statement

## Performance Optimizations

The project includes several performance optimizations to speed up processing and handle large-scale data efficiently:

### Parallel Processing
- Multiprocessing support throughout the pipeline
- Configurable number of CPU cores with `--n-jobs` parameter
- Automatic detection of available cores in `run_pipeline.sh`

### Caching Mechanisms
- Feature caching to avoid recalculating expensive operations
- Intelligent file caching with timestamp-based invalidation
- In-memory data caching for repeated operations

### Algorithmic Improvements
- Vectorized operations for feature calculations
- Batch processing for scoring calculations
- Early termination for unchanged data

### Usage
Run the pipeline with parallel processing:
```bash
./run_pipeline.sh
```

Or specify the number of cores manually:
```bash
python src/main.py --n-jobs 4
```

For individual components:
```bash
python src/features/feature_engineering.py --n-jobs 4
python src/models/heuristic_scorer.py --n-jobs 4
```

These optimizations significantly reduce processing time, especially for large datasets and complex feature calculations.

