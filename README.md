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
  - Advanced ML models and deep learning
- **Wallet behavior analysis** with insights on high and low scoring wallets
- **Advanced optimization components**:
  - Feature selection to identify the most important indicators
  - Automated hyperparameter tuning
  - Deep learning models for complex pattern recognition
  - Model integration for ensemble predictions

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
│   ├── advanced/       # Advanced optimization components
│   │   ├── feature_selection.py    # Feature selection techniques
│   │   ├── hyperparameter_tuning.py # Hyperparameter optimization
│   │   ├── deep_learning.py        # Neural network models
│   │   └── model_integration.py    # Ensemble model integration
│   └── utils/          # Utility functions
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
└── PROJECT_CONTEXT.md  # Implementation context
```

## Requirements
- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter, tqdm, polars, pyarrow, joblib, plotly
- Advanced components: tensorflow, optuna, xgboost

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
Place your Compound V2 transaction data in the `data/raw/` directory:
- `transactions.csv`: Contains transaction records
- `users.csv`: Contains user information

## Usage

### Basic Pipeline
Run the complete pipeline:
```bash
python src/pipeline.py
```

This will:
1. Load and preprocess transaction data
2. Engineer features for each wallet
3. Calculate heuristic scores
4. Generate final wallet scores

### Advanced Pipeline
Run the pipeline with advanced optimization components:
```bash
python src/pipeline.py --advanced
```

This will run all basic steps plus:
1. Feature selection to identify optimal predictors
2. Hyperparameter tuning for traditional models
3. Deep learning model training
4. Model integration with ensemble techniques

### Specific Steps
Run only specific steps of the pipeline:
```bash
python src/pipeline.py --steps load,preprocess,engineer
```

### Output
The final scores will be saved in:
- `results/wallet_scores.csv`: Basic pipeline results
- `results/integrated_wallet_scores.csv`: Advanced pipeline with ensemble models

## Configuration
You can configure the system by modifying parameters in `src/config.py`:

- `FEATURE_WEIGHTS`: Adjust the importance of different features
- `ANOMALY_DETECTION`: Configure anomaly detection parameters
- `SCORING`: Change the method for transforming raw scores to final scores
- `OUTPUT_SETTINGS`: Modify output file settings
- `ADVANCED_CONFIG`: Settings for advanced optimization components

## Advanced Optimization Components

### Feature Selection
The feature selection module identifies the most predictive features using multiple techniques:
- Correlation analysis
- Mutual information
- Random forest importance
- Recursive feature elimination

```bash
python -m src.advanced.feature_selection --method combined --n-features 15
```

### Hyperparameter Tuning
Automatically finds optimal model parameters using:
- Grid search
- Random search
- Bayesian optimization with Optuna

```bash
python -m src.advanced.hyperparameter_tuning --model random_forest --method optuna
```

### Deep Learning Models
Neural network models for wallet scoring:
- Dense architectures of varying depth
- Sequence models for transaction history
- Regularization techniques to prevent overfitting

```bash
python -m src.advanced.deep_learning --architecture compare
```

### Model Integration
Combines predictions from multiple models:
- Voting ensemble
- Stacking ensemble
- Hybrid predictions with deep learning

```bash
python -m src.advanced.model_integration
```

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

