# Compound V2 Wallet Scoring - Implementation Context

## Project Overview
This project implements a machine learning model to score wallets interacting with the Compound V2 protocol based on their transaction history. The scores range from 0-100, with higher scores indicating more reliable and responsible usage.

## Implementation Progress

### Step 1: Project Setup ✅
- [x] Create project directory structure
- [x] Set up environment and dependencies
- [x] Create modules and configuration

### Step 2: Data Processing ✅
- [x] Implement data loader
- [x] Implement data preprocessor
- [x] Implement wallet aggregation

### Step 3: Feature Engineering ✅
- [x] Implement transaction activity features
- [x] Implement temporal pattern features
- [x] Implement financial health features
- [x] Implement protocol interaction features

### Step 4: Model Development ✅
- [x] Implement anomaly detection component
- [x] Implement heuristic scoring system
- [x] Implement hybrid model

### Step 5: Scoring and Output ✅
- [x] Implement score transformation
- [x] Generate sorted wallet scores
- [x] Perform wallet analysis

### Step 6: Integration and Pipeline ✅
- [x] Create main pipeline module
- [x] Add command-line interface
- [x] Create data exploration notebook

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

## Usage Instructions
1. Download the Compound V2 dataset and place the largest files in the `data/raw` directory
2. Run the full pipeline: `python src/main.py`
3. For exploring the data: use `notebooks/data_exploration.ipynb`
4. For running specific stages: `python src/main.py --skip-to [stage]`
   - Available stages: 'load', 'preprocess', 'features', 'anomaly', 'heuristic', 'score'

## Key Decisions
- Working with the 3 largest files from the Compound V2 dataset
- Using a hybrid approach with heuristic scoring and anomaly detection
- Features focused on transaction activity, temporal patterns, financial health, and protocol interaction
- Score range: 0-100 with percentile ranking
- Heavy penalties for liquidations and high-risk behavior
- Bonuses for long-term consistent users 