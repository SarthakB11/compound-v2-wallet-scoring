# Notebook Contents

The `data_exploration.ipynb` notebook is designed to help you explore the Compound V2 transaction data. Due to system limitations, it might be best to manually create this notebook based on the following content structure:

## Cell Structure for Data Exploration Notebook

### Markdown Cell: Title
```markdown
# Compound V2 Data Exploration

This notebook explores the Compound V2 transaction data to understand its structure and characteristics.
```

### Code Cell: Imports
```python
# Import necessary libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append('..')

# Import project modules
from src.utils.helpers import get_largest_files
from src.data.loader import DataLoader
import src.config as config
```

### Markdown Cell: Section 1
```markdown
## 1. Find and Inspect the Largest Data Files
```

### Code Cell: Find Large Files
```python
# Directory containing raw data files
data_dir = os.path.join('..', 'data', 'raw')

# Check if files exist
if os.path.exists(data_dir) and os.listdir(data_dir):
    # Get the largest files
    large_files = get_largest_files(data_dir, n=3)
    
    print(f"Found {len(large_files)} files:")
    for file in large_files:
        file_size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"- {os.path.basename(file)}: {file_size_mb:.2f} MB")
else:
    print("No data files found. Please download the dataset first.")
```

### Markdown Cell: Section 2
```markdown
## 2. Sample and Inspect Data Format
```

### Code Cell: Inspect Data
```python
# Initialize the data loader
loader = DataLoader()

# Check if files exist before trying to load
try:
    # Get the list of input files
    input_files = loader.get_input_files()
    
    # Take the first file for inspection
    if input_files:
        first_file = input_files[0]
        print(f"Inspecting file: {os.path.basename(first_file)}")
        
        # Infer file format
        file_format = loader.infer_file_format(first_file)
        print(f"Detected format: {file_format}")
        
        # Parse the file
        try:
            df = loader.parse_file(first_file)
            print(f"Successfully parsed {len(df)} records")
            print(f"\nColumns: {df.columns.tolist()}")
            print(f"\nSample data:")
            display(df.head())
        except Exception as e:
            print(f"Error parsing file: {str(e)}")
except FileNotFoundError:
    print("No data files found. Please download the dataset first.")
```

### Markdown Cell: Section 3
```markdown
## 3. Analyze Event Types
```

### Code Cell: Event Analysis
```python
# Continue if we have data
if 'df' in locals() and len(df) > 0:
    # Count event types
    if 'event_type' in df.columns:
        event_counts = df['event_type'].value_counts()
        print("Event type distribution:")
        print(event_counts)
        
        # Visualize
        plt.figure(figsize=(10, 6))
        sns.barplot(x=event_counts.index, y=event_counts.values)
        plt.title('Event Type Distribution')
        plt.xlabel('Event Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("No 'event_type' column found in the data")
```

### Markdown Cell: Section 4
```markdown
## 4. Analyze Temporal Distribution
```

### Code Cell: Temporal Analysis
```python
# Continue if we have data
if 'df' in locals() and len(df) > 0:
    # Check if timestamp column exists
    if 'timestamp' in df.columns:
        # Ensure timestamp is datetime
        if df['timestamp'].dtype != 'datetime64[ns]':
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                print("Could not convert timestamp to datetime")
        
        if df['timestamp'].dtype == 'datetime64[ns]':
            # Extract date for daily counts
            df['date'] = df['timestamp'].dt.date
            daily_counts = df.groupby('date').size()
            
            # Plot daily transaction counts
            plt.figure(figsize=(15, 6))
            daily_counts.plot()
            plt.title('Daily Transaction Volume')
            plt.xlabel('Date')
            plt.ylabel('Number of Transactions')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            # Report time range
            min_date = df['timestamp'].min()
            max_date = df['timestamp'].max()
            print(f"Data spans from {min_date} to {max_date}")
            print(f"Total time span: {(max_date - min_date).days} days")
    else:
        print("No 'timestamp' column found in the data")
```

### Markdown Cell: Section 5
```markdown
## 5. Analyze Wallet Distribution
```

### Code Cell: Wallet Analysis
```python
# Continue if we have data
if 'df' in locals() and len(df) > 0:
    # Check if account column exists
    account_col = None
    for col in ['account', 'wallet', 'user', 'from', 'minter', 'borrower']:
        if col in df.columns:
            account_col = col
            break
            
    if account_col:
        # Count transactions per wallet
        wallet_counts = df[account_col].value_counts()
        
        # Basic statistics
        print(f"Number of unique wallets: {len(wallet_counts)}")
        print(f"Average transactions per wallet: {wallet_counts.mean():.2f}")
        print(f"Median transactions per wallet: {wallet_counts.median():.2f}")
        print(f"Max transactions per wallet: {wallet_counts.max()}")
        
        # Distribution of transaction counts
        plt.figure(figsize=(12, 6))
        sns.histplot(wallet_counts, log_scale=True)
        plt.title('Distribution of Transactions per Wallet (Log Scale)')
        plt.xlabel('Number of Transactions (Log Scale)')
        plt.ylabel('Number of Wallets')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Show top wallets by transaction count
        print("\nTop 10 wallets by transaction count:")
        display(wallet_counts.head(10).reset_index().rename(
            columns={account_col: 'Transaction Count', 'index': 'Wallet'}))
    else:
        print("No wallet/account column found in the data")
```

### Markdown Cell: Section 6
```markdown
## 6. Event Type by Wallet
```

### Code Cell: Event Types by Wallet
```python
# Continue if we have data and the necessary columns
if 'df' in locals() and len(df) > 0 and account_col and 'event_type' in df.columns:
    # Cross-tabulate event types by wallet
    wallet_events = pd.crosstab(df[account_col], df['event_type'])
    
    # Add a total column
    wallet_events['Total'] = wallet_events.sum(axis=1)
    
    # Display the first few rows
    print("Event types per wallet:")
    display(wallet_events.head())
    
    # Calculate percentage of wallets with each event type
    event_pct = (wallet_events > 0).mean() * 100
    
    # Visualize
    plt.figure(figsize=(10, 6))
    event_pct.drop('Total').plot(kind='bar')
    plt.title('Percentage of Wallets with Each Event Type')
    plt.xlabel('Event Type')
    plt.ylabel('Percentage of Wallets')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
```

### Markdown Cell: Section 7
```markdown
## 7. Liquidation Analysis
```

### Code Cell: Liquidation Analysis
```python
# Continue if we have data
if 'df' in locals() and len(df) > 0 and 'event_type' in df.columns:
    # Filter to liquidation events
    liquidation_events = df[df['event_type'].str.contains('Liquidate', case=False, na=False)]
    
    if len(liquidation_events) > 0:
        print(f"Found {len(liquidation_events)} liquidation events")
        
        # Check if we have liquidator information
        liquidator_col = None
        for col in ['liquidator', 'to']:
            if col in liquidation_events.columns:
                liquidator_col = col
                break
        
        if liquidator_col and account_col:
            # Count liquidations by borrower
            borrower_liquidations = liquidation_events[account_col].value_counts()
            
            # Count liquidations by liquidator
            liquidator_counts = liquidation_events[liquidator_col].value_counts()
            
            print(f"Number of borrowers liquidated: {len(borrower_liquidations)}")
            print(f"Number of liquidators: {len(liquidator_counts)}")
            
            # Show top liquidated borrowers
            print("\nTop 10 liquidated borrowers:")
            display(borrower_liquidations.head(10).reset_index().rename(
                columns={account_col: 'Liquidation Count', 'index': 'Borrower'}))
            
            # Show top liquidators
            print("\nTop 10 liquidators:")
            display(liquidator_counts.head(10).reset_index().rename(
                columns={liquidator_col: 'Liquidation Count', 'index': 'Liquidator'}))
            
            # Temporal distribution of liquidations
            if 'timestamp' in liquidation_events.columns and liquidation_events['timestamp'].dtype == 'datetime64[ns]':
                liquidation_events['date'] = liquidation_events['timestamp'].dt.date
                daily_liquidations = liquidation_events.groupby('date').size()
                
                plt.figure(figsize=(15, 6))
                daily_liquidations.plot()
                plt.title('Daily Liquidation Volume')
                plt.xlabel('Date')
                plt.ylabel('Number of Liquidations')
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.show()
        else:
            print("Missing liquidator or borrower columns for detailed analysis")
    else:
        print("No liquidation events found in the sample data")
```

### Markdown Cell: Section 8
```markdown
## 8. Transaction Amount Analysis
```

### Code Cell: Amount Analysis
```python
# Continue if we have data
if 'df' in locals() and len(df) > 0:
    # Check for amount columns
    amount_cols = [col for col in df.columns if 'amount' in col.lower()]
    
    if amount_cols:
        print(f"Found amount columns: {amount_cols}")
        
        # Use the first amount column for analysis
        amount_col = amount_cols[0]
        
        # Ensure numeric
        if df[amount_col].dtype != 'float64' and df[amount_col].dtype != 'int64':
            try:
                df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
                print(f"Converted {amount_col} to numeric")
            except:
                print(f"Could not convert {amount_col} to numeric")
        
        if df[amount_col].dtype == 'float64' or df[amount_col].dtype == 'int64':
            # Basic statistics
            print(f"\nTransaction amount statistics:")
            print(f"Min: {df[amount_col].min()}")
            print(f"Max: {df[amount_col].max()}")
            print(f"Mean: {df[amount_col].mean()}")
            print(f"Median: {df[amount_col].median()}")
            
            # Distribution by event type
            if 'event_type' in df.columns:
                plt.figure(figsize=(12, 6))
                sns.boxplot(x='event_type', y=amount_col, data=df)
                plt.title(f'Transaction Amount by Event Type')
                plt.xlabel('Event Type')
                plt.ylabel('Amount')
                plt.yscale('log')  # Log scale for better visibility
                plt.xticks(rotation=45)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.show()
    else:
        print("No amount columns found in the data")
```

### Markdown Cell: Section 9
```markdown
## 9. Summary and Insights
```

### Markdown Cell: Summary
```markdown
**Key observations from the data exploration:**

1. **Data Structure:** Describe the structure of the data based on what you found
2. **Event Distribution:** Summarize the distribution of event types
3. **Wallet Behavior:** Note patterns in how wallets interact with the protocol
4. **Liquidations:** Any insights on liquidation patterns
5. **Transaction Patterns:** Observations about transaction amounts and frequency

**Implications for Scoring Strategy:**

Based on the observations, the scoring model should consider:
- Historical liquidations as a strong negative signal
- Transaction consistency and tenure as positive signals
- Diversity of interactions across different event types
- Balance of borrow vs supply activities
```

You can manually create a notebook with this content using Jupyter Notebook or JupyterLab. Simply create a new notebook and copy these cells in sequence. 