"""
Final scoring module for generating wallet credit scores.
"""
import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

import src.config as config

logger = logging.getLogger(__name__)

class WalletScorer:
    """
    Class for generating final credit scores for wallets.
    """
    
    def __init__(self, processed_data_dir=None, results_dir=None):
        """
        Initialize the WalletScorer.
        
        Args:
            processed_data_dir (str, optional): Directory containing processed data
            results_dir (str, optional): Directory to save results
        """
        self.processed_data_dir = processed_data_dir or config.PROCESSED_DATA_DIR
        self.results_dir = results_dir or config.RESULTS_DIR
        
        # Scoring parameters
        self.scoring_method = config.SCORING.get('method', 'percentile')
        self.reverse = config.SCORING.get('reverse', True)
        self.sigmoid_params = config.SCORING.get('sigmoid_params', {'k': 1.0, 'x0': 0.0})
        
        # Output settings
        self.top_n_wallets = config.OUTPUT_SETTINGS.get('top_n_wallets', 1000)
        self.output_filename = config.OUTPUT_SETTINGS.get('output_filename', 'wallet_scores.csv')
        
        # Ensure directories exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_heuristic_scores(self):
        """
        Load heuristic scores.
        
        Returns:
            pd.DataFrame: Heuristic scores
        """
        score_file_path = os.path.join(self.processed_data_dir, "heuristic_scores.parquet")
        if not os.path.exists(score_file_path):
            raise FileNotFoundError(f"Heuristic scores file not found: {score_file_path}")
        
        scores_df = pd.read_parquet(score_file_path)
        logger.info(f"Loaded heuristic scores for {len(scores_df)} wallets from {score_file_path}")
        
        return scores_df
    
    def transform_to_final_score(self, scores_df):
        """
        Transform raw scores to the final 0-100 scale.
        
        Args:
            scores_df (pd.DataFrame): DataFrame with raw scores
            
        Returns:
            pd.DataFrame: DataFrame with final scores
        """
        logger.info(f"Transforming scores using {self.scoring_method} method...")
        
        # Create a copy of the input DataFrame
        final_df = scores_df.copy()
        
        # Get raw scores
        raw_scores = final_df['raw_score'].values
        
        # Apply transformation based on method
        if self.scoring_method == 'percentile':
            # Calculate percentile rank (higher raw score = higher percentile)
            # If reverse is True, then lower raw score = higher percentile
            ascending = self.reverse
            final_df['final_score'] = final_df['raw_score'].rank(
                method='min', pct=True, ascending=ascending
            ) * 100
            
        elif self.scoring_method == 'min_max':
            # Apply min-max scaling to map to 0-100 range
            min_val = np.min(raw_scores)
            max_val = np.max(raw_scores)
            
            if max_val > min_val:
                if self.reverse:
                    # Lower raw score = higher final score
                    final_df['final_score'] = 100 * (1 - (raw_scores - min_val) / (max_val - min_val))
                else:
                    # Higher raw score = higher final score
                    final_df['final_score'] = 100 * (raw_scores - min_val) / (max_val - min_val)
            else:
                # All values are the same
                final_df['final_score'] = 50  # Assign a neutral score
                
        elif self.scoring_method == 'sigmoid':
            # Apply sigmoid transformation
            k = self.sigmoid_params.get('k', 1.0)  # Controls steepness
            x0 = self.sigmoid_params.get('x0', 0.0)  # Controls midpoint
            
            if self.reverse:
                # Lower raw score = higher final score
                final_df['final_score'] = 100 * (1 - 1 / (1 + np.exp(-k * (raw_scores - x0))))
            else:
                # Higher raw score = higher final score
                final_df['final_score'] = 100 * (1 / (1 + np.exp(-k * (raw_scores - x0))))
                
        else:
            raise ValueError(f"Unsupported scoring method: {self.scoring_method}")
        
        # Round to integers
        final_df['final_score'] = np.round(final_df['final_score']).astype(int)
        
        # Ensure scores are in the 0-100 range
        final_df['final_score'] = np.clip(final_df['final_score'], 0, 100)
        
        # Log score distribution
        score_distribution = final_df['final_score'].value_counts().sort_index()
        logger.info(f"Score distribution:\n{score_distribution}")
        
        return final_df
    
    def generate_output(self, final_df):
        """
        Generate output files with wallet scores.
        
        Args:
            final_df (pd.DataFrame): DataFrame with final scores
            
        Returns:
            pd.DataFrame: Top N wallets with scores
        """
        logger.info("Generating output files...")
        
        # Sort by final score (descending)
        sorted_df = final_df.sort_values('final_score', ascending=False)
        
        # Select top N wallets
        top_wallets = sorted_df.head(self.top_n_wallets)
        
        # Create output DataFrame
        output_df = pd.DataFrame({
            'wallet': top_wallets['wallet'],
            'score': top_wallets['final_score']
        })
        
        # Save to CSV
        output_path = os.path.join(self.results_dir, self.output_filename)
        output_df.to_csv(output_path, index=False)
        logger.info(f"Saved top {len(output_df)} wallets to {output_path}")
        
        # Save full results for analysis
        full_output_path = os.path.join(self.results_dir, "all_wallet_scores.parquet")
        final_df.to_parquet(full_output_path, index=False)
        logger.info(f"Saved all {len(final_df)} wallet scores to {full_output_path}")
        
        return output_df
    
    def analyze_top_and_bottom_wallets(self, final_df, features_df=None, n=5):
        """
        Analyze the top and bottom scoring wallets.
        
        Args:
            final_df (pd.DataFrame): DataFrame with final scores
            features_df (pd.DataFrame, optional): DataFrame with wallet features
            n (int, optional): Number of wallets to analyze
            
        Returns:
            dict: Analysis of top and bottom wallets
        """
        logger.info(f"Analyzing top and bottom {n} wallets...")
        
        # Sort by final score (descending)
        sorted_df = final_df.sort_values('final_score', ascending=False)
        
        # Get top and bottom N wallets
        top_wallets = sorted_df.head(n)
        bottom_wallets = sorted_df.tail(n)
        
        # Combine into one DataFrame for analysis
        analysis_wallets = pd.concat([
            top_wallets.assign(group='top'),
            bottom_wallets.assign(group='bottom')
        ])
        
        # If features are available, join with analysis wallets
        if features_df is not None:
            analysis_wallets = analysis_wallets.merge(
                features_df, on='wallet', how='left'
            )
        
        # Save the analysis
        analysis_path = os.path.join(self.results_dir, "wallet_analysis.csv")
        analysis_wallets.to_csv(analysis_path, index=False)
        logger.info(f"Saved wallet analysis to {analysis_path}")
        
        # Create a summary document
        summary_path = os.path.join(self.results_dir, "wallet_analysis_summary.md")
        
        with open(summary_path, 'w') as f:
            f.write("# Wallet Behavior Analysis\n\n")
            
            f.write("## Top-Scoring Wallets\n\n")
            for _, wallet in top_wallets.iterrows():
                f.write(f"### Wallet: {wallet['wallet'][:10]}... (Score: {wallet['final_score']})\n\n")
                if features_df is not None:
                    wallet_features = features_df[features_df['wallet'] == wallet['wallet']]
                    if not wallet_features.empty:
                        # Write key features
                        f.write("Key characteristics:\n")
                        f.write(f"- Transactions: {wallet_features['tx_count'].values[0]}\n")
                        f.write(f"- Account age: {wallet_features['wallet_age_days'].values[0]:.1f} days\n")
                        
                        if 'market_count' in wallet_features.columns:
                            f.write(f"- Markets: {wallet_features['market_count'].values[0]}\n")
                        
                        if 'liquidation_count_borrower' in wallet_features.columns:
                            f.write(f"- Liquidations: {wallet_features['liquidation_count_borrower'].values[0]}\n")
                        
                        if 'avg_ltv_alltime' in wallet_features.columns:
                            f.write(f"- Avg LTV: {wallet_features['avg_ltv_alltime'].values[0]:.2f}\n")
                        
                        if 'time_near_liquidation_pct' in wallet_features.columns:
                            f.write(f"- Time near liquidation: {wallet_features['time_near_liquidation_pct'].values[0]*100:.1f}%\n")
                        
                        f.write("\n")
            
            f.write("## Bottom-Scoring Wallets\n\n")
            for _, wallet in bottom_wallets.iterrows():
                f.write(f"### Wallet: {wallet['wallet'][:10]}... (Score: {wallet['final_score']})\n\n")
                if features_df is not None:
                    wallet_features = features_df[features_df['wallet'] == wallet['wallet']]
                    if not wallet_features.empty:
                        # Write key features
                        f.write("Key characteristics:\n")
                        f.write(f"- Transactions: {wallet_features['tx_count'].values[0]}\n")
                        f.write(f"- Account age: {wallet_features['wallet_age_days'].values[0]:.1f} days\n")
                        
                        if 'market_count' in wallet_features.columns:
                            f.write(f"- Markets: {wallet_features['market_count'].values[0]}\n")
                        
                        if 'liquidation_count_borrower' in wallet_features.columns:
                            f.write(f"- Liquidations: {wallet_features['liquidation_count_borrower'].values[0]}\n")
                        
                        if 'avg_ltv_alltime' in wallet_features.columns:
                            f.write(f"- Avg LTV: {wallet_features['avg_ltv_alltime'].values[0]:.2f}\n")
                        
                        if 'time_near_liquidation_pct' in wallet_features.columns:
                            f.write(f"- Time near liquidation: {wallet_features['time_near_liquidation_pct'].values[0]*100:.1f}%\n")
                        
                        f.write("\n")
            
            f.write("## Behavioral Patterns\n\n")
            f.write("### High-Scoring Wallet Patterns\n\n")
            f.write("High-scoring wallets typically exhibit these behavior patterns:\n")
            f.write("- Maintain healthy collateralization ratios\n")
            f.write("- Show consistent long-term participation\n")
            f.write("- Rarely or never experience liquidations\n")
            f.write("- Engage with multiple markets (diversification)\n")
            f.write("- Have high repayment ratios\n\n")
            
            f.write("### Low-Scoring Wallet Patterns\n\n")
            f.write("Low-scoring wallets typically exhibit these behavior patterns:\n")
            f.write("- Experience multiple liquidations\n")
            f.write("- Operate with high loan-to-value ratios\n")
            f.write("- Spend significant time near liquidation thresholds\n")
            f.write("- Show patterns of high leverage\n")
            f.write("- May exhibit unusual transaction patterns flagged by anomaly detection\n")
            
        logger.info(f"Saved wallet analysis summary to {summary_path}")
        
        return analysis_wallets
    
    def generate_scores(self):
        """
        Generate final wallet scores.
        
        Returns:
            pd.DataFrame: DataFrame with wallet scores
        """
        # Load heuristic scores
        heuristic_df = self.load_heuristic_scores()
        
        # Transform to final 0-100 scale
        final_df = self.transform_to_final_score(heuristic_df)
        
        # Generate output files
        output_df = self.generate_output(final_df)
        
        # Load features for analysis if available
        try:
            features_path = os.path.join(self.processed_data_dir, "wallet_features.parquet")
            features_df = pd.read_parquet(features_path)
            
            # Analyze top and bottom wallets
            analysis_df = self.analyze_top_and_bottom_wallets(final_df, features_df)
        except FileNotFoundError:
            logger.warning("Could not find features file for wallet analysis")
            # Analyze without features
            analysis_df = self.analyze_top_and_bottom_wallets(final_df)
        
        return output_df

def main():
    """
    Main function to generate wallet scores.
    """
    scorer = WalletScorer()
    output_df = scorer.generate_scores()
    
    print(f"Generated scores for top {len(output_df)} wallets")
    print("\nTop 10 wallets:")
    print(output_df.head(10))
    
if __name__ == "__main__":
    main() 