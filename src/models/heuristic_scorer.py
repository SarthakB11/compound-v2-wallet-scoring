"""
Heuristic scoring model for wallet credit scoring.
"""
import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

import src.config as config

logger = logging.getLogger(__name__)

class HeuristicScorer:
    """
    Class for scoring wallets using a heuristic approach.
    """
    
    def __init__(self, processed_data_dir=None):
        """
        Initialize the HeuristicScorer.
        
        Args:
            processed_data_dir (str, optional): Directory containing processed data
        """
        self.processed_data_dir = processed_data_dir or config.PROCESSED_DATA_DIR
        
        # Feature weights for scoring
        self.feature_weights = config.FEATURE_WEIGHTS
        
        # Ensure directory exists
        os.makedirs(self.processed_data_dir, exist_ok=True)
    
    def load_features(self):
        """
        Load wallet features.
        
        Returns:
            pd.DataFrame: Wallet features
        """
        features_file_path = os.path.join(self.processed_data_dir, "wallet_features.parquet")
        if not os.path.exists(features_file_path):
            raise FileNotFoundError(f"Features file not found: {features_file_path}")
        
        features_df = pd.read_parquet(features_file_path)
        logger.info(f"Loaded features for {len(features_df)} wallets from {features_file_path}")
        
        return features_df
    
    def load_anomaly_scores(self):
        """
        Load anomaly scores.
        
        Returns:
            pd.DataFrame: Anomaly scores
        """
        anomaly_file_path = os.path.join(self.processed_data_dir, "anomaly_scores.parquet")
        if not os.path.exists(anomaly_file_path):
            logger.warning(f"Anomaly scores file not found: {anomaly_file_path}")
            return None
        
        anomaly_df = pd.read_parquet(anomaly_file_path)
        logger.info(f"Loaded anomaly scores for {len(anomaly_df)} wallets from {anomaly_file_path}")
        
        return anomaly_df
    
    def prepare_features(self, features_df, anomaly_df=None):
        """
        Prepare features for scoring.
        
        Args:
            features_df (pd.DataFrame): Wallet features
            anomaly_df (pd.DataFrame, optional): Anomaly scores
            
        Returns:
            pd.DataFrame: Combined features for scoring
        """
        logger.info("Preparing features for scoring...")
        
        # Create a copy of the features DataFrame
        scoring_df = features_df.copy()
        
        # Add anomaly scores if available
        if anomaly_df is not None:
            # Merge on wallet address
            scoring_df = scoring_df.merge(
                anomaly_df[['wallet', 'anomaly_score']], 
                on='wallet', 
                how='left'
            )
            
            # Fill missing anomaly scores (if any)
            if scoring_df['anomaly_score'].isna().any():
                logger.warning(f"Found {scoring_df['anomaly_score'].isna().sum()} wallets without anomaly scores")
                scoring_df['anomaly_score'] = scoring_df['anomaly_score'].fillna(0.5)  # Neutral score
        else:
            # Add a default anomaly score column
            scoring_df['anomaly_score'] = 0.5  # Neutral score
        
        # Check which weighted features are actually available
        missing_features = [feature for feature in self.feature_weights.keys() 
                           if feature not in scoring_df.columns]
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features for scoring: {missing_features}")
        
        return scoring_df
    
    def calculate_weighted_score(self, scoring_df):
        """
        Calculate a weighted score based on features.
        
        Args:
            scoring_df (pd.DataFrame): Features for scoring
            
        Returns:
            pd.DataFrame: DataFrame with calculated scores
        """
        logger.info("Calculating weighted scores...")
        
        # Create a copy of the input DataFrame
        scores_df = scoring_df.copy()
        
        # Calculate raw score based on weighted features
        scores_df['raw_score'] = 0
        
        # Filter to only use weights for available features
        available_weights = {
            feature: weight for feature, weight in self.feature_weights.items()
            if feature in scores_df.columns
        }
        
        # Calculate score components for each feature
        for feature, weight in tqdm(available_weights.items(), desc="Calculating feature scores"):
            # Skip if the feature has no data
            if feature not in scores_df.columns:
                continue
                
            # Log feature stats
            logger.info(f"Feature: {feature}, Weight: {weight}")
            logger.info(f"  Min: {scores_df[feature].min()}, Max: {scores_df[feature].max()}, Mean: {scores_df[feature].mean()}")
            
            # Create a normalized feature column
            # This helps ensure comparable scales across features
            norm_col = f"{feature}_norm"
            
            # Normalize the feature to [0, 1] range
            if scores_df[feature].min() != scores_df[feature].max():
                scores_df[norm_col] = (scores_df[feature] - scores_df[feature].min()) / (
                    scores_df[feature].max() - scores_df[feature].min()
                )
            else:
                scores_df[norm_col] = 0  # If all values are identical
            
            # For some features, lower values are better
            # If the weight is negative, we want to invert the feature
            if weight < 0:
                scores_df[norm_col] = 1 - scores_df[norm_col]
                weight = abs(weight)  # Use positive weight value for calculation
            
            # Calculate this feature's contribution to the raw score
            scores_df['raw_score'] += scores_df[norm_col] * weight
            
            # Remove temporary normalized column
            scores_df = scores_df.drop(columns=[norm_col])
        
        # Normalize the raw score by the sum of weights
        sum_weights = sum(abs(w) for w in available_weights.values())
        if sum_weights > 0:
            scores_df['raw_score'] = scores_df['raw_score'] / sum_weights
        
        # Log score stats
        logger.info(f"Raw score - Min: {scores_df['raw_score'].min()}, Max: {scores_df['raw_score'].max()}, Mean: {scores_df['raw_score'].mean()}")
        
        return scores_df
    
    def apply_score_adjustments(self, scores_df):
        """
        Apply adjustments to the raw scores based on specific rules.
        
        Args:
            scores_df (pd.DataFrame): DataFrame with raw scores
            
        Returns:
            pd.DataFrame: DataFrame with adjusted scores
        """
        logger.info("Applying score adjustments...")
        
        # Create a copy of the input DataFrame
        adjusted_df = scores_df.copy()
        
        # Apply adjustments based on business rules
        
        # 1. Heavy penalty for wallets with liquidations
        if 'liquidation_count_borrower' in adjusted_df.columns:
            liquidated_wallets = adjusted_df['liquidation_count_borrower'] > 0
            if liquidated_wallets.any():
                logger.info(f"Applying liquidation penalty to {liquidated_wallets.sum()} wallets")
                # Penalize proportionally to the number of liquidations, but cap the reduction
                liquidation_penalty = np.minimum(
                    adjusted_df['liquidation_count_borrower'] * 0.1,  # 10% per liquidation
                    0.5  # Maximum 50% reduction
                )
                adjusted_df.loc[liquidated_wallets, 'raw_score'] *= (1 - liquidation_penalty)
        
        # 2. Penalty for wallets with high time spent near liquidation
        if 'time_near_liquidation_pct' in adjusted_df.columns:
            high_risk_wallets = adjusted_df['time_near_liquidation_pct'] > 0.5  # >50% time near liquidation
            if high_risk_wallets.any():
                logger.info(f"Applying high-risk penalty to {high_risk_wallets.sum()} wallets")
                high_risk_penalty = adjusted_df['time_near_liquidation_pct'] * 0.3  # Up to 30% reduction
                adjusted_df.loc[high_risk_wallets, 'raw_score'] *= (1 - high_risk_penalty)
        
        # 3. Bonus for long-term consistent users
        if 'wallet_age_days' in adjusted_df.columns and 'tx_count' in adjusted_df.columns:
            # Identify wallets with both long age (>180 days) and consistent activity (>10 transactions)
            long_term_users = (adjusted_df['wallet_age_days'] > 180) & (adjusted_df['tx_count'] > 10)
            if long_term_users.any():
                logger.info(f"Applying long-term user bonus to {long_term_users.sum()} wallets")
                # Bonus up to 10%
                long_term_bonus = 0.1
                adjusted_df.loc[long_term_users, 'raw_score'] *= (1 + long_term_bonus)
        
        # 4. Cap the adjusted score at 1.0
        adjusted_df['raw_score'] = np.minimum(adjusted_df['raw_score'], 1.0)
        
        # Log adjusted score stats
        logger.info(f"Adjusted score - Min: {adjusted_df['raw_score'].min()}, Max: {adjusted_df['raw_score'].max()}, Mean: {adjusted_df['raw_score'].mean()}")
        
        return adjusted_df
    
    def score_wallets(self):
        """
        Score wallets using the heuristic model.
        
        Returns:
            pd.DataFrame: Wallet scores
        """
        # Load features
        features_df = self.load_features()
        
        # Load anomaly scores if available
        anomaly_df = self.load_anomaly_scores()
        
        # Prepare features for scoring
        scoring_df = self.prepare_features(features_df, anomaly_df)
        
        # Calculate weighted scores
        scores_df = self.calculate_weighted_score(scoring_df)
        
        # Apply score adjustments
        adjusted_df = self.apply_score_adjustments(scores_df)
        
        # Keep only necessary columns for the output
        output_columns = ['wallet', 'raw_score']
        output_df = adjusted_df[output_columns].copy()
        
        # Save the scores
        scores_path = os.path.join(self.processed_data_dir, "heuristic_scores.parquet")
        output_df.to_parquet(scores_path, index=False)
        logger.info(f"Saved heuristic scores for {len(output_df)} wallets to {scores_path}")
        
        return output_df

def main():
    """
    Main function to run heuristic scoring.
    """
    scorer = HeuristicScorer()
    scores_df = scorer.score_wallets()
    
    print(f"Scored {len(scores_df)} wallets")
    print("\nTop 5 wallets by score:")
    print(scores_df.sort_values('raw_score', ascending=False).head())
    print("\nBottom 5 wallets by score:")
    print(scores_df.sort_values('raw_score', ascending=True).head())
    
if __name__ == "__main__":
    main() 