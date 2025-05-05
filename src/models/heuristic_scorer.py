"""
Heuristic scoring model for wallet credit scoring.
"""
import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from functools import lru_cache

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
        
        # Cache for optimizing data loading
        self._cached_data = {}
        
        # Ensure directory exists
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(config.CACHE_DIR, exist_ok=True)
    
    def load_features(self):
        """
        Load wallet features.
        
        Returns:
            pd.DataFrame: Wallet features
        """
        # Use cached data if available
        if 'features_df' in self._cached_data:
            logger.info(f"Using cached features for {len(self._cached_data['features_df'])} wallets")
            return self._cached_data['features_df']
        
        features_file_path = os.path.join(self.processed_data_dir, "wallet_features.parquet")
        # Check for cached file
        cache_file = os.path.join(config.CACHE_DIR, "features_cache.pkl")
        
        if os.path.exists(cache_file) and os.path.getmtime(cache_file) > os.path.getmtime(features_file_path):
            try:
                logger.info(f"Loading features from cache file {cache_file}")
                with open(cache_file, 'rb') as f:
                    features_df = pickle.load(f)
                logger.info(f"Loaded cached features for {len(features_df)} wallets")
                self._cached_data['features_df'] = features_df
                return features_df
            except Exception as e:
                logger.warning(f"Failed to load features from cache: {str(e)}")
        
        if not os.path.exists(features_file_path):
            raise FileNotFoundError(f"Features file not found: {features_file_path}")
        
        features_df = pd.read_parquet(features_file_path)
        logger.info(f"Loaded features for {len(features_df)} wallets from {features_file_path}")
        
        # Cache the loaded data
        self._cached_data['features_df'] = features_df
        
        # Save to cache file for future use
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(features_df, f)
            logger.info(f"Saved features to cache file {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save features to cache: {str(e)}")
        
        return features_df
    
    def load_anomaly_scores(self):
        """
        Load anomaly scores.
        
        Returns:
            pd.DataFrame: Anomaly scores
        """
        # Use cached data if available
        if 'anomaly_df' in self._cached_data:
            logger.info(f"Using cached anomaly scores for {len(self._cached_data['anomaly_df'])} wallets")
            return self._cached_data['anomaly_df']
        
        anomaly_file_path = os.path.join(self.processed_data_dir, "anomaly_scores.parquet")
        # Check for cached file
        cache_file = os.path.join(config.CACHE_DIR, "anomaly_cache.pkl")
        
        if os.path.exists(cache_file) and os.path.exists(anomaly_file_path) and os.path.getmtime(cache_file) > os.path.getmtime(anomaly_file_path):
            try:
                logger.info(f"Loading anomaly scores from cache file {cache_file}")
                with open(cache_file, 'rb') as f:
                    anomaly_df = pickle.load(f)
                logger.info(f"Loaded cached anomaly scores for {len(anomaly_df)} wallets")
                self._cached_data['anomaly_df'] = anomaly_df
                return anomaly_df
            except Exception as e:
                logger.warning(f"Failed to load anomaly scores from cache: {str(e)}")
        
        if not os.path.exists(anomaly_file_path):
            logger.warning(f"Anomaly scores file not found: {anomaly_file_path}")
            return None
        
        anomaly_df = pd.read_parquet(anomaly_file_path)
        logger.info(f"Loaded anomaly scores for {len(anomaly_df)} wallets from {anomaly_file_path}")
        
        # Cache the loaded data
        self._cached_data['anomaly_df'] = anomaly_df
        
        # Save to cache file for future use
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(anomaly_df, f)
            logger.info(f"Saved anomaly scores to cache file {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save anomaly scores to cache: {str(e)}")
        
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
            # Merge on wallet address - optimized with sorted merge
            if 'wallet' in scoring_df.columns and 'wallet' in anomaly_df.columns:
                # Sort both DataFrames by wallet for faster merge
                scoring_df = scoring_df.sort_values('wallet')
                anomaly_df_sorted = anomaly_df.sort_values('wallet')
                
                # Merge using sorted DataFrames
                scoring_df = pd.merge(
                    scoring_df,
                    anomaly_df_sorted[['wallet', 'anomaly_score']],
                    on='wallet',
                    how='left'
                )
            else:
                # Fall back to regular merge if column names are different
                scoring_df = scoring_df.merge(
                    anomaly_df[['wallet', 'anomaly_score']], 
                    on='wallet', 
                    how='left'
                )
            
            # Fill missing anomaly scores (if any)
            missing_anomaly_count = scoring_df['anomaly_score'].isna().sum()
            if missing_anomaly_count > 0:
                logger.warning(f"Found {missing_anomaly_count} wallets without anomaly scores")
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
    
    @lru_cache(maxsize=1024)
    def _normalize_value(self, value, min_val, max_val):
        """
        Normalize a value to [0, 1] range with caching.
        
        Args:
            value (float): Value to normalize
            min_val (float): Minimum value in the range
            max_val (float): Maximum value in the range
            
        Returns:
            float: Normalized value
        """
        if pd.isna(value):
            return 0.5  # Default value for NaN
            
        if max_val > min_val:
            return (value - min_val) / (max_val - min_val)
        else:
            return 0.5  # Default for constant features
    
    def calculate_weighted_score(self, scoring_df, optimize=True):
        """
        Calculate a weighted score based on features.
        
        Args:
            scoring_df (pd.DataFrame): Features for scoring
            optimize (bool): Whether to use optimized algorithms
            
        Returns:
            pd.DataFrame: DataFrame with calculated scores
        """
        logger.info("Calculating weighted scores...")
        
        # Create a copy of the input DataFrame
        scores_df = scoring_df.copy()
        
        # Filter to only use weights for available features
        available_weights = {
            feature: weight for feature, weight in self.feature_weights.items()
            if feature in scores_df.columns
        }
        
        # Log feature stats
        for feature, weight in available_weights.items():
            logger.info(f"Feature: {feature}, Weight: {weight}")
            logger.info(f"  Min: {scores_df[feature].min()}, Max: {scores_df[feature].max()}, Mean: {scores_df[feature].mean()}")
        
        if optimize:
            # Optimized vectorized calculations - much faster than row-by-row
            
            # Initialize raw score column
            scores_df['raw_score'] = 0
            scores_df['feature_contributions'] = None
            
            # Precompute feature min/max values for normalization
            feature_mins = {feature: scores_df[feature].min() for feature in available_weights}
            feature_maxs = {feature: scores_df[feature].max() for feature in available_weights}
            
            # Calculate normalized features and contributions in a vectorized way
            for feature, weight in tqdm(available_weights.items(), desc="Calculating feature scores"):
                # Skip if the feature is all missing
                if scores_df[feature].isna().all():
                    continue
                
                # Create a normalized feature column
                min_val = feature_mins[feature]
                max_val = feature_maxs[feature]
                
                if max_val > min_val:
                    # Normalize the feature to [0, 1] range - vectorized
                    normalized_feature = (scores_df[feature] - min_val) / (max_val - min_val)
                    
                    # For negative weights, invert the feature
                    if weight < 0:
                        normalized_feature = 1 - normalized_feature
                        weight_abs = abs(weight)
                    else:
                        weight_abs = weight
                    
                    # Calculate contribution and add to raw score
                    contribution = normalized_feature * weight_abs
                    scores_df['raw_score'] += contribution
                    
                    # Store feature contribution for transparency
                    scores_df[f'contribution_{feature}'] = contribution
            
            # Store contribution information
            scores_df['feature_contributions'] = scores_df.apply(
                lambda row: {f: row[f'contribution_{f}'] for f in available_weights.keys() 
                            if f'contribution_{f}' in row}, 
                axis=1
            )
            
            # Clean up temporary columns
            for feature in available_weights.keys():
                if f'contribution_{feature}' in scores_df.columns:
                    scores_df = scores_df.drop(columns=[f'contribution_{feature}'])
            
        else:
            # Original implementation (slower but works row by row)
            # Initialize score columns
            scores_df['raw_score'] = 0
            scores_df['feature_contributions'] = None
            
            # Calculate feature min/max for normalization
            feature_mins = {}
            feature_maxs = {}
            for feature in available_weights.keys():
                feature_mins[feature] = scores_df[feature].min() 
                feature_maxs[feature] = scores_df[feature].max()
            
            # Process each wallet
            for idx, row in tqdm(scores_df.iterrows(), total=len(scores_df), desc="Calculating wallet scores"):
                raw_score = 0
                contributions = {}
                
                for feature, weight in available_weights.items():
                    if feature not in row:
                        continue
                        
                    feature_val = row[feature]
                    if pd.isna(feature_val):
                        continue
                    
                    # Normalize the feature to [0, 1] range
                    min_val = feature_mins[feature]
                    max_val = feature_maxs[feature]
                    
                    if max_val > min_val:
                        norm_val = (feature_val - min_val) / (max_val - min_val)
                    else:
                        norm_val = 0.5  # Default for constant features
                    
                    # For negative weights, invert the normalized value
                    if weight < 0:
                        norm_val = 1 - norm_val
                        weight_abs = abs(weight)
                    else:
                        weight_abs = weight
                    
                    # Calculate contribution to raw score
                    contribution = norm_val * weight_abs
                    raw_score += contribution
                    
                    # Store contribution
                    contributions[feature] = contribution
                
                # Update scores DataFrame
                scores_df.at[idx, 'raw_score'] = raw_score
                scores_df.at[idx, 'feature_contributions'] = contributions
        
        # Normalize the raw score by the sum of weights
        sum_weights = sum(abs(w) for w in available_weights.values())
        if sum_weights > 0:
            scores_df['raw_score'] = scores_df['raw_score'] / sum_weights
        
        # Log score stats
        logger.info(f"Raw score - Min: {scores_df['raw_score'].min()}, Max: {scores_df['raw_score'].max()}, Mean: {scores_df['raw_score'].mean()}")
        
        return scores_df
    
    def apply_score_adjustments(self, scores_df, optimize=True):
        """
        Apply adjustments to the raw scores based on specific rules.
        
        Args:
            scores_df (pd.DataFrame): DataFrame with raw scores
            optimize (bool): Whether to use optimized algorithms
            
        Returns:
            pd.DataFrame: DataFrame with adjusted scores
        """
        logger.info("Applying score adjustments...")
        
        # Create a copy of the input DataFrame
        adjusted_df = scores_df.copy()
        
        if optimize:
            # Vectorized adjustments - more efficient than row-by-row
            
            # 1. Heavy penalty for wallets with liquidations - vectorized
            if 'liquidation_count_borrower' in adjusted_df.columns:
                liquidated_wallets = adjusted_df['liquidation_count_borrower'] > 0
                liquidation_count = adjusted_df.loc[liquidated_wallets, 'liquidation_count_borrower']
                
                if liquidated_wallets.any():
                    logger.info(f"Applying liquidation penalty to {liquidated_wallets.sum()} wallets (vectorized)")
                    
                    # Calculate penalties - vectorized operations
                    liquidation_penalty = np.minimum(liquidation_count * 0.1, 0.5)
                    adjusted_df.loc[liquidated_wallets, 'raw_score'] *= (1 - liquidation_penalty)
                    
                    # Store adjustment reason
                    if 'adjustments' not in adjusted_df.columns:
                        adjusted_df['adjustments'] = pd.Series([{} for _ in range(len(adjusted_df))])
                    
                    # Record the adjustment for each wallet
                    for idx in adjusted_df.index[liquidated_wallets]:
                        if not isinstance(adjusted_df.at[idx, 'adjustments'], dict):
                            adjusted_df.at[idx, 'adjustments'] = {}
                        penalty = liquidation_penalty.loc[idx] if idx in liquidation_penalty.index else 0
                        adjusted_df.at[idx, 'adjustments']['liquidation_penalty'] = -penalty
            
            # 2. Reward for wallets with good repayment behavior - vectorized
            if all(col in adjusted_df.columns for col in ['repay_ratio', 'liquidation_count_borrower']):
                good_repayers = (adjusted_df['repay_ratio'] > 0.9) & (adjusted_df['liquidation_count_borrower'] == 0)
                
                if good_repayers.any():
                    logger.info(f"Applying good repayer bonus to {good_repayers.sum()} wallets (vectorized)")
                    adjusted_df.loc[good_repayers, 'raw_score'] *= 1.1  # 10% bonus
                    
                    # Record the adjustment for each wallet
                    for idx in adjusted_df.index[good_repayers]:
                        if 'adjustments' not in adjusted_df.columns:
                            adjusted_df['adjustments'] = pd.Series([{} for _ in range(len(adjusted_df))])
                        if not isinstance(adjusted_df.at[idx, 'adjustments'], dict):
                            adjusted_df.at[idx, 'adjustments'] = {}
                        adjusted_df.at[idx, 'adjustments']['good_repayer_bonus'] = 0.1
            
            # 3. Penalty for extremely high loan-to-value ratios - vectorized
            if 'max_ltv_alltime' in adjusted_df.columns:
                high_ltv_wallets = adjusted_df['max_ltv_alltime'] > 0.9
                
                if high_ltv_wallets.any():
                    logger.info(f"Applying high LTV penalty to {high_ltv_wallets.sum()} wallets (vectorized)")
                    adjusted_df.loc[high_ltv_wallets, 'raw_score'] *= 0.9  # 10% reduction
                    
                    # Record the adjustment for each wallet
                    for idx in adjusted_df.index[high_ltv_wallets]:
                        if 'adjustments' not in adjusted_df.columns:
                            adjusted_df['adjustments'] = pd.Series([{} for _ in range(len(adjusted_df))])
                        if not isinstance(adjusted_df.at[idx, 'adjustments'], dict):
                            adjusted_df.at[idx, 'adjustments'] = {}
                        adjusted_df.at[idx, 'adjustments']['high_ltv_penalty'] = -0.1
            
            # 4. New: Consistency bonus - reward wallets with long history and low variability
            if all(col in adjusted_df.columns for col in ['account_age_days', 'activity_consistency_stddev']):
                consistent_users = (adjusted_df['account_age_days'] > 180) & (adjusted_df['activity_consistency_stddev'] < 5)
                
                if consistent_users.any():
                    logger.info(f"Applying consistency bonus to {consistent_users.sum()} wallets (vectorized)")
                    adjusted_df.loc[consistent_users, 'raw_score'] *= 1.05  # 5% bonus
                    
                    # Record the adjustment for each wallet
                    for idx in adjusted_df.index[consistent_users]:
                        if 'adjustments' not in adjusted_df.columns:
                            adjusted_df['adjustments'] = pd.Series([{} for _ in range(len(adjusted_df))])
                        if not isinstance(adjusted_df.at[idx, 'adjustments'], dict):
                            adjusted_df.at[idx, 'adjustments'] = {}
                        adjusted_df.at[idx, 'adjustments']['consistency_bonus'] = 0.05
            
            # 5. New: Inactivity penalty - penalize wallets with long periods of inactivity
            if 'days_since_last_tx' in adjusted_df.columns:
                inactive_wallets = adjusted_df['days_since_last_tx'] > 90  # Inactive for > 90 days
                
                if inactive_wallets.any():
                    logger.info(f"Applying inactivity penalty to {inactive_wallets.sum()} wallets (vectorized)")
                    
                    # Calculate penalty based on inactivity duration - 5% penalty for each 30 days of inactivity, max 25%
                    inactivity_days = adjusted_df.loc[inactive_wallets, 'days_since_last_tx']
                    inactivity_penalty = np.minimum(0.05 * (inactivity_days / 30), 0.25)
                    
                    adjusted_df.loc[inactive_wallets, 'raw_score'] *= (1 - inactivity_penalty)
                    
                    # Record the adjustment for each wallet
                    for idx in adjusted_df.index[inactive_wallets]:
                        if 'adjustments' not in adjusted_df.columns:
                            adjusted_df['adjustments'] = pd.Series([{} for _ in range(len(adjusted_df))])
                        if not isinstance(adjusted_df.at[idx, 'adjustments'], dict):
                            adjusted_df.at[idx, 'adjustments'] = {}
                        penalty = inactivity_penalty.loc[idx] if idx in inactivity_penalty.index else 0
                        adjusted_df.at[idx, 'adjustments']['inactivity_penalty'] = -penalty
        
        else:
            # Original implementation
            # Apply adjustments based on business rules
            
            # 1. Heavy penalty for wallets with liquidations
            if 'liquidation_count_borrower' in adjusted_df.columns:
                liquidated_wallets = adjusted_df['liquidation_count_borrower'] > 0
                if liquidated_wallets.any():
                    logger.info(f"Applying liquidation penalty to {liquidated_wallets.sum()} wallets")
                    # Penalize proportionally to the number of liquidations, but cap the reduction
                    liquidation_penalty = np.minimum(
                        adjusted_df.loc[liquidated_wallets, 'liquidation_count_borrower'] * 0.1,
                        0.5  # Maximum 50% reduction
                    )
                    adjusted_df.loc[liquidated_wallets, 'raw_score'] *= (1 - liquidation_penalty)
                    
            # 2. Bonus for consistent repayments without liquidations
            if all(col in adjusted_df.columns for col in ['repay_ratio', 'liquidation_count_borrower']):
                good_repayers = (adjusted_df['repay_ratio'] > 0.9) & (adjusted_df['liquidation_count_borrower'] == 0)
                if good_repayers.any():
                    logger.info(f"Applying good repayer bonus to {good_repayers.sum()} wallets")
                    adjusted_df.loc[good_repayers, 'raw_score'] *= 1.1  # 10% bonus
                    
            # 3. Penalty for extremely high loan-to-value ratios
            if 'max_ltv_alltime' in adjusted_df.columns:
                high_ltv_wallets = adjusted_df['max_ltv_alltime'] > 0.9
                if high_ltv_wallets.any():
                    logger.info(f"Applying high LTV penalty to {high_ltv_wallets.sum()} wallets")
                    adjusted_df.loc[high_ltv_wallets, 'raw_score'] *= 0.9  # 10% reduction
        
        # Log adjustments
        logger.info(f"After adjustments - Raw score min: {adjusted_df['raw_score'].min()}, max: {adjusted_df['raw_score'].max()}, mean: {adjusted_df['raw_score'].mean()}")
        
        return adjusted_df
    
    def score_wallets(self, optimize=True):
        """
        Score wallets using the heuristic model.
        
        Args:
            optimize (bool): Whether to use optimized algorithms
        
        Returns:
            pd.DataFrame: DataFrame with scored wallets
        """
        logger.info(f"Scoring wallets using heuristic model with optimization={'enabled' if optimize else 'disabled'}...")
        
        # Check for cached scores to avoid reprocessing
        scores_cache_file = os.path.join(config.CACHE_DIR, "wallet_scores.pkl")
        if optimize and os.path.exists(scores_cache_file):
            # Check if any input data is newer than the scores file
            features_file = os.path.join(self.processed_data_dir, "wallet_features.parquet")
            anomaly_file = os.path.join(self.processed_data_dir, "anomaly_scores.parquet")
            
            scores_mtime = os.path.getmtime(scores_cache_file)
            features_is_newer = os.path.exists(features_file) and os.path.getmtime(features_file) > scores_mtime
            anomaly_is_newer = os.path.exists(anomaly_file) and os.path.getmtime(anomaly_file) > scores_mtime
            
            if not features_is_newer and not anomaly_is_newer:
                logger.info(f"Loading cached wallet scores from {scores_cache_file}")
                try:
                    with open(scores_cache_file, 'rb') as f:
                        wallet_scores = pickle.load(f)
                    logger.info(f"Loaded {len(wallet_scores)} wallet scores from cache")
                    return wallet_scores
                except Exception as e:
                    logger.warning(f"Error loading cached scores: {str(e)}")
        
        # Load input data
        features_df = self.load_features()
        anomaly_df = self.load_anomaly_scores()
        
        # Prepare features for scoring
        scoring_df = self.prepare_features(features_df, anomaly_df)
        
        # Calculate weighted scores
        scores_df = self.calculate_weighted_score(scoring_df, optimize=optimize)
        
        # Apply adjustments
        final_df = self.apply_score_adjustments(scores_df, optimize=optimize)
        
        # Save wallet scores for later use
        output_file = os.path.join(self.processed_data_dir, "wallet_scores.parquet")
        final_df.to_parquet(output_file, index=False)
        logger.info(f"Saved {len(final_df)} wallet scores to {output_file}")
        
        # Cache results if optimization is enabled
        if optimize:
            try:
                with open(scores_cache_file, 'wb') as f:
                    pickle.dump(final_df, f)
                logger.info(f"Cached {len(final_df)} wallet scores to {scores_cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache scores: {str(e)}")
        
        return final_df

def main():
    """
    Run heuristic scoring as a standalone script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Score wallets using heuristic model")
    parser.add_argument('--no-optimize', action='store_true', help='Disable algorithmic optimizations')
    args = parser.parse_args()
    
    heuristic_scorer = HeuristicScorer()
    heuristic_scorer.score_wallets(optimize=not args.no_optimize)

if __name__ == "__main__":
    main() 