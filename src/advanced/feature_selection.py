"""
Advanced feature selection techniques for Compound V2 wallet scoring.
"""
import os
import logging
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression, RFE
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import src.config as config

logger = logging.getLogger(__name__)

class FeatureSelector:
    """
    Class for selecting optimal features for wallet scoring.
    """
    
    def __init__(self, processed_data_dir=None, results_dir=None):
        """
        Initialize the FeatureSelector.
        
        Args:
            processed_data_dir (str, optional): Directory containing processed data
            results_dir (str, optional): Directory to save results
        """
        self.processed_data_dir = processed_data_dir or config.PROCESSED_DATA_DIR
        self.results_dir = results_dir or config.RESULTS_DIR
        
        # Create analysis directory
        self.analysis_dir = os.path.join(self.results_dir, "feature_analysis")
        os.makedirs(self.analysis_dir, exist_ok=True)
    
    def load_data(self):
        """
        Load feature data and target values.
        
        Returns:
            tuple: (X, y) feature matrix and target values
        """
        # Load feature data
        features_path = os.path.join(self.processed_data_dir, "wallet_features.parquet")
        scores_path = os.path.join(self.processed_data_dir, "wallet_scores.parquet")
        
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        if not os.path.exists(scores_path):
            raise FileNotFoundError(f"Scores file not found: {scores_path}")
        
        features_df = pd.read_parquet(features_path)
        scores_df = pd.read_parquet(scores_path)
        
        # Merge features with scores
        df = features_df.merge(scores_df[['wallet', 'raw_score']], on='wallet', how='inner')
        logger.info(f"Loaded and merged data for {len(df)} wallets")
        
        # Separate features from target
        X = df.drop(['wallet', 'raw_score'], axis=1)
        y = df['raw_score']
        
        return X, y
    
    def correlation_analysis(self, X, y, threshold=0.1):
        """
        Perform correlation analysis between features and target.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target values
            threshold (float): Minimum correlation threshold
            
        Returns:
            pd.DataFrame: Features sorted by correlation strength
        """
        logger.info("Performing correlation analysis...")
        
        # Calculate correlation with target
        correlations = pd.DataFrame()
        correlations['feature'] = X.columns
        correlations['correlation'] = [np.corrcoef(X[col].fillna(0), y)[0, 1] for col in X.columns]
        correlations['abs_correlation'] = correlations['correlation'].abs()
        
        # Sort by absolute correlation
        correlations = correlations.sort_values('abs_correlation', ascending=False)
        
        # Filter by threshold
        strong_correlations = correlations[correlations['abs_correlation'] >= threshold]
        logger.info(f"Found {len(strong_correlations)} features with correlation >= {threshold}")
        
        # Visualize top correlations
        plt.figure(figsize=(10, 8))
        plt.title('Feature Correlation with Target Score')
        sns.barplot(
            x='correlation', 
            y='feature', 
            data=correlations.head(20),
            palette='viridis'
        )
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'feature_correlations.png'))
        plt.close()
        
        return correlations
    
    def mutual_information_analysis(self, X, y, k=15):
        """
        Perform mutual information analysis to identify nonlinear relationships.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target values
            k (int): Number of top features to select
            
        Returns:
            pd.DataFrame: Features sorted by mutual information
        """
        logger.info("Performing mutual information analysis...")
        
        # Fill NaN values (required for mutual_info_regression)
        X_filled = X.fillna(0)
        
        # Calculate mutual information
        mi = mutual_info_regression(X_filled, y)
        mi_scores = pd.DataFrame({
            'feature': X.columns,
            'mutual_info': mi
        })
        
        # Sort by mutual information score
        mi_scores = mi_scores.sort_values('mutual_info', ascending=False)
        
        # Visualize top features by mutual information
        plt.figure(figsize=(10, 8))
        plt.title('Feature Mutual Information with Target Score')
        sns.barplot(
            x='mutual_info', 
            y='feature', 
            data=mi_scores.head(20),
            palette='magma'
        )
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'mutual_information.png'))
        plt.close()
        
        # Select top k features
        selected_features = mi_scores.head(k)['feature'].tolist()
        logger.info(f"Selected top {k} features using mutual information")
        
        return mi_scores
    
    def random_forest_importance(self, X, y):
        """
        Use Random Forest to calculate feature importance.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target values
            
        Returns:
            pd.DataFrame: Features sorted by importance
        """
        logger.info("Calculating Random Forest feature importance...")
        
        # Fill NaN values
        X_filled = X.fillna(0)
        
        # Train a Random Forest model
        rf = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42, 
            n_jobs=-1
        )
        rf.fit(X_filled, y)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        })
        
        # Sort by importance
        importance = importance.sort_values('importance', ascending=False)
        
        # Visualize feature importance
        plt.figure(figsize=(10, 8))
        plt.title('Random Forest Feature Importance')
        sns.barplot(
            x='importance', 
            y='feature', 
            data=importance.head(20),
            palette='crest'
        )
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'rf_importance.png'))
        plt.close()
        
        return importance
    
    def feature_correlation_matrix(self, X):
        """
        Calculate and visualize feature correlation matrix to identify redundancy.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            pd.DataFrame: Feature correlation matrix
        """
        logger.info("Calculating feature correlation matrix...")
        
        # Calculate correlation matrix
        corr_matrix = X.fillna(0).corr().abs()
        
        # Visualize correlation matrix
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, 
            mask=mask, 
            cmap='coolwarm',
            vmax=1.0, 
            vmin=0.0, 
            center=0.5,
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5},
            annot=False
        )
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'correlation_matrix.png'))
        plt.close()
        
        # Identify highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.8:  # Threshold for high correlation
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        # Sort by correlation strength
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return corr_matrix, high_corr_pairs
    
    def recursive_feature_elimination(self, X, y, n_features=10):
        """
        Perform Recursive Feature Elimination (RFE) to select features.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target values
            n_features (int): Number of features to select
            
        Returns:
            list: Selected feature names
        """
        logger.info(f"Performing Recursive Feature Elimination to select {n_features} features...")
        
        # Fill NaN values
        X_filled = X.fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filled)
        
        # Create base estimator
        estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Create RFE selector
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        selector = selector.fit(X_scaled, y)
        
        # Get selected features
        selected_features = [feature for feature, selected in zip(X.columns, selector.support_) if selected]
        
        logger.info(f"Selected features: {selected_features}")
        
        return selected_features
    
    def select_optimal_features(self, method='combined', n_features=15):
        """
        Select optimal features using the specified method.
        
        Args:
            method (str): Feature selection method ('correlation', 'mutual_info', 'rf_importance', 'rfe', 'combined')
            n_features (int): Number of features to select
            
        Returns:
            list: Selected feature names
        """
        logger.info(f"Selecting optimal features using {method} method...")
        
        # Load data
        X, y = self.load_data()
        
        # Dictionary to hold all results
        results = {}
        
        # Perform correlation analysis
        correlations = self.correlation_analysis(X, y)
        results['correlation'] = correlations.head(n_features)['feature'].tolist()
        
        # Perform mutual information analysis
        mi_scores = self.mutual_information_analysis(X, y, k=n_features)
        results['mutual_info'] = mi_scores.head(n_features)['feature'].tolist()
        
        # Calculate Random Forest importance
        rf_importance = self.random_forest_importance(X, y)
        results['rf_importance'] = rf_importance.head(n_features)['feature'].tolist()
        
        # Perform Recursive Feature Elimination
        rfe_features = self.recursive_feature_elimination(X, y, n_features=n_features)
        results['rfe'] = rfe_features
        
        # Analyze feature correlation matrix for redundancy
        _, high_corr_pairs = self.feature_correlation_matrix(X)
        
        # Choose final features based on specified method
        if method == 'correlation':
            selected_features = results['correlation']
        elif method == 'mutual_info':
            selected_features = results['mutual_info']
        elif method == 'rf_importance':
            selected_features = results['rf_importance']
        elif method == 'rfe':
            selected_features = results['rfe']
        elif method == 'combined':
            # Count feature appearance in all methods
            feature_votes = {}
            for feature_list in results.values():
                for feature in feature_list:
                    if feature not in feature_votes:
                        feature_votes[feature] = 0
                    feature_votes[feature] += 1
            
            # Sort by vote count
            sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
            
            # Select top features
            selected_features = [feature for feature, _ in sorted_features[:n_features]]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Save selected features
        selected_df = pd.DataFrame({
            'feature': selected_features,
            'selected_by': method
        })
        selected_df.to_csv(os.path.join(self.analysis_dir, 'selected_features.csv'), index=False)
        
        # Create feature selection summary
        with open(os.path.join(self.analysis_dir, 'feature_selection_summary.md'), 'w') as f:
            f.write("# Feature Selection Summary\n\n")
            f.write(f"## Method: {method}\n\n")
            f.write(f"Selected {len(selected_features)} features from {X.shape[1]} original features.\n\n")
            
            f.write("## Selected Features\n\n")
            for i, feature in enumerate(selected_features):
                f.write(f"{i+1}. {feature}\n")
            
            f.write("\n## Features by Selection Method\n\n")
            for method_name, feature_list in results.items():
                f.write(f"### {method_name}\n\n")
                for i, feature in enumerate(feature_list):
                    f.write(f"{i+1}. {feature}\n")
                f.write("\n")
            
            f.write("## Highly Correlated Feature Pairs\n\n")
            for feature1, feature2, corr in high_corr_pairs[:20]:
                f.write(f"- {feature1} & {feature2}: {corr:.3f}\n")
        
        logger.info(f"Selected {len(selected_features)} features using {method} method")
        
        return selected_features

def main():
    """
    Run feature selection as a standalone script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Select optimal features for wallet scoring")
    parser.add_argument('--method', type=str, default='combined', 
                        choices=['correlation', 'mutual_info', 'rf_importance', 'rfe', 'combined'],
                        help='Feature selection method')
    parser.add_argument('--n-features', type=int, default=15, 
                        help='Number of features to select')
    
    args = parser.parse_args()
    
    selector = FeatureSelector()
    selector.select_optimal_features(method=args.method, n_features=args.n_features)

if __name__ == "__main__":
    main() 