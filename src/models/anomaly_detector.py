"""
Anomaly detection model for identifying unusual wallet behavior.
"""
import os
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

import src.config as config

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Class for detecting anomalous wallet behavior.
    """
    
    def __init__(self, processed_data_dir=None):
        """
        Initialize the AnomalyDetector.
        
        Args:
            processed_data_dir (str, optional): Directory containing processed data
        """
        self.processed_data_dir = processed_data_dir or config.PROCESSED_DATA_DIR
        self.models_dir = os.path.join(os.path.dirname(self.processed_data_dir), "models")
        
        # Anomaly detection parameters
        self.model_type = config.ANOMALY_DETECTION.get('model', 'isolation_forest')
        self.contamination = config.ANOMALY_DETECTION.get('contamination', 0.05)
        self.random_state = config.ANOMALY_DETECTION.get('random_state', 42)
        
        # Ensure directories exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
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
    
    def preprocess_features(self, features_df):
        """
        Preprocess features for anomaly detection.
        
        Args:
            features_df (pd.DataFrame): Wallet features
            
        Returns:
            tuple: (wallet_addresses, preprocessed_features)
        """
        logger.info("Preprocessing features for anomaly detection...")
        
        # Extract wallet addresses
        wallet_addresses = features_df['wallet'].values
        
        # Select numerical features only
        feature_columns = [col for col in features_df.columns 
                          if col != 'wallet' 
                          and features_df[col].dtype in ['int64', 'float64']]
        
        # Deal with potential infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # Fill any remaining NaN values with column medians
        for col in feature_columns:
            if features_df[col].isna().any():
                median_val = features_df[col].median()
                features_df[col] = features_df[col].fillna(median_val)
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df[feature_columns])
        
        # Save the scaler
        scaler_path = os.path.join(self.models_dir, "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved feature scaler to {scaler_path}")
        
        # Optionally, apply PCA for dimensionality reduction
        # This is helpful for high-dimensional feature spaces
        if len(feature_columns) > 10:
            logger.info(f"Applying PCA to reduce dimensions from {len(feature_columns)}")
            pca = PCA(n_components=min(10, len(feature_columns)), random_state=self.random_state)
            reduced_features = pca.fit_transform(scaled_features)
            
            # Save the PCA model
            pca_path = os.path.join(self.models_dir, "pca.joblib")
            joblib.dump(pca, pca_path)
            logger.info(f"Saved PCA model to {pca_path}")
            
            return wallet_addresses, reduced_features, feature_columns
        
        return wallet_addresses, scaled_features, feature_columns
    
    def train_model(self, features):
        """
        Train an anomaly detection model.
        
        Args:
            features (np.ndarray): Preprocessed features
            
        Returns:
            object: Trained model
        """
        logger.info(f"Training {self.model_type} model...")
        
        if self.model_type == 'isolation_forest':
            model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1  # Use all available processors
            )
        elif self.model_type == 'one_class_svm':
            model = OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='scale'
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Train the model
        model.fit(features)
        
        # Save the model
        model_path = os.path.join(self.models_dir, f"{self.model_type}.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Saved trained model to {model_path}")
        
        return model
    
    def score_anomalies(self, model, features, wallet_addresses):
        """
        Score the anomalies and create a DataFrame with results.
        
        Args:
            model (object): Trained anomaly detection model
            features (np.ndarray): Preprocessed features
            wallet_addresses (np.ndarray): Wallet addresses
            
        Returns:
            pd.DataFrame: Anomaly scores for each wallet
        """
        logger.info("Scoring anomalies...")
        
        # Get anomaly scores
        if self.model_type == 'isolation_forest':
            # For Isolation Forest, convert to anomaly score (higher = more anomalous)
            raw_scores = -model.decision_function(features)
        elif self.model_type == 'one_class_svm':
            # For One-Class SVM, convert to anomaly score (higher = more anomalous)
            raw_scores = -model.decision_function(features)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Normalize scores to [0, 1] range
        if len(raw_scores) > 1:
            min_score = np.min(raw_scores)
            max_score = np.max(raw_scores)
            if max_score > min_score:
                normalized_scores = (raw_scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.zeros_like(raw_scores)
        else:
            normalized_scores = np.array([0.5])  # Default for single wallet
        
        # Create DataFrame with wallet addresses and anomaly scores
        anomaly_df = pd.DataFrame({
            'wallet': wallet_addresses,
            'anomaly_score': normalized_scores,
            'is_anomaly': model.predict(features) == -1  # -1 indicates anomaly in sklearn
        })
        
        # Calculate percentile rank for each score
        anomaly_df['anomaly_percentile'] = anomaly_df['anomaly_score'].rank(pct=True) * 100
        
        # Save the anomaly scores
        scores_path = os.path.join(self.processed_data_dir, "anomaly_scores.parquet")
        anomaly_df.to_parquet(scores_path, index=False)
        logger.info(f"Saved anomaly scores for {len(anomaly_df)} wallets to {scores_path}")
        
        # Log summary statistics
        anomaly_count = anomaly_df['is_anomaly'].sum()
        logger.info(f"Detected {anomaly_count} anomalous wallets ({anomaly_count/len(anomaly_df)*100:.2f}%)")
        
        return anomaly_df
    
    def detect_anomalies(self):
        """
        Run the full anomaly detection pipeline.
        
        Returns:
            pd.DataFrame: Anomaly scores for each wallet
        """
        # Load features
        features_df = self.load_features()
        
        # Preprocess features
        wallet_addresses, preprocessed_features, feature_columns = self.preprocess_features(features_df)
        
        # Train model
        model = self.train_model(preprocessed_features)
        
        # Score anomalies
        anomaly_df = self.score_anomalies(model, preprocessed_features, wallet_addresses)
        
        # Optionally, analyze feature importance for anomalies
        if self.model_type == 'isolation_forest' and len(feature_columns) > 0:
            self.analyze_feature_importance(model, feature_columns, anomaly_df)
        
        return anomaly_df
    
    def analyze_feature_importance(self, model, feature_columns, anomaly_df):
        """
        Analyze feature importance for anomaly detection.
        
        Args:
            model (object): Trained model
            feature_columns (list): Feature column names
            anomaly_df (pd.DataFrame): Anomaly scores
        """
        if not hasattr(model, 'feature_importances_'):
            logger.info("Feature importance analysis not available for this model")
            return
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Create DataFrame with feature importances
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Save the feature importances
        importance_path = os.path.join(self.processed_data_dir, "feature_importances.csv")
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Saved feature importances to {importance_path}")
        
        # Log top important features
        top_features = importance_df.head(5)
        logger.info("Top features contributing to anomaly detection:")
        for i, (feature, importance) in enumerate(zip(top_features['feature'], top_features['importance'])):
            logger.info(f"{i+1}. {feature}: {importance:.4f}")

def main():
    """
    Main function to run anomaly detection.
    """
    detector = AnomalyDetector()
    anomaly_df = detector.detect_anomalies()
    
    print(f"Detected anomalies in {len(anomaly_df)} wallets")
    print(f"Number of anomalous wallets: {anomaly_df['is_anomaly'].sum()}")
    print("\nSample anomaly scores:")
    print(anomaly_df.sort_values('anomaly_score', ascending=False).head())
    
if __name__ == "__main__":
    main() 