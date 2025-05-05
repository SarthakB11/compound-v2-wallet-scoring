"""
Anomaly detection for wallet behavior.
"""
import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import joblib

from src.utils.helpers import save_dataframe
import src.config as config

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Class for detecting anomalous wallet behavior.
    """
    
    def __init__(self, model_dir=None, processed_dir=None, model_params=None):
        """
        Initialize the anomaly detector.
        
        Args:
            model_dir (str, optional): Directory to save/load models
            processed_dir (str, optional): Directory with processed data
            model_params (dict, optional): Model parameters
        """
        self.model_dir = model_dir or os.path.join(config.DATA_DIR, "models")
        self.processed_dir = processed_dir or config.PROCESSED_DATA_DIR
        self.model_params = model_params or config.ANOMALY_DETECTION
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # File paths
        self.features_file = os.path.join(self.processed_dir, "wallet_features.parquet")
        self.anomaly_scores_file = os.path.join(self.processed_dir, "anomaly_scores.parquet")
        self.scaler_file = os.path.join(self.model_dir, "scaler.joblib")
        self.model_file = os.path.join(self.model_dir, "anomaly_model.joblib")
    
    def preprocess_features(self, features_df):
        """
        Preprocess the features for anomaly detection.
        
        Args:
            features_df (pd.DataFrame): DataFrame with wallet features
            
        Returns:
            tuple: Wallet addresses, preprocessed features, feature column names
        """
        logger.info("Preprocessing features for anomaly detection...")
        
        # Drop non-feature columns
        feature_columns = [col for col in features_df.columns 
                          if col not in ['wallet', 'first_tx_timestamp', 'last_tx_timestamp']]
        
        # Get wallet addresses
        wallet_addresses = features_df['wallet'].values
        
        # Extract features
        X = features_df[feature_columns].values
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(X)
        
        # Save scaler for future use
        joblib.dump(scaler, self.scaler_file)
        logger.info(f"Saved feature scaler to {self.scaler_file}")
        
        # Apply PCA for dimension reduction
        num_components = min(10, len(wallet_addresses) - 1, len(feature_columns))
        logger.info(f"Applying PCA to reduce dimensions from {len(feature_columns)}")
        if num_components > 0:
            pca = PCA(n_components=num_components)
            reduced_features = pca.fit_transform(scaled_features)
            logger.info(f"Reduced dimensions to {num_components} components")
            return wallet_addresses, reduced_features, feature_columns
        else:
            logger.info("Skipping PCA due to insufficient samples")
            return wallet_addresses, scaled_features, feature_columns
    
    def train_isolation_forest(self, features):
        """
        Train an Isolation Forest model for anomaly detection.
        
        Args:
            features (np.ndarray): Preprocessed features
            
        Returns:
            object: Trained model
        """
        logger.info("Training Isolation Forest model...")
        model = IsolationForest(
            contamination=self.model_params.get('contamination', 0.05),
            random_state=self.model_params.get('random_state', 42),
            n_estimators=100,
            max_samples='auto'
        )
        model.fit(features)
        return model
    
    def train_one_class_svm(self, features):
        """
        Train a One-Class SVM model for anomaly detection.
        
        Args:
            features (np.ndarray): Preprocessed features
            
        Returns:
            object: Trained model
        """
        logger.info("Training One-Class SVM model...")
        model = OneClassSVM(
            nu=self.model_params.get('contamination', 0.05),
            kernel='rbf',
            gamma='scale'
        )
        model.fit(features)
        return model
    
    def create_anomaly_scores(self, wallet_addresses, features, model):
        """
        Create a DataFrame with anomaly scores.
        
        Args:
            wallet_addresses (np.ndarray): Wallet addresses
            features (np.ndarray): Preprocessed features
            model (object): Trained anomaly detection model
            
        Returns:
            pd.DataFrame: DataFrame with anomaly scores
        """
        # Get anomaly scores
        if hasattr(model, 'decision_function'):
            # Isolation Forest or One-Class SVM
            raw_scores = model.decision_function(features)
            # Convert to a 0-1 anomaly score (higher is more anomalous)
            anomaly_scores = 1 - (raw_scores - np.min(raw_scores)) / (np.max(raw_scores) - np.min(raw_scores) + 1e-10)
        else:
            # Fallback for models without decision_function
            predictions = model.predict(features)
            # Convert predictions (-1 for anomalies, 1 for normal) to scores
            anomaly_scores = np.where(predictions == -1, 0.9, 0.1)
        
        # Flag anomalies
        is_anomaly = model.predict(features) == -1
        
        # Create DataFrame
        anomaly_df = pd.DataFrame({
            'wallet': wallet_addresses,
            'anomaly_score': anomaly_scores,
            'is_anomaly': is_anomaly
        })
        
        logger.info(f"Identified {anomaly_df['is_anomaly'].sum()} anomalous wallets out of {len(anomaly_df)}")
        
        return anomaly_df
    
    def detect_anomalies(self):
        """
        Detect anomalies in wallet behavior.
        
        Returns:
            pd.DataFrame: DataFrame with anomaly scores
        """
        # Load features
        features_df = pd.read_parquet(self.features_file)
        logger.info(f"Loaded features for {len(features_df)} wallets from {self.features_file}")
        
        # Preprocess features
        wallet_addresses, preprocessed_features, feature_columns = self.preprocess_features(features_df)
        
        # Train model based on the selected method
        model_type = self.model_params.get('model', 'isolation_forest')
        
        if model_type == 'isolation_forest':
            model = self.train_isolation_forest(preprocessed_features)
        elif model_type == 'one_class_svm':
            model = self.train_one_class_svm(preprocessed_features)
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
        # Save the model
        joblib.dump(model, self.model_file)
        logger.info(f"Saved anomaly detection model to {self.model_file}")
        
        # Create anomaly scores
        anomaly_df = self.create_anomaly_scores(wallet_addresses, preprocessed_features, model)
        
        # Save results
        save_dataframe(anomaly_df, self.anomaly_scores_file)
        logger.info(f"Saved anomaly scores to {self.anomaly_scores_file}")
        
        return anomaly_df

def main():
    """
    Main function to run anomaly detection.
    """
    detector = AnomalyDetector()
    anomaly_df = detector.detect_anomalies()
    print(f"Detected {anomaly_df['is_anomaly'].sum()} anomalous wallets out of {len(anomaly_df)}")
    
if __name__ == "__main__":
    main() 