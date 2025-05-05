"""
Model integration module for Compound V2 wallet scoring.
"""
import os
import logging
import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
import joblib
import tensorflow as tf

import src.config as config
from src.advanced.feature_selection import FeatureSelector
from src.advanced.hyperparameter_tuning import HyperparameterTuner
from src.advanced.deep_learning import DeepLearningModel

logger = logging.getLogger(__name__)

class ModelIntegrator:
    """
    Class for integrating different models for wallet scoring.
    """
    
    def __init__(self, processed_data_dir=None, results_dir=None):
        """
        Initialize the ModelIntegrator.
        
        Args:
            processed_data_dir (str, optional): Directory containing processed data
            results_dir (str, optional): Directory to save results
        """
        self.processed_data_dir = processed_data_dir or config.PROCESSED_DATA_DIR
        self.results_dir = results_dir or config.RESULTS_DIR
        
        # Create integration directory
        self.integration_dir = os.path.join(self.results_dir, "model_integration")
        os.makedirs(self.integration_dir, exist_ok=True)
        
        # Create models directory
        self.models_dir = os.path.join(self.integration_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize component classes
        self.feature_selector = FeatureSelector(processed_data_dir, results_dir)
        self.hyperparameter_tuner = HyperparameterTuner(processed_data_dir, results_dir)
        self.deep_learning_model = DeepLearningModel(processed_data_dir, results_dir)
    
    def load_data(self, use_selected_features=True):
        """
        Load feature data and target values.
        
        Args:
            use_selected_features (bool): Whether to use only selected features
            
        Returns:
            tuple: (X, y) feature matrix and target values
        """
        # Use the load_data method from the hyperparameter tuner
        return self.hyperparameter_tuner.load_data(use_selected_features=use_selected_features)
    
    def load_models(self):
        """
        Load all available models from different components.
        
        Returns:
            dict: Dictionary of loaded models
        """
        models = {}
        
        # Load traditional models from hyperparameter tuning
        tuning_dir = os.path.join(self.results_dir, "hyperparameter_tuning")
        for model_type in ['random_forest', 'gradient_boosting', 'ridge', 'lasso', 'elastic_net']:
            # Check which tuning methods are available for this model
            best_method = None
            best_score = float('inf')
            
            for method in ['optuna', 'random_search', 'grid_search']:
                result_path = os.path.join(tuning_dir, f"{model_type}_{method}_results.json")
                if os.path.exists(result_path):
                    with open(result_path, 'r') as f:
                        result = json.load(f)
                    if result['best_score'] < best_score:
                        best_score = result['best_score']
                        best_method = method
            
            if best_method:
                logger.info(f"Loading {model_type} model tuned with {best_method}")
                
                # Load parameters
                with open(os.path.join(tuning_dir, f"{model_type}_{best_method}_results.json"), 'r') as f:
                    params = json.load(f)['best_params']
                
                # Create model with best parameters
                model = self.hyperparameter_tuner.models[model_type].__class__(**params, random_state=42)
                models[f"{model_type}_{best_method}"] = model
        
        # Load deep learning models
        dl_dir = os.path.join(self.results_dir, "deep_learning", "models")
        if os.path.exists(dl_dir):
            for model_file in os.listdir(dl_dir):
                if model_file.endswith('.h5'):
                    model_name = model_file[:-3]  # Remove .h5 extension
                    try:
                        # Try to load the model
                        model_path = os.path.join(dl_dir, model_file)
                        model = tf.keras.models.load_model(model_path)
                        models[f"dl_{model_name}"] = model
                        logger.info(f"Loaded deep learning model: {model_name}")
                    except Exception as e:
                        logger.error(f"Failed to load model {model_file}: {e}")
        
        return models
    
    def train_models(self, X, y):
        """
        Train all available models on the provided data.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target values
            
        Returns:
            dict: Dictionary of trained models
        """
        models = self.load_models()
        trained_models = {}
        
        # Train traditional sklearn models
        for name, model in models.items():
            if not name.startswith('dl_'):
                logger.info(f"Training model: {name}")
                model.fit(X.fillna(0), y)
                trained_models[name] = model
        
        # We don't train deep learning models here as they have their own training process
        # that includes validation, early stopping, etc.
        
        return trained_models
    
    def build_ensemble(self, models, ensemble_type='voting'):
        """
        Build an ensemble model.
        
        Args:
            models (dict): Dictionary of trained models
            ensemble_type (str): Type of ensemble ('voting' or 'stacking')
            
        Returns:
            object: Ensemble model
        """
        # Filter out deep learning models which need special handling
        sklearn_models = {name: model for name, model in models.items() if not name.startswith('dl_')}
        
        if len(sklearn_models) < 2:
            logger.warning("Not enough models for ensemble. Need at least 2 models.")
            return None
        
        if ensemble_type == 'voting':
            # Create a voting regressor
            ensemble = VotingRegressor(
                estimators=[(name, model) for name, model in sklearn_models.items()],
                weights=None  # Equal weights for all models
            )
        elif ensemble_type == 'stacking':
            # Create a stacking regressor
            ensemble = StackingRegressor(
                estimators=[(name, model) for name, model in sklearn_models.items()],
                final_estimator=Ridge(alpha=1.0, random_state=42),
                cv=5
            )
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
        
        return ensemble
    
    def train_ensemble(self, X, y, ensemble_type='voting'):
        """
        Train an ensemble model.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target values
            ensemble_type (str): Type of ensemble ('voting' or 'stacking')
            
        Returns:
            object: Trained ensemble model
        """
        # Train individual models
        trained_models = self.train_models(X, y)
        
        # Build ensemble
        ensemble = self.build_ensemble(trained_models, ensemble_type)
        
        if ensemble is None:
            return None
        
        # Train ensemble
        logger.info(f"Training {ensemble_type} ensemble")
        ensemble.fit(X.fillna(0), y)
        
        # Save ensemble
        ensemble_path = os.path.join(self.models_dir, f"{ensemble_type}_ensemble.pkl")
        joblib.dump(ensemble, ensemble_path)
        logger.info(f"Ensemble saved to {ensemble_path}")
        
        return ensemble
    
    def predict_with_ensemble(self, X, ensemble_type='voting'):
        """
        Make predictions using an ensemble model.
        
        Args:
            X (pd.DataFrame): Features to predict on
            ensemble_type (str): Type of ensemble ('voting' or 'stacking')
            
        Returns:
            np.array: Predictions
        """
        # Load ensemble
        ensemble_path = os.path.join(self.models_dir, f"{ensemble_type}_ensemble.pkl")
        if not os.path.exists(ensemble_path):
            raise FileNotFoundError(f"Ensemble model not found: {ensemble_path}")
        
        ensemble = joblib.load(ensemble_path)
        logger.info(f"Loaded {ensemble_type} ensemble from {ensemble_path}")
        
        # Make predictions
        predictions = ensemble.predict(X.fillna(0))
        
        return predictions
    
    def hybrid_prediction(self, wallet_features, use_deep_learning=True, use_ensemble=True):
        """
        Make predictions using a hybrid approach combining traditional and deep learning models.
        
        Args:
            wallet_features (pd.DataFrame): Wallet features
            use_deep_learning (bool): Whether to use deep learning models
            use_ensemble (bool): Whether to use ensemble models
            
        Returns:
            pd.DataFrame: Predictions
        """
        predictions = {}
        
        # Prepare data
        wallet_ids = wallet_features['wallet'].values
        X = wallet_features.drop('wallet', axis=1)
        
        # Load models
        models = self.load_models()
        
        # Make predictions with traditional models
        for name, model in models.items():
            if not name.startswith('dl_'):
                logger.info(f"Making predictions with {name}")
                try:
                    preds = model.predict(X.fillna(0))
                    predictions[name] = preds
                except Exception as e:
                    logger.error(f"Error predicting with {name}: {e}")
        
        # Make predictions with deep learning models if available and requested
        if use_deep_learning:
            dl_models = {name: model for name, model in models.items() if name.startswith('dl_')}
            if dl_models:
                # Load scaler
                scaler_path = os.path.join(self.results_dir, "deep_learning", "models", "scaler.pkl")
                scaler = None
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                
                for name, model in dl_models.items():
                    logger.info(f"Making predictions with {name}")
                    try:
                        # Preprocess features if scaler is available
                        X_scaled = scaler.transform(X.fillna(0)) if scaler else X.fillna(0)
                        preds = model.predict(X_scaled)
                        predictions[name] = preds.flatten()  # Flatten in case of 2D output
                    except Exception as e:
                        logger.error(f"Error predicting with {name}: {e}")
        
        # Make predictions with ensemble if available and requested
        if use_ensemble:
            for ensemble_type in ['voting', 'stacking']:
                ensemble_path = os.path.join(self.models_dir, f"{ensemble_type}_ensemble.pkl")
                if os.path.exists(ensemble_path):
                    logger.info(f"Making predictions with {ensemble_type} ensemble")
                    try:
                        ensemble = joblib.load(ensemble_path)
                        preds = ensemble.predict(X.fillna(0))
                        predictions[f"{ensemble_type}_ensemble"] = preds
                    except Exception as e:
                        logger.error(f"Error predicting with {ensemble_type} ensemble: {e}")
        
        # Create results DataFrame
        results = pd.DataFrame({'wallet': wallet_ids})
        
        # Add predictions from each model
        for name, preds in predictions.items():
            results[f"score_{name}"] = preds
        
        # Add ensemble of all predictions
        if len(predictions) > 0:
            all_preds = np.column_stack(list(predictions.values()))
            results['score_mean'] = np.mean(all_preds, axis=1)
            results['score_median'] = np.median(all_preds, axis=1)
            results['score_min'] = np.min(all_preds, axis=1)
            results['score_max'] = np.max(all_preds, axis=1)
        
        return results
    
    def save_predictions(self, predictions, output_file='integrated_scores.csv'):
        """
        Save predictions to a file.
        
        Args:
            predictions (pd.DataFrame): Predictions DataFrame
            output_file (str): Output file name
        """
        output_path = os.path.join(self.results_dir, output_file)
        predictions.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
    
    def run_integration_pipeline(self, use_selected_features=True):
        """
        Run the complete model integration pipeline.
        
        Args:
            use_selected_features (bool): Whether to use only selected features
            
        Returns:
            pd.DataFrame: Final predictions
        """
        logger.info("Starting model integration pipeline")
        
        # Load data
        X, y = self.load_data(use_selected_features=use_selected_features)
        logger.info(f"Loaded data with {X.shape[1]} features for {X.shape[0]} wallets")
        
        # Train ensemble models
        for ensemble_type in ['voting', 'stacking']:
            self.train_ensemble(X, y, ensemble_type=ensemble_type)
        
        # Load wallet features for scoring
        features_path = os.path.join(self.processed_data_dir, "wallet_features.parquet")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        wallet_features = pd.read_parquet(features_path)
        
        # Make hybrid predictions
        predictions = self.hybrid_prediction(wallet_features)
        
        # Save predictions
        self.save_predictions(predictions)
        
        # Create a summary of the integration
        self._create_integration_summary(predictions)
        
        logger.info("Model integration pipeline complete")
        
        return predictions
    
    def _create_integration_summary(self, predictions):
        """
        Create a summary of the model integration.
        
        Args:
            predictions (pd.DataFrame): Predictions DataFrame
        """
        # Calculate statistics
        model_columns = [col for col in predictions.columns if col.startswith('score_') and col not in ['score_mean', 'score_median', 'score_min', 'score_max']]
        
        # Calculate correlations between model predictions
        corr_matrix = predictions[model_columns].corr()
        
        # Save correlation matrix
        corr_matrix.to_csv(os.path.join(self.integration_dir, "model_correlations.csv"))
        
        # Create summary
        with open(os.path.join(self.integration_dir, "integration_summary.md"), 'w') as f:
            f.write("# Model Integration Summary\n\n")
            
            f.write("## Models Used\n\n")
            for col in model_columns:
                model_name = col.replace('score_', '')
                f.write(f"- {model_name}\n")
            
            f.write("\n## Prediction Statistics\n\n")
            f.write("| Statistic | Value |\n")
            f.write("|-----------|-------|\n")
            f.write(f"| Number of wallets | {len(predictions)} |\n")
            f.write(f"| Number of models | {len(model_columns)} |\n")
            f.write(f"| Mean score | {predictions['score_mean'].mean():.4f} |\n")
            f.write(f"| Median score | {predictions['score_median'].mean():.4f} |\n")
            f.write(f"| Min score | {predictions['score_min'].min():.4f} |\n")
            f.write(f"| Max score | {predictions['score_max'].max():.4f} |\n")
            
            f.write("\n## Model Correlations\n\n")
            f.write("The correlation matrix between model predictions has been saved to `model_correlations.csv`.\n\n")
            
            # Find highly correlated model pairs
            high_corr_pairs = []
            for i, model1 in enumerate(model_columns):
                for j, model2 in enumerate(model_columns):
                    if i < j:  # upper triangle only
                        corr = corr_matrix.loc[model1, model2]
                        if abs(corr) > 0.8:
                            high_corr_pairs.append((model1, model2, corr))
            
            if high_corr_pairs:
                f.write("### Highly Correlated Models (|r| > 0.8)\n\n")
                for model1, model2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                    model1_name = model1.replace('score_', '')
                    model2_name = model2.replace('score_', '')
                    f.write(f"- {model1_name} & {model2_name}: r = {corr:.4f}\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("For final scoring, we recommend using:\n\n")
            f.write("1. The stacking ensemble model if available, as it combines all traditional models with optimized weights\n")
            f.write("2. The mean of all model predictions as a robust alternative\n")
            f.write("3. The best-performing individual model for simplicity\n\n")
            
            f.write("The integrated scores have been saved to the results directory as `integrated_scores.csv`.\n")

def main():
    """
    Run model integration as a standalone script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run model integration for wallet scoring")
    parser.add_argument('--selected-features', action='store_true',
                        help='Use only selected features')
    
    args = parser.parse_args()
    
    integrator = ModelIntegrator()
    integrator.run_integration_pipeline(use_selected_features=args.selected_features)

if __name__ == "__main__":
    main() 