"""
Hyperparameter tuning for Compound V2 wallet scoring models.
"""
import os
import logging
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import optuna
from optuna.visualization import plot_param_importances, plot_optimization_history

import src.config as config

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """
    Class for tuning hyperparameters of wallet scoring models.
    """
    
    def __init__(self, processed_data_dir=None, results_dir=None):
        """
        Initialize the HyperparameterTuner.
        
        Args:
            processed_data_dir (str, optional): Directory containing processed data
            results_dir (str, optional): Directory to save results
        """
        self.processed_data_dir = processed_data_dir or config.PROCESSED_DATA_DIR
        self.results_dir = results_dir or config.RESULTS_DIR
        
        # Create tuning directory
        self.tuning_dir = os.path.join(self.results_dir, "hyperparameter_tuning")
        os.makedirs(self.tuning_dir, exist_ok=True)
        
        # Available models
        self.models = {
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'ridge': Ridge(random_state=42),
            'lasso': Lasso(random_state=42),
            'elastic_net': ElasticNet(random_state=42)
        }
        
        # Cross-validation strategy
        self.cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    def load_data(self, use_selected_features=True):
        """
        Load feature data and target values.
        
        Args:
            use_selected_features (bool): Whether to use only selected features
            
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
        
        # Check if selected features file exists and should be used
        selected_features_path = os.path.join(self.results_dir, "feature_analysis", "selected_features.csv")
        if use_selected_features and os.path.exists(selected_features_path):
            selected_features = pd.read_csv(selected_features_path)['feature'].tolist()
            logger.info(f"Using {len(selected_features)} selected features")
            
            # Ensure all selected features exist in the dataset
            valid_features = [f for f in selected_features if f in features_df.columns]
            if len(valid_features) < len(selected_features):
                logger.warning(f"Some selected features are not in the dataset. Using {len(valid_features)} features.")
            
            # Keep wallet column and selected features
            feature_columns = ['wallet'] + valid_features
            features_df = features_df[feature_columns]
        
        # Separate features from target
        X = df.drop(['wallet', 'raw_score'], axis=1)
        y = df['raw_score']
        
        return X, y
    
    def grid_search_tune(self, model_name, param_grid, X, y):
        """
        Perform Grid Search CV to find optimal hyperparameters.
        
        Args:
            model_name (str): Name of the model to tune
            param_grid (dict): Parameter grid to search
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target values
            
        Returns:
            dict: Best parameters and scores
        """
        logger.info(f"Starting Grid Search for {model_name}...")
        
        # Create pipeline with preprocessing and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', self.models[model_name])
        ])
        
        # Create grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid={f'model__{key}': value for key, value in param_grid.items()},
            cv=self.cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X.fillna(0), y)
        
        # Get results
        results = {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,  # Convert back to positive MSE
            'cv_results': grid_search.cv_results_
        }
        
        # Save results
        with open(os.path.join(self.tuning_dir, f"{model_name}_grid_search_results.json"), 'w') as f:
            json.dump({
                'best_params': {k.replace('model__', ''): v for k, v in results['best_params'].items()},
                'best_score': results['best_score']
            }, f, indent=4)
        
        # Plot CV results
        self._plot_cv_results(model_name, grid_search.cv_results_, "grid_search")
        
        logger.info(f"Grid Search for {model_name} complete. Best MSE: {results['best_score']:.4f}")
        
        return results
    
    def random_search_tune(self, model_name, param_distributions, X, y, n_iter=20):
        """
        Perform Randomized Search CV to find optimal hyperparameters.
        
        Args:
            model_name (str): Name of the model to tune
            param_distributions (dict): Parameter distributions to sample from
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target values
            n_iter (int): Number of parameter settings to try
            
        Returns:
            dict: Best parameters and scores
        """
        logger.info(f"Starting Randomized Search for {model_name}...")
        
        # Create pipeline with preprocessing and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', self.models[model_name])
        ])
        
        # Create randomized search
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions={f'model__{key}': value for key, value in param_distributions.items()},
            n_iter=n_iter,
            cv=self.cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        # Fit randomized search
        random_search.fit(X.fillna(0), y)
        
        # Get results
        results = {
            'best_params': random_search.best_params_,
            'best_score': -random_search.best_score_,  # Convert back to positive MSE
            'cv_results': random_search.cv_results_
        }
        
        # Save results
        with open(os.path.join(self.tuning_dir, f"{model_name}_random_search_results.json"), 'w') as f:
            json.dump({
                'best_params': {k.replace('model__', ''): v for k, v in results['best_params'].items()},
                'best_score': results['best_score']
            }, f, indent=4)
        
        # Plot CV results
        self._plot_cv_results(model_name, random_search.cv_results_, "random_search")
        
        logger.info(f"Randomized Search for {model_name} complete. Best MSE: {results['best_score']:.4f}")
        
        return results
    
    def optuna_tune(self, model_name, param_space, X, y, n_trials=100):
        """
        Perform Optuna optimization to find optimal hyperparameters.
        
        Args:
            model_name (str): Name of the model to tune
            param_space (function): Function that defines parameter space
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target values
            n_trials (int): Number of trials to run
            
        Returns:
            dict: Best parameters and scores
        """
        logger.info(f"Starting Optuna optimization for {model_name}...")
        
        # Fill NaN values
        X_filled = X.fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filled)
        
        # Define objective function
        def objective(trial):
            # Get parameters for this trial
            params = param_space(trial)
            
            # Create model with these parameters
            model = self.models[model_name].__class__(**params, random_state=42)
            
            # Evaluate model using cross-validation
            score = cross_val_score(
                model,
                X_scaled,
                y,
                cv=self.cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            # Return mean negative MSE
            return score.mean()
        
        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters and score
        best_params = study.best_params
        best_score = -study.best_value  # Convert back to positive MSE
        
        # Save results
        with open(os.path.join(self.tuning_dir, f"{model_name}_optuna_results.json"), 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_score': best_score
            }, f, indent=4)
        
        # Plot optimization history
        fig = plot_optimization_history(study)
        fig.write_image(os.path.join(self.tuning_dir, f"{model_name}_optuna_history.png"))
        
        # Plot parameter importances
        fig = plot_param_importances(study)
        fig.write_image(os.path.join(self.tuning_dir, f"{model_name}_optuna_param_importances.png"))
        
        logger.info(f"Optuna optimization for {model_name} complete. Best MSE: {best_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'study': study
        }
    
    def _plot_cv_results(self, model_name, cv_results, method):
        """
        Plot cross-validation results.
        
        Args:
            model_name (str): Name of the model
            cv_results (dict): CV results from GridSearchCV or RandomizedSearchCV
            method (str): Method name (grid_search or random_search)
        """
        # Convert to DataFrame
        results_df = pd.DataFrame(cv_results)
        
        # Extract mean test scores and sort
        mean_scores = -results_df['mean_test_score']  # Convert back to positive MSE
        
        # Plot mean test scores
        plt.figure(figsize=(10, 6))
        plt.title(f"{model_name} - {method.replace('_', ' ').title()} Results")
        plt.plot(range(len(mean_scores)), np.sort(mean_scores))
        plt.xlabel('Parameter Combination (sorted by MSE)')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.tuning_dir, f"{model_name}_{method}_results.png"))
        plt.close()
    
    def run_tuning(self, model_name, method='all', use_selected_features=True):
        """
        Run hyperparameter tuning for the specified model.
        
        Args:
            model_name (str): Name of the model to tune
            method (str): Tuning method ('grid', 'random', 'optuna', or 'all')
            use_selected_features (bool): Whether to use only selected features
            
        Returns:
            dict: Tuning results
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.models.keys())}")
        
        # Load data
        X, y = self.load_data(use_selected_features=use_selected_features)
        
        results = {}
        
        # Define parameter grids/distributions for each model
        if model_name == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            param_distributions = {
                'n_estimators': np.arange(50, 300, 10),
                'max_depth': np.arange(5, 50, 5),
                'min_samples_split': np.arange(2, 20, 2),
                'min_samples_leaf': np.arange(1, 10, 1)
            }
            
            def param_space(trial):
                return {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 50),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
                }
        
        elif model_name == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
            
            param_distributions = {
                'n_estimators': np.arange(50, 300, 10),
                'learning_rate': np.logspace(-3, 0, 20),
                'max_depth': np.arange(3, 15, 1),
                'min_samples_split': np.arange(2, 20, 2)
            }
            
            def param_space(trial):
                return {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0)
                }
        
        elif model_name in ['ridge', 'lasso', 'elastic_net']:
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            }
            
            param_distributions = {
                'alpha': np.logspace(-3, 2, 20)
            }
            
            if model_name == 'elastic_net':
                param_grid['l1_ratio'] = [0.1, 0.3, 0.5, 0.7, 0.9]
                param_distributions['l1_ratio'] = np.linspace(0.1, 0.9, 9)
                
                def param_space(trial):
                    return {
                        'alpha': trial.suggest_float('alpha', 0.001, 100.0, log=True),
                        'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9)
                    }
            else:
                def param_space(trial):
                    return {
                        'alpha': trial.suggest_float('alpha', 0.001, 100.0, log=True)
                    }
        
        # Run grid search if requested
        if method in ['grid', 'all']:
            results['grid_search'] = self.grid_search_tune(model_name, param_grid, X, y)
        
        # Run randomized search if requested
        if method in ['random', 'all']:
            results['random_search'] = self.random_search_tune(
                model_name, param_distributions, X, y, n_iter=30
            )
        
        # Run Optuna optimization if requested
        if method in ['optuna', 'all']:
            results['optuna'] = self.optuna_tune(model_name, param_space, X, y, n_trials=100)
        
        # Create summary of results
        self._create_tuning_summary(model_name, results)
        
        return results
    
    def _create_tuning_summary(self, model_name, results):
        """
        Create a summary of tuning results.
        
        Args:
            model_name (str): Name of the model
            results (dict): Tuning results
        """
        with open(os.path.join(self.tuning_dir, f"{model_name}_tuning_summary.md"), 'w') as f:
            f.write(f"# Hyperparameter Tuning Summary for {model_name}\n\n")
            
            # Compare methods
            f.write("## Comparison of Methods\n\n")
            f.write("| Method | Best MSE | Best Parameters |\n")
            f.write("|--------|----------|----------------|\n")
            
            for method, method_results in results.items():
                best_params_str = ", ".join([
                    f"{k.replace('model__', '')}: {v}" 
                    for k, v in method_results['best_params'].items()
                ])
                f.write(f"| {method.replace('_', ' ').title()} | {method_results['best_score']:.4f} | {best_params_str} |\n")
            
            f.write("\n## Recommended Configuration\n\n")
            
            # Determine best method based on lowest MSE
            best_method = min(results.items(), key=lambda x: x[1]['best_score'])[0]
            best_score = results[best_method]['best_score']
            best_params = results[best_method]['best_params']
            
            f.write(f"Based on the results, we recommend using the parameters found by **{best_method.replace('_', ' ').title()}**:\n\n")
            f.write(f"- Mean Squared Error: {best_score:.4f}\n")
            f.write("- Parameters:\n")
            
            for param, value in best_params.items():
                param_name = param.replace('model__', '')
                f.write(f"  - {param_name}: {value}\n")

def main():
    """
    Run hyperparameter tuning as a standalone script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Tune hyperparameters for wallet scoring models")
    parser.add_argument('--model', type=str, required=True, 
                        choices=['random_forest', 'gradient_boosting', 'ridge', 'lasso', 'elastic_net'],
                        help='Model to tune')
    parser.add_argument('--method', type=str, default='all', 
                        choices=['grid', 'random', 'optuna', 'all'],
                        help='Tuning method')
    parser.add_argument('--selected-features', action='store_true',
                        help='Use only selected features')
    
    args = parser.parse_args()
    
    tuner = HyperparameterTuner()
    tuner.run_tuning(args.model, method=args.method, use_selected_features=args.selected_features)

if __name__ == "__main__":
    main() 