"""
Deep learning models for Compound V2 wallet scoring.
"""
import os
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, 
    Concatenate, LSTM, Bidirectional, Embedding,
    GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path

import src.config as config

logger = logging.getLogger(__name__)

class DeepLearningModel:
    """
    Class for creating and training deep learning models for wallet scoring.
    """
    
    def __init__(self, processed_data_dir=None, results_dir=None):
        """
        Initialize the DeepLearningModel.
        
        Args:
            processed_data_dir (str, optional): Directory containing processed data
            results_dir (str, optional): Directory to save results
        """
        self.processed_data_dir = processed_data_dir or config.PROCESSED_DATA_DIR
        self.results_dir = results_dir or config.RESULTS_DIR
        
        # Create deep learning directory
        self.dl_dir = os.path.join(self.results_dir, "deep_learning")
        os.makedirs(self.dl_dir, exist_ok=True)
        
        # Create models directory
        self.models_dir = os.path.join(self.dl_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Create logs directory for TensorBoard
        self.logs_dir = os.path.join(self.dl_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def load_data(self, use_selected_features=True, include_sequence_data=False):
        """
        Load feature data and target values.
        
        Args:
            use_selected_features (bool): Whether to use only selected features
            include_sequence_data (bool): Whether to include transaction sequence data
            
        Returns:
            tuple: Features and target values
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
        sequence_data = None
        if use_selected_features:
            selected_features_path = os.path.join(self.results_dir, "feature_analysis", "selected_features.csv")
            if os.path.exists(selected_features_path):
                selected_features = pd.read_csv(selected_features_path)['feature'].tolist()
                logger.info(f"Using {len(selected_features)} selected features")
                
                # Ensure all selected features exist in the dataset
                valid_features = [f for f in selected_features if f in features_df.columns]
                if len(valid_features) < len(selected_features):
                    logger.warning(f"Some selected features are not in the dataset. Using {len(valid_features)} features.")
                
                # Keep wallet column and selected features
                feature_columns = ['wallet'] + valid_features
                df = df[['wallet', 'raw_score'] + valid_features]
        
        # Load sequence data if requested
        if include_sequence_data:
            seq_path = os.path.join(self.processed_data_dir, "transaction_sequences.npz")
            if os.path.exists(seq_path):
                sequence_data = np.load(seq_path, allow_pickle=True)
                logger.info("Loaded transaction sequence data")
            else:
                logger.warning("Transaction sequence data not found. Creating a simple model without sequence data.")
        
        # Separate features, wallets and target
        wallets = df['wallet'].values
        X = df.drop(['wallet', 'raw_score'], axis=1)
        y = df['raw_score'].values
        
        return wallets, X, y, sequence_data
    
    def preprocess_data(self, X, y, test_size=0.2):
        """
        Preprocess data for deep learning.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (np.array): Target values
            test_size (float): Proportion of data to use for testing
            
        Returns:
            tuple: Preprocessed data splits and preprocessing objects
        """
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale numerical features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler for later use
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        import joblib
        joblib.dump(scaler, scaler_path)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def build_dense_model(self, input_dim, architecture='standard'):
        """
        Build a dense neural network model.
        
        Args:
            input_dim (int): Input dimension
            architecture (str): Architecture type ('standard', 'deep', or 'wide')
            
        Returns:
            tf.keras.Model: Compiled model
        """
        if architecture == 'standard':
            model = Sequential([
                Input(shape=(input_dim,)),
                Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                BatchNormalization(),
                Dense(1)
            ])
        elif architecture == 'deep':
            model = Sequential([
                Input(shape=(input_dim,)),
                Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                BatchNormalization(),
                Dropout(0.4),
                Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                BatchNormalization(),
                Dropout(0.4),
                Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                BatchNormalization(),
                Dense(1)
            ])
        elif architecture == 'wide':
            model = Sequential([
                Input(shape=(input_dim,)),
                Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                BatchNormalization(),
                Dropout(0.5),
                Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                BatchNormalization(),
                Dropout(0.5),
                Dense(1)
            ])
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_sequence_model(self, static_input_dim, seq_len, seq_features):
        """
        Build a model that combines static features with sequence data.
        
        Args:
            static_input_dim (int): Dimension of static features
            seq_len (int): Length of sequences
            seq_features (int): Number of features per sequence item
            
        Returns:
            tf.keras.Model: Compiled model
        """
        # Static features input
        static_input = Input(shape=(static_input_dim,), name='static_input')
        static_dense = Dense(64, activation='relu')(static_input)
        static_bn = BatchNormalization()(static_dense)
        static_output = Dropout(0.3)(static_bn)
        
        # Sequence input
        seq_input = Input(shape=(seq_len, seq_features), name='sequence_input')
        
        # Bidirectional LSTM for sequence processing
        lstm = Bidirectional(LSTM(64, return_sequences=True))(seq_input)
        lstm_bn = BatchNormalization()(lstm)
        lstm_dropout = Dropout(0.3)(lstm_bn)
        lstm2 = Bidirectional(LSTM(32, return_sequences=False))(lstm_dropout)
        seq_output = BatchNormalization()(lstm2)
        
        # Combine static and sequence features
        combined = Concatenate()([static_output, seq_output])
        
        # Final dense layers
        dense1 = Dense(64, activation='relu')(combined)
        bn1 = BatchNormalization()(dense1)
        dropout1 = Dropout(0.3)(bn1)
        dense2 = Dense(32, activation='relu')(dropout1)
        bn2 = BatchNormalization()(dense2)
        dropout2 = Dropout(0.3)(bn2)
        
        # Output layer
        output = Dense(1)(dropout2)
        
        # Create and compile model
        model = Model(inputs=[static_input, seq_input], outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, model, X_train, y_train, X_test, y_test, 
                   batch_size=32, epochs=100, model_name='wallet_scoring_model'):
        """
        Train a deep learning model.
        
        Args:
            model (tf.keras.Model): Model to train
            X_train (np.array): Training features
            y_train (np.array): Training targets
            X_test (np.array): Test features
            y_test (np.array): Test targets
            batch_size (int): Batch size
            epochs (int): Maximum number of epochs
            model_name (str): Name for saving the model
            
        Returns:
            dict: Training history and evaluation metrics
        """
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.models_dir, f"{model_name}.h5"),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            TensorBoard(log_dir=os.path.join(self.logs_dir, model_name))
        ]
        
        # Train model
        logger.info(f"Training {model_name}...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=2
        )
        
        # Evaluate model
        evaluation = model.evaluate(X_test, y_test, verbose=0)
        
        # Create evaluation report
        eval_results = {
            'test_loss': evaluation[0],
            'test_mae': evaluation[1],
            'training_history': {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'mae': [float(x) for x in history.history['mae']],
                'val_mae': [float(x) for x in history.history['val_mae']]
            }
        }
        
        # Save evaluation results
        with open(os.path.join(self.dl_dir, f"{model_name}_evaluation.json"), 'w') as f:
            json.dump(eval_results, f, indent=4)
        
        # Plot training history
        self._plot_training_history(history, model_name)
        
        logger.info(f"Model training complete. Test Loss: {evaluation[0]:.4f}, Test MAE: {evaluation[1]:.4f}")
        
        return eval_results
    
    def train_sequence_model(self, model, static_train, seq_train, y_train, 
                            static_test, seq_test, y_test, batch_size=32, 
                            epochs=100, model_name='sequence_model'):
        """
        Train a sequence model.
        
        Args:
            model (tf.keras.Model): Model to train
            static_train (np.array): Static training features
            seq_train (np.array): Sequence training features
            y_train (np.array): Training targets
            static_test (np.array): Static test features
            seq_test (np.array): Sequence test features
            y_test (np.array): Test targets
            batch_size (int): Batch size
            epochs (int): Maximum number of epochs
            model_name (str): Name for saving the model
            
        Returns:
            dict: Training history and evaluation metrics
        """
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.models_dir, f"{model_name}.h5"),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            TensorBoard(log_dir=os.path.join(self.logs_dir, model_name))
        ]
        
        # Train model
        logger.info(f"Training {model_name}...")
        history = model.fit(
            {'static_input': static_train, 'sequence_input': seq_train},
            y_train,
            validation_data=({'static_input': static_test, 'sequence_input': seq_test}, y_test),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=2
        )
        
        # Evaluate model
        evaluation = model.evaluate(
            {'static_input': static_test, 'sequence_input': seq_test},
            y_test,
            verbose=0
        )
        
        # Create evaluation report
        eval_results = {
            'test_loss': evaluation[0],
            'test_mae': evaluation[1],
            'training_history': {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'mae': [float(x) for x in history.history['mae']],
                'val_mae': [float(x) for x in history.history['val_mae']]
            }
        }
        
        # Save evaluation results
        with open(os.path.join(self.dl_dir, f"{model_name}_evaluation.json"), 'w') as f:
            json.dump(eval_results, f, indent=4)
        
        # Plot training history
        self._plot_training_history(history, model_name)
        
        logger.info(f"Model training complete. Test Loss: {evaluation[0]:.4f}, Test MAE: {evaluation[1]:.4f}")
        
        return eval_results
    
    def _plot_training_history(self, history, model_name):
        """
        Plot training history.
        
        Args:
            history (tf.keras.callbacks.History): Training history
            model_name (str): Model name for saving plots
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot MAE
        ax2.plot(history.history['mae'], label='Training MAE')
        ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Mean Absolute Error')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.dl_dir, f"{model_name}_training_history.png"))
        plt.close()
    
    def predict(self, model, X, scaler=None):
        """
        Make predictions using a trained model.
        
        Args:
            model (tf.keras.Model): Trained model
            X (pd.DataFrame): Features to predict on
            scaler (sklearn.preprocessing.StandardScaler, optional): Scaler for preprocessing
            
        Returns:
            np.array: Predictions
        """
        # Preprocess features if scaler is provided
        if scaler is not None:
            X = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X)
        
        return predictions
    
    def predict_with_sequence_model(self, model, static_features, sequence_features, scaler=None):
        """
        Make predictions using a trained sequence model.
        
        Args:
            model (tf.keras.Model): Trained model
            static_features (pd.DataFrame): Static features
            sequence_features (np.array): Sequence features
            scaler (sklearn.preprocessing.StandardScaler, optional): Scaler for preprocessing
            
        Returns:
            np.array: Predictions
        """
        # Preprocess static features if scaler is provided
        if scaler is not None:
            static_features = scaler.transform(static_features)
        
        # Make predictions
        predictions = model.predict({
            'static_input': static_features,
            'sequence_input': sequence_features
        })
        
        return predictions
    
    def save_model(self, model, model_name):
        """
        Save model to disk.
        
        Args:
            model (tf.keras.Model): Model to save
            model_name (str): Name for saving the model
        """
        model_path = os.path.join(self.models_dir, f"{model_name}.h5")
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save model architecture
        with open(os.path.join(self.models_dir, f"{model_name}_architecture.json"), 'w') as f:
            f.write(model.to_json())
    
    def load_model(self, model_name):
        """
        Load model from disk.
        
        Args:
            model_name (str): Name of the model to load
            
        Returns:
            tf.keras.Model: Loaded model
        """
        model_path = os.path.join(self.models_dir, f"{model_name}.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        return model
    
    def run_experiment(self, architecture='standard', batch_size=32, epochs=100, use_selected_features=True):
        """
        Run a deep learning experiment.
        
        Args:
            architecture (str): Model architecture ('standard', 'deep', or 'wide')
            batch_size (int): Batch size
            epochs (int): Maximum number of epochs
            use_selected_features (bool): Whether to use only selected features
            
        Returns:
            dict: Evaluation results
        """
        # Load data
        wallets, X, y, _ = self.load_data(use_selected_features=use_selected_features)
        
        # Preprocess data
        X_train, X_test, y_train, y_test, scaler = self.preprocess_data(X, y)
        
        # Build model
        model = self.build_dense_model(X_train.shape[1], architecture=architecture)
        
        # Train model
        model_name = f"wallet_scoring_{architecture}"
        results = self.train_model(
            model, X_train, y_train, X_test, y_test,
            batch_size=batch_size, epochs=epochs, model_name=model_name
        )
        
        # Save model
        self.save_model(model, model_name)
        
        return results
    
    def run_sequence_experiment(self, batch_size=32, epochs=100, use_selected_features=True):
        """
        Run a sequence-based deep learning experiment.
        
        Args:
            batch_size (int): Batch size
            epochs (int): Maximum number of epochs
            use_selected_features (bool): Whether to use only selected features
            
        Returns:
            dict: Evaluation results or None if sequence data not available
        """
        # Load data including sequence data
        wallets, X, y, sequence_data = self.load_data(
            use_selected_features=use_selected_features,
            include_sequence_data=True
        )
        
        # Check if sequence data is available
        if sequence_data is None:
            logger.error("No sequence data available for sequence experiment")
            return None
        
        # Extract sequence features and lengths
        sequences = sequence_data['sequences']
        seq_len = sequences.shape[1]
        seq_features = sequences.shape[2]
        
        # Preprocess static data
        X_train_static, X_test_static, y_train, y_test, scaler = self.preprocess_data(X, y)
        
        # Split sequence data
        train_indices = np.arange(len(X))
        np.random.shuffle(train_indices)
        train_size = int(0.8 * len(X))
        train_indices, test_indices = train_indices[:train_size], train_indices[train_size:]
        
        seq_train = sequences[train_indices]
        seq_test = sequences[test_indices]
        
        # Build sequence model
        model = self.build_sequence_model(X_train_static.shape[1], seq_len, seq_features)
        
        # Train model
        model_name = "wallet_scoring_sequence"
        results = self.train_sequence_model(
            model, X_train_static, seq_train, y_train, X_test_static, seq_test, y_test,
            batch_size=batch_size, epochs=epochs, model_name=model_name
        )
        
        # Save model
        self.save_model(model, model_name)
        
        return results
    
    def compare_architectures(self):
        """
        Compare different model architectures.
        
        Returns:
            dict: Comparison results
        """
        architectures = ['standard', 'deep', 'wide']
        results = {}
        
        for arch in architectures:
            logger.info(f"Running experiment with {arch} architecture")
            res = self.run_experiment(architecture=arch)
            results[arch] = res
        
        # Compare results
        comparison = {
            'test_loss': {arch: res['test_loss'] for arch, res in results.items()},
            'test_mae': {arch: res['test_mae'] for arch, res in results.items()}
        }
        
        # Find best architecture
        best_arch = min(results.items(), key=lambda x: x[1]['test_loss'])[0]
        comparison['best_architecture'] = best_arch
        
        # Save comparison
        with open(os.path.join(self.dl_dir, "architecture_comparison.json"), 'w') as f:
            json.dump(comparison, f, indent=4)
        
        # Create comparison plots
        self._plot_architecture_comparison(results)
        
        logger.info(f"Architecture comparison complete. Best architecture: {best_arch}")
        
        return comparison
    
    def _plot_architecture_comparison(self, results):
        """
        Plot architecture comparison.
        
        Args:
            results (dict): Results for different architectures
        """
        architectures = list(results.keys())
        
        # Extract metrics
        test_loss = [results[arch]['test_loss'] for arch in architectures]
        test_mae = [results[arch]['test_mae'] for arch in architectures]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        ax1.bar(architectures, test_loss)
        ax1.set_title('Test Loss (MSE)')
        ax1.set_xlabel('Architecture')
        ax1.set_ylabel('Loss')
        
        # Plot MAE
        ax2.bar(architectures, test_mae)
        ax2.set_title('Test MAE')
        ax2.set_xlabel('Architecture')
        ax2.set_ylabel('MAE')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.dl_dir, "architecture_comparison.png"))
        plt.close()

def main():
    """
    Run deep learning models as a standalone script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Train deep learning models for wallet scoring")
    parser.add_argument('--architecture', type=str, default='standard', 
                        choices=['standard', 'deep', 'wide', 'sequence', 'compare'],
                        help='Model architecture or comparison')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--selected-features', action='store_true',
                        help='Use only selected features')
    
    args = parser.parse_args()
    
    dl_model = DeepLearningModel()
    
    if args.architecture == 'compare':
        dl_model.compare_architectures()
    elif args.architecture == 'sequence':
        dl_model.run_sequence_experiment(
            batch_size=args.batch_size,
            epochs=args.epochs,
            use_selected_features=args.selected_features
        )
    else:
        dl_model.run_experiment(
            architecture=args.architecture,
            batch_size=args.batch_size,
            epochs=args.epochs,
            use_selected_features=args.selected_features
        )

if __name__ == "__main__":
    main() 