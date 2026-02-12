"""
LSTM Model Architecture
Time-series deep learning model for micro-muscle fatigue prediction

Why LSTM?
- LSTMs (Long Short-Term Memory networks) are designed for sequential data
- They can capture temporal dependencies and patterns over time
- Memory cells remember important information from past timesteps
- Perfect for predicting fatigue which accumulates progressively
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import pickle


class FatigueLSTMModel:
    """
    LSTM-based fatigue prediction model
    """
    
    def __init__(self, sequence_length=30, n_features=6):
        """
        Initialize LSTM model
        
        Args:
            sequence_length: Number of timesteps in each input sequence
            n_features: Number of sensor features per timestep
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = None
        self.history = None
        
    def build_model(self, lstm_units=[64, 32], dropout_rate=0.2):
        """
        Build LSTM architecture
        
        Architecture:
        1. Input layer: (sequence_length, n_features)
        2. LSTM layers: Learn temporal patterns
        3. Dropout: Prevent overfitting
        4. Dense layers: Final prediction
        5. Output: Fatigue score (0-100)
        
        Args:
            lstm_units: List of LSTM layer sizes
            dropout_rate: Dropout probability
        """
        
        model = models.Sequential([
            # First LSTM layer - returns sequences for stacking
            layers.LSTM(
                lstm_units[0], 
                return_sequences=True,
                input_shape=(self.sequence_length, self.n_features),
                name='lstm_1'
            ),
            layers.Dropout(dropout_rate, name='dropout_1'),
            
            # Second LSTM layer - returns final hidden state
            layers.LSTM(
                lstm_units[1],
                return_sequences=False,
                name='lstm_2'
            ),
            layers.Dropout(dropout_rate, name='dropout_2'),
            
            # Dense layers for prediction
            layers.Dense(32, activation='relu', name='dense_1'),
            layers.Dropout(dropout_rate / 2, name='dropout_3'),
            
            layers.Dense(16, activation='relu', name='dense_2'),
            
            # Output layer - fatigue score (0-100)
            # Using linear activation for regression
            layers.Dense(1, activation='linear', name='output')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        return model
    
    def preprocess_data(self, X, y=None, fit_scaler=False):
        """
        Normalize features and targets
        
        Time-series windowing:
        - Each sample is a sequence of past 30 timesteps
        - Features are normalized across all samples
        - Target (fatigue score) is also normalized for stable training
        
        Args:
            X: Input sequences (n_samples, sequence_length, n_features)
            y: Target fatigue scores (n_samples,)
            fit_scaler: Whether to fit scalers (True for training data)
        """
        
        # Reshape X for scaling: (n_samples * sequence_length, n_features)
        n_samples = X.shape[0]
        X_reshaped = X.reshape(-1, self.n_features)
        
        if fit_scaler:
            X_scaled = self.scaler_X.fit_transform(X_reshaped)
        else:
            X_scaled = self.scaler_X.transform(X_reshaped)
        
        # Reshape back to sequences
        X_scaled = X_scaled.reshape(n_samples, self.sequence_length, self.n_features)
        
        # Scale targets if provided
        if y is not None:
            # Normalize to 0-1 range for better training
            y_scaled = y / 100.0
            return X_scaled, y_scaled
        
        return X_scaled
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the LSTM model
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        
        # Preprocess data
        X_train_scaled, y_train_scaled = self.preprocess_data(X_train, y_train, fit_scaler=True)
        X_val_scaled, y_val_scaled = self.preprocess_data(X_val, y_val, fit_scaler=False)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train model
        print("Training LSTM model...")
        print(f"Training samples: {len(X_train_scaled)}")
        print(f"Validation samples: {len(X_val_scaled)}")
        
        self.history = self.model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """
        Predict fatigue scores
        
        Args:
            X: Input sequences (n_samples, sequence_length, n_features)
            
        Returns:
            Predicted fatigue scores (0-100)
        """
        
        X_scaled = self.preprocess_data(X, fit_scaler=False)
        predictions_scaled = self.model.predict(X_scaled, verbose=0)
        
        # Convert back to 0-100 scale
        predictions = predictions_scaled * 100.0
        
        return predictions.flatten()
    
    def predict_future(self, current_sequence, steps_ahead=30):
        """
        Forecast future fatigue assuming current conditions continue
        
        Args:
            current_sequence: Current sequence (sequence_length, n_features)
            steps_ahead: Number of future timesteps to predict
            
        Returns:
            Array of future fatigue predictions
        """
        
        predictions = []
        sequence = current_sequence.copy()
        
        for _ in range(steps_ahead):
            # Predict next timestep
            X_input = sequence[-self.sequence_length:].reshape(1, self.sequence_length, self.n_features)
            next_fatigue = self.predict(X_input)[0]
            predictions.append(next_fatigue)
            
            # For simplicity, assume features stay similar but fatigue increases
            # In real system, you'd model how features evolve
            next_features = sequence[-1].copy()
            sequence = np.vstack([sequence, next_features])
        
        return np.array(predictions)
    
    def save_model(self, filepath='fatigue_model.h5'):
        """Save trained model and scalers"""
        self.model.save(filepath)
        with open(filepath.replace('.h5', '_scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler_X, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='fatigue_model.h5'):
        """Load trained model and scalers"""
        self.model = keras.models.load_model(filepath)
        with open(filepath.replace('.h5', '_scaler.pkl'), 'rb') as f:
            self.scaler_X = pickle.load(f)
        print(f"Model loaded from {filepath}")
    
    def get_summary(self):
        """Print model architecture"""
        if self.model:
            return self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")


if __name__ == "__main__":
    # Test model creation
    model = FatigueLSTMModel(sequence_length=30, n_features=6)
    model.build_model()
    print("\nModel Architecture:")
    model.get_summary()
