"""
Training Script
Generates data, trains LSTM model, and evaluates performance
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data_generation import WearableDataGenerator, create_sequences
from model import FatigueLSTMModel


def train_fatigue_model(n_samples=5000, sequence_length=30, epochs=50):
    """
    Complete training pipeline
    
    Args:
        n_samples: Number of data points to generate
        sequence_length: Length of input sequences
        epochs: Number of training epochs
    """
    
    print("="*60)
    print("LSTM Micro-Muscle Fatigue Prediction System")
    print("="*60)
    
    # Step 1: Generate synthetic data
    print("\n[1/5] Generating synthetic wearable data...")
    generator = WearableDataGenerator(seed=42)
    data = generator.generate_training_data(n_samples=n_samples)
    
    print(f"Generated {len(data)} timesteps")
    print(f"Fatigue distribution:")
    print(f"  Low risk (<40%): {(data['fatigue_score'] < 40).sum()}")
    print(f"  Medium risk (40-70%): {((data['fatigue_score'] >= 40) & (data['fatigue_score'] <= 70)).sum()}")
    print(f"  High risk (>70%): {(data['fatigue_score'] > 70).sum()}")
    
    # Step 2: Create sequences
    print("\n[2/5] Creating time-series sequences...")
    X, y = create_sequences(data, sequence_length=sequence_length)
    
    print(f"Sequence shape: {X.shape}")
    print(f"  - Samples: {X.shape[0]}")
    print(f"  - Timesteps per sequence: {X.shape[1]}")
    print(f"  - Features per timestep: {X.shape[2]}")
    
    # Step 3: Train-test split
    print("\n[3/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} sequences")
    print(f"Validation set: {len(X_val)} sequences")
    print(f"Test set: {len(X_test)} sequences")
    
    # Step 4: Build and train model
    print("\n[4/5] Building LSTM model...")
    model = FatigueLSTMModel(sequence_length=sequence_length, n_features=6)
    model.build_model(lstm_units=[64, 32], dropout_rate=0.2)
    
    print("\nModel Architecture:")
    model.get_summary()
    
    print("\n[5/5] Training model...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=32
    )
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = np.mean(np.abs(y_pred - y_test))
    rmse = np.sqrt(np.mean((y_pred - y_test)**2))
    
    print(f"\nTest Set Performance:")
    print(f"  Mean Absolute Error: {mae:.2f}%")
    print(f"  Root Mean Squared Error: {rmse:.2f}%")
    
    # Accuracy within tolerance
    tolerance_10 = np.mean(np.abs(y_pred - y_test) < 10) * 100
    tolerance_15 = np.mean(np.abs(y_pred - y_test) < 15) * 100
    
    print(f"\nPrediction Accuracy:")
    print(f"  Within ±10%: {tolerance_10:.1f}%")
    print(f"  Within ±15%: {tolerance_15:.1f}%")
    
    # Save model
    print("\n" + "="*60)
    print("Saving model...")
    model.save_model('fatigue_model.h5')
    
    # Plot training history
    plot_training_history(history)
    
    # Plot predictions vs actual
    plot_predictions(y_test, y_pred)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    return model, history


def plot_training_history(history):
    """Plot training and validation loss"""
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Model Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # MAE plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.title('Model MAE Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("Training history plot saved as 'training_history.png'")
    plt.close()


def plot_predictions(y_true, y_pred):
    """Plot actual vs predicted fatigue scores"""
    plt.figure(figsize=(12, 5))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    plt.plot([0, 100], [0, 100], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Fatigue Score')
    plt.ylabel('Predicted Fatigue Score')
    plt.title('Prediction Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    
    # Time series comparison (first 200 samples)
    plt.subplot(1, 2, 2)
    samples_to_plot = min(200, len(y_true))
    x_axis = range(samples_to_plot)
    plt.plot(x_axis, y_true[:samples_to_plot], label='Actual', alpha=0.7, linewidth=2)
    plt.plot(x_axis, y_pred[:samples_to_plot], label='Predicted', alpha=0.7, linewidth=2)
    plt.xlabel('Sample Index')
    plt.ylabel('Fatigue Score')
    plt.title('Prediction vs Actual (First 200 Samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=150, bbox_inches='tight')
    print("Prediction results plot saved as 'prediction_results.png'")
    plt.close()


if __name__ == "__main__":
    # Train the model
    model, history = train_fatigue_model(
        n_samples=5000,
        sequence_length=30,
        epochs=50
    )
