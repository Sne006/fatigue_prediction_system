"""
System Test Script
Validates all components of the fatigue prediction system
"""

import numpy as np
import sys

def test_data_generation():
    """Test synthetic data generation"""
    print("\n🧪 Testing Data Generation...")
    try:
        from data_generation import WearableDataGenerator, create_sequences
        
        generator = WearableDataGenerator(seed=42)
        data = generator.generate_training_data(n_samples=1000)
        
        assert len(data) == 1000, "Data length mismatch"
        assert 'fatigue_score' in data.columns, "Missing fatigue_score column"
        assert data['fatigue_score'].min() >= 0, "Invalid fatigue score (negative)"
        assert data['fatigue_score'].max() <= 100, "Invalid fatigue score (>100)"
        
        # Test sequences
        X, y = create_sequences(data, sequence_length=30)
        assert X.shape[1] == 30, "Sequence length mismatch"
        assert X.shape[2] == 6, "Feature count mismatch"
        
        print("   ✅ Data generation working correctly")
        return True
    except Exception as e:
        print(f"   ❌ Data generation failed: {e}")
        return False

def test_model_creation():
    """Test LSTM model architecture"""
    print("\n🧪 Testing Model Architecture...")
    try:
        from model import FatigueLSTMModel
        
        model = FatigueLSTMModel(sequence_length=30, n_features=6)
        model.build_model(lstm_units=[64, 32], dropout_rate=0.2)
        
        # Check model structure
        assert model.model is not None, "Model not built"
        assert len(model.model.layers) > 0, "No layers in model"
        
        # Test prediction shape
        X_test = np.random.randn(10, 30, 6)
        predictions = model.predict(X_test)
        assert len(predictions) == 10, "Prediction shape mismatch"
        
        print("   ✅ Model architecture working correctly")
        return True
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return False

def test_training_pipeline():
    """Test model training with small dataset"""
    print("\n🧪 Testing Training Pipeline...")
    try:
        from data_generation import WearableDataGenerator, create_sequences
        from model import FatigueLSTMModel
        from sklearn.model_selection import train_test_split
        
        # Generate small dataset
        generator = WearableDataGenerator(seed=42)
        data = generator.generate_training_data(n_samples=500)
        X, y = create_sequences(data, sequence_length=30)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Build and train
        model = FatigueLSTMModel(sequence_length=30, n_features=6)
        model.build_model()
        
        # Quick training (2 epochs for testing)
        history = model.train(
            X_train[:100], y_train[:100],
            X_test[:20], y_test[:20],
            epochs=2,
            batch_size=16
        )
        
        assert 'loss' in history.history, "Training history missing"
        
        print("   ✅ Training pipeline working correctly")
        return True
    except Exception as e:
        print(f"   ❌ Training pipeline failed: {e}")
        return False

def test_prediction():
    """Test prediction and forecasting"""
    print("\n🧪 Testing Prediction System...")
    try:
        from data_generation import WearableDataGenerator
        from model import FatigueLSTMModel
        
        # Create model and generate test data
        model = FatigueLSTMModel(sequence_length=30, n_features=6)
        model.build_model()
        
        generator = WearableDataGenerator(seed=42)
        session = generator.generate_session_data(60)
        
        # Prepare data
        feature_cols = ['heart_rate', 'acceleration', 'muscle_strain', 
                       'training_load', 'recovery_time', 'sleep_hours']
        sequence = session[feature_cols].iloc[-30:].values
        X = sequence.reshape(1, 30, 6)
        
        # Test current prediction
        current_fatigue = model.predict(X)
        assert len(current_fatigue) == 1, "Prediction shape mismatch"
        assert 0 <= current_fatigue[0] <= 100, "Prediction out of range"
        
        # Test future forecasting
        future_predictions = model.predict_future(sequence, steps_ahead=10)
        assert len(future_predictions) == 10, "Forecast length mismatch"
        
        print("   ✅ Prediction system working correctly")
        return True
    except Exception as e:
        print(f"   ❌ Prediction system failed: {e}")
        return False

def test_imports():
    """Test all required imports"""
    print("\n🧪 Testing Dependencies...")
    required_packages = [
        'tensorflow',
        'numpy',
        'pandas',
        'matplotlib',
        'plotly',
        'streamlit',
        'sklearn'
    ]
    
    all_imported = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - NOT INSTALLED")
            all_imported = False
    
    return all_imported

def run_all_tests():
    """Run complete test suite"""
    print("="*60)
    print("FATIGUE PREDICTION SYSTEM - TEST SUITE")
    print("="*60)
    
    results = {
        'Dependencies': test_imports(),
        'Data Generation': test_data_generation(),
        'Model Architecture': test_model_creation(),
        'Training Pipeline': test_training_pipeline(),
        'Prediction System': test_prediction()
    }
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("System is ready for deployment")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("Please fix issues before proceeding")
    print("="*60 + "\n")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
