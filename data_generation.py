"""
Data Generation Module
Generates synthetic wearable sensor time-series data for fatigue prediction
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class WearableDataGenerator:
    """
    Simulates realistic wearable sensor data with fatigue progression patterns
    """
    
    def __init__(self, seed=42):
        np.random.seed(seed)
    
    def generate_training_data(self, n_samples=5000, sequence_length=30):
        """
        Generate synthetic training data with realistic fatigue patterns
        
        Args:
            n_samples: Number of time steps to generate
            sequence_length: Length of sequences for LSTM input
            
        Returns:
            DataFrame with sensor data and fatigue labels
        """
        
        data = []
        base_time = datetime.now()
        
        # Initialize variables for progressive fatigue
        cumulative_strain = 0
        recovery_deficit = 0
        
        for i in range(n_samples):
            # Time progression
            timestamp = base_time + timedelta(minutes=i)
            
            # Simulate circadian rhythm and training sessions
            hour = timestamp.hour
            is_training = (8 <= hour <= 12) or (14 <= hour <= 18)
            is_rest_period = (22 <= hour or hour <= 6)
            
            # Sleep hours (varies by day)
            day_cycle = i % 1440  # Minutes in a day
            if day_cycle == 0:
                daily_sleep = np.random.normal(7, 1.5)
                daily_sleep = np.clip(daily_sleep, 4, 9)
            else:
                daily_sleep = 7  # Default
            
            # Heart Rate simulation
            base_hr = 65
            if is_training:
                hr = np.random.normal(140, 15)
            elif is_rest_period:
                hr = np.random.normal(55, 5)
            else:
                hr = np.random.normal(75, 10)
            hr = np.clip(hr, 50, 190)
            
            # Acceleration (movement intensity)
            if is_training:
                acceleration = np.random.normal(8, 2)
            elif is_rest_period:
                acceleration = np.random.normal(0.5, 0.2)
            else:
                acceleration = np.random.normal(2, 1)
            acceleration = np.clip(acceleration, 0, 12)
            
            # Muscle Strain Index
            # Higher during training, accumulates with fatigue
            if is_training:
                strain = np.random.normal(7, 1.5) + cumulative_strain * 0.01
            else:
                strain = np.random.normal(2, 0.5)
            strain = np.clip(strain, 0, 10)
            
            # Training Load (cumulative)
            if is_training:
                training_load = np.random.normal(75, 15)
                cumulative_strain += 0.5
            else:
                training_load = np.random.normal(20, 5)
                # Recovery during rest
                if is_rest_period and daily_sleep > 6:
                    cumulative_strain = max(0, cumulative_strain - 0.3)
            training_load = np.clip(training_load, 0, 100)
            
            # Recovery Time (hours since last training)
            if is_training:
                recovery_time = np.random.normal(2, 1)
            else:
                recovery_time = np.random.normal(8, 3)
            recovery_time = np.clip(recovery_time, 0, 24)
            
            # Calculate Fatigue Score (0-100)
            fatigue_score = self._calculate_fatigue(
                hr, acceleration, strain, training_load, 
                recovery_time, daily_sleep, cumulative_strain
            )
            
            # Binary fatigue label (1 if high risk)
            fatigue_label = 1 if fatigue_score > 60 else 0
            
            data.append({
                'timestamp': timestamp,
                'heart_rate': hr,
                'acceleration': acceleration,
                'muscle_strain': strain,
                'training_load': training_load,
                'recovery_time': recovery_time,
                'sleep_hours': daily_sleep,
                'fatigue_score': fatigue_score,
                'fatigue_label': fatigue_label
            })
        
        df = pd.DataFrame(data)
        return df
    
    def _calculate_fatigue(self, hr, accel, strain, load, recovery, sleep, cumulative):
        """
        Calculate fatigue score based on physiological factors
        
        Fatigue increases with:
        - High heart rate variability
        - High movement intensity
        - High muscle strain
        - High training load
        - Low recovery time
        - Poor sleep
        - Cumulative strain
        """
        
        # Normalize inputs
        hr_factor = min(hr / 180, 1.0) * 20  # Max 20 points
        accel_factor = min(accel / 10, 1.0) * 15  # Max 15 points
        strain_factor = min(strain / 10, 1.0) * 25  # Max 25 points
        load_factor = min(load / 100, 1.0) * 20  # Max 20 points
        
        # Recovery penalty
        recovery_penalty = max(0, (8 - recovery) / 8) * 10  # Max 10 points
        
        # Sleep penalty
        sleep_penalty = max(0, (7 - sleep) / 7) * 15  # Max 15 points
        
        # Cumulative strain
        cumulative_factor = min(cumulative / 50, 1.0) * 10  # Max 10 points
        
        # Total fatigue score
        fatigue = (hr_factor + accel_factor + strain_factor + load_factor + 
                   recovery_penalty + sleep_penalty + cumulative_factor)
        
        # Add some random noise
        fatigue += np.random.normal(0, 3)
        
        return np.clip(fatigue, 0, 100)
    
    def generate_session_data(self, duration_minutes=120, fatigue_trend='increasing'):
        """
        Generate a single training session for real-time prediction
        
        Args:
            duration_minutes: Length of session
            fatigue_trend: 'increasing', 'stable', or 'decreasing'
        """
        
        data = []
        base_time = datetime.now()
        
        for i in range(duration_minutes):
            timestamp = base_time + timedelta(minutes=i)
            
            # Progressive fatigue during session
            if fatigue_trend == 'increasing':
                fatigue_modifier = i / duration_minutes * 30
            elif fatigue_trend == 'decreasing':
                fatigue_modifier = (duration_minutes - i) / duration_minutes * 20
            else:
                fatigue_modifier = 15
            
            # Training session data
            hr = np.random.normal(145 + fatigue_modifier * 0.5, 10)
            hr = np.clip(hr, 100, 190)
            
            acceleration = np.random.normal(7 + fatigue_modifier * 0.1, 1.5)
            acceleration = np.clip(acceleration, 3, 12)
            
            strain = np.random.normal(6 + fatigue_modifier * 0.15, 1)
            strain = np.clip(strain, 3, 10)
            
            training_load = np.random.normal(70 + fatigue_modifier * 0.5, 10)
            training_load = np.clip(training_load, 40, 100)
            
            recovery_time = np.clip(4 - i / 30, 0.5, 12)
            sleep_hours = 6.5
            
            fatigue_score = self._calculate_fatigue(
                hr, acceleration, strain, training_load,
                recovery_time, sleep_hours, i * 0.3
            )
            
            data.append({
                'timestamp': timestamp,
                'heart_rate': hr,
                'acceleration': acceleration,
                'muscle_strain': strain,
                'training_load': training_load,
                'recovery_time': recovery_time,
                'sleep_hours': sleep_hours,
                'fatigue_score': fatigue_score
            })
        
        return pd.DataFrame(data)


def create_sequences(data, sequence_length=30):
    """
    Create time-series sequences for LSTM training
    
    Args:
        data: DataFrame with features
        sequence_length: Number of timesteps per sequence
        
    Returns:
        X: sequences of features (n_samples, sequence_length, n_features)
        y: fatigue labels or scores
    """
    
    feature_columns = ['heart_rate', 'acceleration', 'muscle_strain', 
                      'training_load', 'recovery_time', 'sleep_hours']
    
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        # Get sequence of features
        sequence = data[feature_columns].iloc[i:i+sequence_length].values
        X.append(sequence)
        
        # Get target (fatigue score at end of sequence)
        target = data['fatigue_score'].iloc[i + sequence_length]
        y.append(target)
    
    return np.array(X), np.array(y)


if __name__ == "__main__":
    # Test data generation
    generator = WearableDataGenerator()
    df = generator.generate_training_data(n_samples=1000)
    print("Generated training data:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nFatigue distribution:")
    print(df['fatigue_label'].value_counts())
