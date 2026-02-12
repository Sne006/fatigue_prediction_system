"""
Sample CSV Generator
Creates example training session files for dashboard upload
"""

from data_generation import WearableDataGenerator
import argparse

def generate_sample_csv(filename='sample_session.csv', 
                       duration=120, 
                       fatigue_trend='increasing'):
    """
    Generate and save sample training session data
    
    Args:
        filename: Output CSV filename
        duration: Session duration in minutes
        fatigue_trend: 'increasing', 'stable', or 'decreasing'
    """
    
    print(f"Generating {duration}-minute session with {fatigue_trend} fatigue trend...")
    
    generator = WearableDataGenerator(seed=42)
    session_data = generator.generate_session_data(
        duration_minutes=duration,
        fatigue_trend=fatigue_trend
    )
    
    # Save to CSV
    session_data.to_csv(filename, index=False)
    
    print(f"✅ Saved to {filename}")
    print(f"\nSession Statistics:")
    print(f"  Duration: {duration} minutes")
    print(f"  Avg Heart Rate: {session_data['heart_rate'].mean():.1f} bpm")
    print(f"  Avg Muscle Strain: {session_data['muscle_strain'].mean():.1f}/10")
    print(f"  Final Fatigue Score: {session_data['fatigue_score'].iloc[-1]:.1f}%")
    print(f"\nUpload this file in the dashboard's 'Upload CSV' mode!")

def main():
    parser = argparse.ArgumentParser(
        description='Generate sample training session CSV files'
    )
    parser.add_argument(
        '--output', '-o',
        default='sample_session.csv',
        help='Output filename (default: sample_session.csv)'
    )
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=120,
        help='Session duration in minutes (default: 120)'
    )
    parser.add_argument(
        '--trend', '-t',
        choices=['increasing', 'stable', 'decreasing'],
        default='increasing',
        help='Fatigue trend pattern (default: increasing)'
    )
    
    args = parser.parse_args()
    
    generate_sample_csv(
        filename=args.output,
        duration=args.duration,
        fatigue_trend=args.trend
    )

if __name__ == "__main__":
    main()
