#!/usr/bin/env python3
"""
Quick Start Script
Automated setup and launch for the Fatigue Prediction System
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command with status updates"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(result.stdout)
        print(f"✅ {description} - COMPLETE")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(e.stderr)
        return False

def main():
    """Main setup and launch workflow"""
    
    print("""
    ⚡ LSTM Micro-Muscle Fatigue Prediction System
    ============================================
    Hackathon Quick Start Setup
    """)
    
    # Step 1: Check Python version
    print(f"\n📍 Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required")
        sys.exit(1)
    
    # Step 2: Install dependencies
    print("\n" + "="*60)
    response = input("📦 Install dependencies? (y/n): ").lower()
    if response == 'y':
        if not run_command("pip install -r requirements.txt", 
                          "Installing dependencies"):
            print("\n⚠️ Dependency installation failed. Try manually:")
            print("pip install -r requirements.txt")
            sys.exit(1)
    
    # Step 3: Train model (if not exists)
    print("\n" + "="*60)
    if not os.path.exists('fatigue_model.h5'):
        print("🧠 No trained model found.")
        response = input("Train LSTM model now? (Recommended, ~5-10 min) (y/n): ").lower()
        
        if response == 'y':
            if not run_command("python train.py", "Training LSTM model"):
                print("\n⚠️ Model training failed")
                sys.exit(1)
        else:
            print("\n⚠️ Warning: Dashboard will run with untrained model")
    else:
        print("✅ Found existing trained model: fatigue_model.h5")
    
    # Step 4: Launch dashboard
    print("\n" + "="*60)
    print("🚀 LAUNCHING DASHBOARD")
    print("="*60)
    print("\n📱 Dashboard will open at: http://localhost:8501")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run("streamlit run app.py", shell=True, check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")
        print("\nTry manually: streamlit run app.py")

if __name__ == "__main__":
    main()
