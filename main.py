"""
Main entry point for Agentic Edge Deepfake Detection System
Launches the Streamlit UI
"""

import os
import sys
import subprocess
from pathlib import Path


def check_requirements():
    """Check if required packages are installed"""
    try:
        import torch
        import cv2
        import mediapipe
        import librosa
        import streamlit
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e.name}")
        print("\nPlease install requirements using:")
        print("pip install -r requirements.txt")
        return False


def check_models_directory():
    """Check if models directory exists"""
    models_dir = Path(__file__).parent / "models"
    if not models_dir.exists():
        models_dir.mkdir(exist_ok=True)
        print(f"✅ Created models directory: {models_dir}")
    
    # Create README if it doesn't exist
    readme_path = models_dir / "README.md"
    if not readme_path.exists():
        with open(readme_path, 'w') as f:
            f.write("""# Models Directory

This directory contains pretrained ONNX models for deepfake detection.

For this hackathon MVP, we use mock models with random weights.
In production, replace these with:
- MesoNet, EfficientNet, or XceptionNet for video
- RawNet2 or AASIST for audio

The inference code is designed to work with real models once available.
""")
        print(f"✅ Created models README: {readme_path}")


def launch_app():
    """Launch the Streamlit application"""
    print("\n" + "="*60)
    print("🛡️  AGENTIC EDGE DEEPFAKE DETECTION SYSTEM (A-EDDS)")
    print("="*60)
    print("\n📋 Hackathon MVP - Offline Deepfake Detection")
    print("🔒 100% Offline - No Cloud Dependency")
    print("⚡ CPU-Only Inference\n")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check models directory
    check_models_directory()
    
    # Get UI path
    ui_path = Path(__file__).parent / "ui" / "app.py"
    
    if not ui_path.exists():
        print(f"❌ UI file not found: {ui_path}")
        sys.exit(1)
    
    print(f"\n🚀 Launching Streamlit UI...")
    print(f"📁 UI Path: {ui_path}\n")
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ui_path),
            "--server.headless=true",
            "--server.port=8501",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n\n✋ Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching application: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    launch_app()