"""
Audio Deepfake Detector
Detects deepfakes in audio using MFCC features and neural network
"""

import numpy as np
import librosa
import torch
import torch.nn as nn
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class AudioCNN(nn.Module):
    """Lightweight CNN for audio deepfake detection"""
    def __init__(self, input_shape=(40, 128)):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 16))
        self.fc1 = nn.Linear(128 * 5 * 16, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x


class AudioDeepfakeDetector:
    """Main audio deepfake detection class"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize audio detector
        
        Args:
            model_path: Path to pretrained model (optional)
        """
        self.device = torch.device('cpu')
        self.model = AudioCNN().to(self.device)
        self.model.eval()
        
        # Load pretrained weights if available
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✓ Loaded audio model from {model_path}")
            except:
                print(f"⚠ Could not load model from {model_path}, using random weights")
        else:
            print("ℹ Using mock audio model with random weights")
        
        # Audio processing parameters
        self.sample_rate = 16000
        self.n_mfcc = 40
        self.max_duration = 10  # seconds
        self.hop_length = 512
        
    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        Extract MFCC features from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            MFCC feature array
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.max_duration)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length
        )
        
        # Normalize
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        
        return mfcc
    
    def extract_features_from_array(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract MFCC features from audio array
        
        Args:
            audio: Audio signal array
            sr: Sample rate
            
        Returns:
            MFCC feature array
        """
        # Resample if necessary
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        # Truncate or pad to max duration
        max_samples = self.sample_rate * self.max_duration
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < max_samples:
            audio = np.pad(audio, (0, max_samples - len(audio)))
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length
        )
        
        # Normalize
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        
        return mfcc
    
    def preprocess_features(self, mfcc: np.ndarray) -> torch.Tensor:
        """
        Preprocess MFCC features for model input
        
        Args:
            mfcc: MFCC feature array
            
        Returns:
            Preprocessed tensor
        """
        # Pad or truncate to fixed size (40, 128)
        target_width = 128
        
        if mfcc.shape[1] < target_width:
            # Pad with zeros
            pad_width = target_width - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Truncate
            mfcc = mfcc[:, :target_width]
        
        # Convert to tensor (1, H, W)
        mfcc_tensor = torch.from_numpy(mfcc).float().unsqueeze(0).unsqueeze(0)
        
        return mfcc_tensor.to(self.device)
    
    def predict(self, audio_path: str) -> float:
        """
        Predict deepfake probability for audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Deepfake probability (0-1)
        """
        # Extract features
        mfcc = self.extract_features(audio_path)
        
        # Preprocess
        mfcc_tensor = self.preprocess_features(mfcc)
        
        # Inference
        with torch.no_grad():
            output = self.model(mfcc_tensor)
            probability = output.item()
        
        return probability
    
    def predict_from_array(self, audio: np.ndarray, sr: int) -> float:
        """
        Predict deepfake probability from audio array
        
        Args:
            audio: Audio signal array
            sr: Sample rate
            
        Returns:
            Deepfake probability (0-1)
        """
        # Extract features
        mfcc = self.extract_features_from_array(audio, sr)
        
        # Preprocess
        mfcc_tensor = self.preprocess_features(mfcc)
        
        # Inference
        with torch.no_grad():
            output = self.model(mfcc_tensor)
            probability = output.item()
        
        return probability


# Test function
if __name__ == "__main__":
    print("Testing Audio Deepfake Detector...")
    detector = AudioDeepfakeDetector()
    
    # Create dummy audio
    dummy_audio = np.random.randn(16000 * 5)  # 5 seconds
    
    score = detector.predict_from_array(dummy_audio, 16000)
    print(f"Deepfake Score: {score:.4f}")