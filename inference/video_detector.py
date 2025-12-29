"""
Video Deepfake Detector
Detects deepfakes in video frames using face detection and CNN inference
"""

import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
from typing import Optional, Tuple

class SimpleCNN(nn.Module):
    """Lightweight CNN for video deepfake detection (mock model)"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


class VideoDeepfakeDetector:
    """Main video deepfake detection class"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize video detector with face detection and CNN model
        
        Args:
            model_path: Path to pretrained model (optional)
        """
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        
        # Initialize CNN model
        self.device = torch.device('cpu')
        self.model = SimpleCNN().to(self.device)
        self.model.eval()
        
        # Load pretrained weights if available
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✓ Loaded video model from {model_path}")
            except:
                print(f"⚠ Could not load model from {model_path}, using random weights")
        else:
            print("ℹ Using mock video model with random weights")
        
        self.input_size = (112, 112)
        
    def detect_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and extract face from frame
        
        Args:
            frame: Input video frame (BGR)
            
        Returns:
            Cropped face image or None if no face detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(rgb_frame)
        
        if not results.detections:
            return None
        
        # Get first face detection
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape
        
        # Convert relative coordinates to absolute
        x = int(bboxC.xmin * w)
        y = int(bboxC.ymin * h)
        width = int(bboxC.width * w)
        height = int(bboxC.height * h)
        
        # Add padding and ensure within bounds
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        width = min(w - x, width + 2 * padding)
        height = min(h - y, height + 2 * padding)
        
        # Crop face
        face = frame[y:y+height, x:x+width]
        
        return face if face.size > 0 else None
    
    def preprocess_face(self, face: np.ndarray) -> torch.Tensor:
        """
        Preprocess face for model input
        
        Args:
            face: Cropped face image
            
        Returns:
            Preprocessed tensor
        """
        # Resize
        face_resized = cv2.resize(face, self.input_size)
        
        # Normalize to [0, 1]
        face_norm = face_resized.astype(np.float32) / 255.0
        
        # Convert to tensor (C, H, W)
        face_tensor = torch.from_numpy(face_norm).permute(2, 0, 1).unsqueeze(0)
        
        return face_tensor.to(self.device)
    
    def predict(self, frame: np.ndarray) -> Tuple[float, bool]:
        """
        Predict deepfake probability for a frame
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (deepfake_probability, face_detected)
        """
        # Detect face
        face = self.detect_face(frame)
        
        if face is None:
            return 0.0, False
        
        # Preprocess
        face_tensor = self.preprocess_face(face)
        
        # Inference
        with torch.no_grad():
            output = self.model(face_tensor)
            probability = output.item()
        
        return probability, True
    
    def process_video(self, video_path: str, max_frames: int = 30) -> Tuple[float, int, int]:
        """
        Process video file and return average deepfake score
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process
            
        Returns:
            Tuple of (average_score, frames_processed, total_frames)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_scores = []
        frames_processed = 0
        
        # Sample frames uniformly
        frame_interval = max(1, total_frames // max_frames)
        
        frame_idx = 0
        while cap.isOpened() and frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                score, detected = self.predict(frame)
                if detected:
                    frame_scores.append(score)
                frames_processed += 1
            
            frame_idx += 1
        
        cap.release()
        
        # Calculate average score
        avg_score = np.mean(frame_scores) if frame_scores else 0.0
        
        return avg_score, frames_processed, total_frames
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()


# Test function
if __name__ == "__main__":
    print("Testing Video Deepfake Detector...")
    detector = VideoDeepfakeDetector()
    
    # Create a dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    score, detected = detector.predict(dummy_frame)
    print(f"Deepfake Score: {score:.4f}, Face Detected: {detected}")