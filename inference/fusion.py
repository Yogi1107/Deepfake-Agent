"""
Multimodal Fusion Module
Combines audio and video deepfake scores using weighted averaging
"""

import numpy as np
from typing import Optional, Dict


class MultimodalFusion:
    """Fuses audio and video deepfake detection scores"""
    
    def __init__(self, video_weight: float = 0.6, audio_weight: float = 0.4):
        """
        Initialize fusion module
        
        Args:
            video_weight: Weight for video score (default 0.6)
            audio_weight: Weight for audio score (default 0.4)
        """
        # Ensure weights sum to 1.0
        total = video_weight + audio_weight
        self.video_weight = video_weight / total
        self.audio_weight = audio_weight / total
        
        print(f"ℹ Fusion weights - Video: {self.video_weight:.2f}, Audio: {self.audio_weight:.2f}")
    
    def fuse_scores(
        self,
        video_score: Optional[float],
        audio_score: Optional[float]
    ) -> Dict[str, float]:
        """
        Fuse video and audio scores
        
        Args:
            video_score: Video deepfake probability (0-1) or None
            audio_score: Audio deepfake probability (0-1) or None
            
        Returns:
            Dictionary with fused score and individual scores
        """
        # Handle missing modalities
        if video_score is None and audio_score is None:
            return {
                'fused_score': 0.0,
                'video_score': 0.0,
                'audio_score': 0.0,
                'confidence': 0.0,
                'modalities_available': []
            }
        
        available_modalities = []
        
        if video_score is None:
            # Audio only
            fused_score = audio_score
            confidence = 0.5  # Lower confidence with single modality
            available_modalities.append('audio')
        elif audio_score is None:
            # Video only
            fused_score = video_score
            confidence = 0.5  # Lower confidence with single modality
            available_modalities.append('video')
        else:
            # Both modalities available
            fused_score = (
                self.video_weight * video_score +
                self.audio_weight * audio_score
            )
            confidence = 1.0  # High confidence with both modalities
            available_modalities = ['video', 'audio']
        
        return {
            'fused_score': fused_score,
            'video_score': video_score if video_score is not None else 0.0,
            'audio_score': audio_score if audio_score is not None else 0.0,
            'confidence': confidence,
            'modalities_available': available_modalities
        }
    
    def adjust_weights(self, video_weight: float, audio_weight: float):
        """
        Dynamically adjust fusion weights
        
        Args:
            video_weight: New video weight
            audio_weight: New audio weight
        """
        total = video_weight + audio_weight
        self.video_weight = video_weight / total
        self.audio_weight = audio_weight / total
        
        print(f"✓ Updated fusion weights - Video: {self.video_weight:.2f}, Audio: {self.audio_weight:.2f}")


# Test function
if __name__ == "__main__":
    print("Testing Multimodal Fusion...")
    
    fusion = MultimodalFusion()
    
    # Test case 1: Both modalities
    result = fusion.fuse_scores(video_score=0.8, audio_score=0.6)
    print(f"Both modalities: {result}")
    
    # Test case 2: Video only
    result = fusion.fuse_scores(video_score=0.7, audio_score=None)
    print(f"Video only: {result}")
    
    # Test case 3: Audio only
    result = fusion.fuse_scores(video_score=None, audio_score=0.5)
    print(f"Audio only: {result}")