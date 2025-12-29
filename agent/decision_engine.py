"""
Agent Decision Engine
Implements observe-reason-act loop for deepfake detection
"""

import time
from typing import Dict, List, Tuple
from datetime import datetime


class DecisionEngine:
    """Agentic decision engine for deepfake detection"""
    
    def __init__(
        self,
        low_threshold: float = 0.3,
        medium_threshold: float = 0.6,
        high_threshold: float = 0.8
    ):
        """
        Initialize decision engine with risk thresholds
        
        Args:
            low_threshold: Threshold for low risk (0-1)
            medium_threshold: Threshold for medium risk (0-1)
            high_threshold: Threshold for high risk (0-1)
        """
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        
        # History for reasoning
        self.detection_history: List[Dict] = []
        self.max_history = 100
        
        print(f"✓ Decision Engine initialized")
        print(f"  Thresholds - Low: {low_threshold}, Medium: {medium_threshold}, High: {high_threshold}")
    
    def observe(self, fusion_result: Dict) -> Dict:
        """
        OBSERVE: Gather information from fusion module
        
        Args:
            fusion_result: Result from multimodal fusion
            
        Returns:
            Observation dictionary with metadata
        """
        observation = {
            'timestamp': datetime.now().isoformat(),
            'fused_score': fusion_result['fused_score'],
            'video_score': fusion_result['video_score'],
            'audio_score': fusion_result['audio_score'],
            'confidence': fusion_result['confidence'],
            'modalities': fusion_result['modalities_available']
        }
        
        # Add to history
        self.detection_history.append(observation)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
        
        return observation
    
    def reason(self, observation: Dict) -> Dict:
        """
        REASON: Analyze observation and determine risk level
        
        Args:
            observation: Observation from observe step
            
        Returns:
            Reasoning result with risk level and explanation
        """
        score = observation['fused_score']
        confidence = observation['confidence']
        modalities = observation['modalities']
        
        # Determine risk level
        if score >= self.high_threshold:
            risk_level = "HIGH"
            color = "🔴"
            explanation = f"Deepfake probability is very high ({score:.2%})"
        elif score >= self.medium_threshold:
            risk_level = "MEDIUM"
            color = "🟡"
            explanation = f"Deepfake probability is moderate ({score:.2%})"
        elif score >= self.low_threshold:
            risk_level = "LOW"
            color = "🟢"
            explanation = f"Minor deepfake indicators detected ({score:.2%})"
        else:
            risk_level = "AUTHENTIC"
            color = "✅"
            explanation = f"Content appears authentic ({score:.2%})"
        
        # Adjust confidence based on available modalities
        if len(modalities) == 2:
            confidence_text = "High confidence (audio + video)"
        elif len(modalities) == 1:
            confidence_text = f"Moderate confidence ({modalities[0]} only)"
        else:
            confidence_text = "Low confidence (no modalities detected)"
        
        reasoning = {
            'risk_level': risk_level,
            'risk_color': color,
            'score': score,
            'confidence': confidence,
            'confidence_text': confidence_text,
            'explanation': explanation,
            'modalities_used': modalities
        }
        
        return reasoning
    
    def act(self, reasoning: Dict) -> Dict:
        """
        ACT: Take action based on reasoning
        
        Args:
            reasoning: Reasoning result
            
        Returns:
            Action dictionary with recommended actions
        """
        risk_level = reasoning['risk_level']
        
        # Define actions based on risk level
        if risk_level == "HIGH":
            actions = {
                'alert': True,
                'alert_level': "CRITICAL",
                'message': "⚠️ HIGH RISK: This content is likely a deepfake. Do not trust or share.",
                'recommendations': [
                    "Do not share this content",
                    "Report to platform moderators",
                    "Verify through official sources",
                    "Alert content creator if impersonated"
                ],
                'should_block': True
            }
        elif risk_level == "MEDIUM":
            actions = {
                'alert': True,
                'alert_level': "WARNING",
                'message': "⚠️ MEDIUM RISK: This content shows signs of manipulation. Verify before trusting.",
                'recommendations': [
                    "Verify content authenticity",
                    "Check multiple sources",
                    "Exercise caution when sharing",
                    "Look for original source"
                ],
                'should_block': False
            }
        elif risk_level == "LOW":
            actions = {
                'alert': True,
                'alert_level': "CAUTION",
                'message': "ℹ️ LOW RISK: Minor manipulation indicators detected. Proceed with caution.",
                'recommendations': [
                    "Content likely authentic with minor artifacts",
                    "Normal compression artifacts may be present",
                    "Continue monitoring if suspicious"
                ],
                'should_block': False
            }
        else:  # AUTHENTIC
            actions = {
                'alert': False,
                'alert_level': "SAFE",
                'message': "✅ AUTHENTIC: No significant deepfake indicators detected.",
                'recommendations': [
                    "Content appears authentic",
                    "Standard verification practices still recommended"
                ],
                'should_block': False
            }
        
        return actions
    
    def process(self, fusion_result: Dict) -> Dict:
        """
        Complete observe-reason-act loop
        
        Args:
            fusion_result: Result from multimodal fusion
            
        Returns:
            Complete decision result
        """
        # OBSERVE
        observation = self.observe(fusion_result)
        
        # REASON
        reasoning = self.reason(observation)
        
        # ACT
        actions = self.act(reasoning)
        
        # Combine all results
        decision = {
            'observation': observation,
            'reasoning': reasoning,
            'actions': actions,
            'timestamp': observation['timestamp']
        }
        
        return decision
    
    def get_statistics(self) -> Dict:
        """
        Get statistics from detection history
        
        Returns:
            Statistics dictionary
        """
        if not self.detection_history:
            return {
                'total_detections': 0,
                'average_score': 0.0,
                'high_risk_count': 0,
                'medium_risk_count': 0,
                'low_risk_count': 0
            }
        
        scores = [h['fused_score'] for h in self.detection_history]
        
        high_risk = sum(1 for s in scores if s >= self.high_threshold)
        medium_risk = sum(1 for s in scores if self.medium_threshold <= s < self.high_threshold)
        low_risk = sum(1 for s in scores if self.low_threshold <= s < self.medium_threshold)
        
        return {
            'total_detections': len(self.detection_history),
            'average_score': sum(scores) / len(scores),
            'high_risk_count': high_risk,
            'medium_risk_count': medium_risk,
            'low_risk_count': low_risk
        }


# Test function
if __name__ == "__main__":
    print("Testing Decision Engine...")
    
    engine = DecisionEngine()
    
    # Test different risk levels
    test_cases = [
        {'fused_score': 0.9, 'video_score': 0.9, 'audio_score': 0.85, 'confidence': 1.0, 'modalities_available': ['video', 'audio']},
        {'fused_score': 0.5, 'video_score': 0.5, 'audio_score': 0.5, 'confidence': 1.0, 'modalities_available': ['video', 'audio']},
        {'fused_score': 0.2, 'video_score': 0.2, 'audio_score': 0.15, 'confidence': 1.0, 'modalities_available': ['video', 'audio']},
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        decision = engine.process(test)
        print(f"Risk Level: {decision['reasoning']['risk_level']}")
        print(f"Message: {decision['actions']['message']}")
    
    print(f"\nStatistics: {engine.get_statistics()}")