"""
Streamlit UI for Agentic Edge Deepfake Detection System
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.video_detector import VideoDeepfakeDetector
from inference.audio_detector import AudioDeepfakeDetector
from inference.fusion import MultimodalFusion
from agent.decision_engine import DecisionEngine


# Page config
st.set_page_config(
    page_title="A-EDDS: Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .risk-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: bold;
        text-align: center;
    }
    .risk-high {
        background-color: #ffebee;
        border: 2px solid #f44336;
        color: #c62828;
    }
    .risk-medium {
        background-color: #fff8e1;
        border: 2px solid #ffc107;
        color: #f57f17;
    }
    .risk-low {
        background-color: #e8f5e9;
        border: 2px
            color: #2e7d32;
    }
    .risk-authentic {
        background-color: #e3f2fd;
        border: 2px solid #2196f3;
        color: #1565c0;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .offline-badge {
        background-color: #4caf50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load all detection models (cached)"""
    video_detector = VideoDeepfakeDetector()
    audio_detector = AudioDeepfakeDetector()
    fusion = MultimodalFusion(video_weight=0.6, audio_weight=0.4)
    decision_engine = DecisionEngine(
        low_threshold=0.3,
        medium_threshold=0.6,
        high_threshold=0.8
    )
    return video_detector, audio_detector, fusion, decision_engine


def process_uploaded_video(video_file, video_detector):
    """Process uploaded video file"""
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Process video
        with st.spinner('🎥 Analyzing video frames...'):
            avg_score, frames_processed, total_frames = video_detector.process_video(
                tmp_path, max_frames=30
            )
        
        return avg_score, frames_processed, total_frames
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def process_uploaded_audio(audio_file, audio_detector):
    """Process uploaded audio file"""
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Process audio
        with st.spinner('🎵 Analyzing audio features...'):
            score = audio_detector.predict(tmp_path)
        
        return score
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def display_risk_assessment(decision):
    """Display risk assessment results"""
    reasoning = decision['reasoning']
    actions = decision['actions']
    
    # Risk level box
    risk_level = reasoning['risk_level']
    risk_class = f"risk-{risk_level.lower()}"
    
    st.markdown(f"""
        <div class="risk-box {risk_class}">
            <h2>{reasoning['risk_color']} RISK LEVEL: {risk_level}</h2>
            <p style="font-size: 1.1rem; margin-top: 1rem;">{reasoning['explanation']}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Alert message
    if actions['alert']:
        st.warning(actions['message'])
    else:
        st.success(actions['message'])
    
    # Metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Deepfake Score",
            f"{reasoning['score']:.1%}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Confidence",
            reasoning['confidence_text'],
            delta=None
        )
    
    with col3:
        modalities = ", ".join(reasoning['modalities_used']) if reasoning['modalities_used'] else "None"
        st.metric(
            "Modalities Used",
            modalities.title(),
            delta=None
        )
    
    # Recommendations
    with st.expander("📋 Recommended Actions", expanded=actions['alert']):
        for rec in actions['recommendations']:
            st.write(f"• {rec}")


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<p class="main-header">🛡️ A-EDDS: Agentic Edge Deepfake Detector</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Deepfake Detection Running 100% Offline</p>', unsafe_allow_html=True)
    
    # Offline badge
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<center><span class="offline-badge">🔒 100% OFFLINE - NO CLOUD</span></center>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load models
    try:
        video_detector, audio_detector, fusion, decision_engine = load_models()
        st.sidebar.success("✅ All models loaded successfully")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Detection mode
    detection_mode = st.sidebar.selectbox(
        "Detection Mode",
        ["Video + Audio", "Video Only", "Audio Only"]
    )
    
    # Fusion weights (if multimodal)
    if detection_mode == "Video + Audio":
        st.sidebar.subheader("Fusion Weights")
        video_weight = st.sidebar.slider("Video Weight", 0.0, 1.0, 0.6, 0.1)
        audio_weight = 1.0 - video_weight
        st.sidebar.write(f"Audio Weight: {audio_weight:.1f}")
        fusion.adjust_weights(video_weight, audio_weight)
    
    # Risk thresholds
    st.sidebar.subheader("Risk Thresholds")
    low_thresh = st.sidebar.slider("Low Risk", 0.0, 1.0, 0.3, 0.05)
    medium_thresh = st.sidebar.slider("Medium Risk", 0.0, 1.0, 0.6, 0.05)
    high_thresh = st.sidebar.slider("High Risk", 0.0, 1.0, 0.8, 0.05)
    
    decision_engine.low_threshold = low_thresh
    decision_engine.medium_threshold = medium_thresh
    decision_engine.high_threshold = high_thresh
    
    # About section
    with st.sidebar.expander("ℹ️ About A-EDDS"):
        st.write("""
        **Agentic Edge Deepfake Detection System**
        
        - 🎥 Video deepfake detection using facial analysis
        - 🎵 Audio deepfake detection using MFCC features
        - 🤖 Agentic decision-making (Observe-Reason-Act)
        - 🔒 100% offline, no cloud dependency
        - ⚡ Real-time inference on CPU
        
        Built for hackathon demo purposes.
        """)
    
    # Main content
    st.header("📤 Upload Content for Analysis")
    
    # File uploaders based on mode
    video_file = None
    audio_file = None
    
    if detection_mode in ["Video + Audio", "Video Only"]:
        video_file = st.file_uploader(
            "Upload Video File",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to analyze for deepfakes"
        )
    
    if detection_mode in ["Video + Audio", "Audio Only"]:
        audio_file = st.file_uploader(
            "Upload Audio File",
            type=['wav', 'mp3', 'ogg', 'flac'],
            help="Upload an audio file to analyze for voice deepfakes"
        )
    
    # Analyze button
    if st.button("🔍 Analyze Content", type="primary", use_container_width=True):
        
        if not video_file and not audio_file:
            st.warning("Please upload at least one file (video or audio)")
            return
        
        video_score = None
        audio_score = None
        
        # Process video
        if video_file and detection_mode in ["Video + Audio", "Video Only"]:
            try:
                video_score, frames_processed, total_frames = process_uploaded_video(
                    video_file, video_detector
                )
                st.success(f"✅ Video processed: {frames_processed} frames analyzed")
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
        
        # Process audio
        if audio_file and detection_mode in ["Video + Audio", "Audio Only"]:
            try:
                audio_score = process_uploaded_audio(audio_file, audio_detector)
                st.success(f"✅ Audio processed successfully")
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
        
        # Fusion and decision
        if video_score is not None or audio_score is not None:
            st.markdown("---")
            st.header("📊 Analysis Results")
            
            # Fuse scores
            fusion_result = fusion.fuse_scores(video_score, audio_score)
            
            # Display individual scores
            col1, col2 = st.columns(2)
            with col1:
                if video_score is not None:
                    st.metric("🎥 Video Deepfake Score", f"{video_score:.1%}")
                else:
                    st.metric("🎥 Video Deepfake Score", "N/A")
            
            with col2:
                if audio_score is not None:
                    st.metric("🎵 Audio Deepfake Score", f"{audio_score:.1%}")
                else:
                    st.metric("🎵 Audio Deepfake Score", "N/A")
            
            st.markdown("---")
            
            # Agent decision
            with st.spinner('🤖 Agent reasoning...'):
                decision = decision_engine.process(fusion_result)
            
            # Display risk assessment
            display_risk_assessment(decision)
            
            # Show agent reasoning process
            with st.expander("🤖 View Agent Reasoning Process", expanded=False):
                st.subheader("1️⃣ OBSERVE")
                st.json(decision['observation'])
                
                st.subheader("2️⃣ REASON")
                st.json(decision['reasoning'])
                
                st.subheader("3️⃣ ACT")
                st.json(decision['actions'])
    
    # Statistics
    st.markdown("---")
    stats = decision_engine.get_statistics()
    
    if stats['total_detections'] > 0:
        st.header("📈 Detection Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Detections", stats['total_detections'])
        
        with col2:
            st.metric("Avg Score", f"{stats['average_score']:.1%}")
        
        with col3:
            st.metric("High Risk", stats['high_risk_count'])
        
        with col4:
            st.metric("Medium Risk", stats['medium_risk_count'])


if __name__ == "__main__":
    main()