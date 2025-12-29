# 🛡️ A-EDDS: Agentic Edge Deepfake Detection System

**Hackathon MVP** - AI-Powered Deepfake Detection Running 100% Offline

---

## 🎯 Project Overview

A-EDDS is an edge-based deepfake detection system that runs entirely offline on a laptop CPU. It combines:

- **Video Deepfake Detection**: Facial analysis using MediaPipe + CNN
- **Audio Deepfake Detection**: Voice analysis using MFCC + CNN  
- **Multimodal Fusion**: Weighted combination of audio/video scores
- **Agentic Decision Engine**: Observe-Reason-Act loop for intelligent risk assessment
- **Live Demo UI**: Streamlit interface for real-time demonstration

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python main.py
```

The Streamlit UI will launch automatically at `http://localhost:8501`

---

## 📁 Project Structure
```
deepfake_agent/
│
├── models/                  # Pretrained models (mock for MVP)
│   ├── video_model.onnx    # Video deepfake detection model
│   └── audio_model.onnx    # Audio deepfake detection model
│
├── inference/              # Detection modules
│   ├── video_detector.py  # Video deepfake detection
│   ├── audio_detector.py  # Audio deepfake detection
│   └── fusion.py          # Multimodal fusion logic
│
├── agent/                  # Agentic decision engine
│   └── decision_engine.py # Observe-Reason-Act loop
│
├── ui/                     # User interface
│   └── app.py             # Streamlit web app
│
├── main.py                 # Application entry point
└── requirements.txt        # Python dependencies
```

---

## 🎮 How to Use

### Upload Content

1. **Video + Audio Mode**: Upload both video and audio files for comprehensive analysis
2. **Video Only Mode**: Upload video files for visual deepfake detection
3. **Audio Only Mode**: Upload audio files for voice deepfake detection

### Adjust Settings

- **Fusion Weights**: Balance between video and audio scores
- **Risk Thresholds**: Customize low/medium/high risk boundaries

### Analyze Results

- View deepfake probability scores
- See agentic risk assessment (LOW/MEDIUM/HIGH)
- Get recommended actions based on risk level
- Inspect agent reasoning process (Observe-Reason-Act)

---

## 🤖 Agentic Decision Engine

The agent follows a three-step loop:

1. **OBSERVE**: Gather multimodal detection scores
2. **REASON**: Analyze risk level based on thresholds
3. **ACT**: Generate alerts and recommendations

Risk Levels:
- 🔴 **HIGH**: Likely deepfake, do not trust
- 🟡 **MEDIUM**: Suspicious, verify before trusting
- 🟢 **LOW**: Minor artifacts, likely authentic
- ✅ **AUTHENTIC**: No deepfake indicators

---

## 🧠 Technical Details

### Video Detection Pipeline

1. Face detection using MediaPipe
2. Face cropping and preprocessing
3. CNN inference for deepfake probability
4. Frame-level scoring with aggregation

### Audio Detection Pipeline

1. MFCC feature extraction using Librosa
2. Feature normalization and padding
3. CNN inference on spectrogram
4. Audio-level deepfake probability

### Fusion Strategy

- Weighted averaging of video and audio scores
- Default: 60% video, 40% audio
- Confidence adjustment for missing modalities

---

## 🔧 Model Information

**IMPORTANT**: This MVP uses **mock models with random weights** for demonstration purposes.

For production deployment, replace with:

**Video Models**:
- MesoNet
- EfficientNet-B0
- XceptionNet

**Audio Models**:
- RawNet2
- AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal features)

The code is designed to seamlessly integrate real pretrained models.

---

## 🎓 Hackathon Demo Tips

1. **Emphasize Offline Capability**: Show that system works without internet
2. **Show Agent Reasoning**: Expand the agent reasoning process view
3. **Test Multiple Files**: Upload various video/audio samples
4. **Adjust Thresholds Live**: Demonstrate real-time parameter tuning
5. **Explain Risk Levels**: Walk through the observe-reason-act loop

---

## 🛠️ Troubleshooting

### Import Errors
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Camera Not Working
Use uploaded files instead of live camera feed

### Slow Performance
- Reduce max_frames in video processing (default: 30)
- Use shorter video/audio clips
- Ensure no other heavy applications running

---

## 📊 Performance Notes

- **Video Processing**: ~30 frames analyzed per video
- **Audio Processing**: Up to 10 seconds analyzed
- **Inference Time**: ~2-5 seconds per modality on CPU
- **Memory Usage**: ~500MB-1GB RAM

---

## 🔒 Privacy & Security

- ✅ 100% offline processing
- ✅ No data sent to cloud
- ✅ No external API calls
- ✅ All computation on local device

---

## 🏆 Future Enhancements

- [ ] Real pretrained models (MesoNet, RawNet2)
- [ ] GPU acceleration support
- [ ] Batch processing mode
- [ ] Export detection reports
- [ ] Additional modalities (text, metadata)
- [ ] Explainability visualizations

---

## 👥 Credits

Built for hackathon demonstration purposes.

**Technologies Used**:
- PyTorch (Deep Learning)
- MediaPipe (Face Detection)
- Librosa (Audio Processing)
- Streamlit (UI Framework)
- OpenCV (Video Processing)

---

## 📝 License

Hackathon MVP - Educational Use Only

---

## 🆘 Support

For issues or questions during the hackathon:
1. Check console output for error messages
2. Verify all dependencies are installed
3. Ensure video/audio files are valid formats
4. Try with different input files

---

**Good luck with your hackathon demo! 🚀**

