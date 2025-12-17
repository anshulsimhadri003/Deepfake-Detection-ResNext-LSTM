# Detection of Deepfake Videos

A B.Tech final year project that identifies manipulated media by analyzing spatial inconsistencies (ResNext) and temporal flickers (LSTM).

## 🚀 Key Features
- **Hybrid Architecture:** Combines ResNext-50 for frame analysis and LSTM for sequence analysis.
- **Web UI:** Integrated Gradio interface for easy video testing.
- **High Accuracy:** Optimized on the Celeb-DF dataset.

## 🛠️ Tech Stack
- **Language:** Python
- **DL Framework:** PyTorch
- **Computer Vision:** OpenCV, Face_Recognition
- **Interface:** Gradio

## 📊 Methodology
1. **Preprocessing:** Extracting 20 frames per video using OpenCV.
2. **Face Detection:** Isolating facial regions for focused analysis.
3. **Classification:** Passing sequences through the ResNext+LSTM layers.
