# Detection of Deepfake Videos 🛡️
This project implements a robust deep learning system designed to identify manipulated facial media. By utilizing a hybrid ResNext-LSTM architecture, the system analyzes both spatial artifacts within individual frames and temporal inconsistencies across video sequences to distinguish between real and deepfake content.

## 🚀 Overview
As AI-generated misinformation becomes more sophisticated, traditional detection methods often fail. This project addresses the challenge by:

- **Spatial Analysis:** Using ResNext-50 as a feature extractor to detect pixel-level irregularities and facial geometry distortions.
- **Temporal Analysis:** Employing LSTM (Long Short-Term Memory) layers to identify unnatural "flickers" or transitions between frames that are characteristic of deepfakes.
- **Deployment:** Providing a user-friendly web interface via Gradio for real-time video testing.

## 🛠️ Tech Stack
- **Deep Learning Framework:** PyTorch 
- **Model Backbone:** ResNext-50 + LSTM 
- **Computer Vision:** OpenCV, face_recognition 
- **Web Interface:** Gradio 
- **Environment:** Python 3.x 

## 📊 Methodology
- **Preprocessing:** Videos are processed to extract a sequence of 20 frames.
- **Face Extraction:** Facial regions are targeted and isolated using the face_recognition library to reduce background noise.
- **Feature Extraction:** Each frame is passed through the ResNext layers to generate a high-dimensional feature vector.
- **Sequence Processing:** The vectors are fed into an LSTM network to evaluate the consistency of movements over time.
- **Classification:** The final layer outputs a probability score indicating whether the video is "Real" or "Fake".

## 📈 Performance & Results
- **Dataset:** Benchmarked on the Celeb-DF dataset.
- **Accuracy:** Achieved a detection accuracy of approximately 88% on the validation set.
- **Confusion Matrix:** The model demonstrates high precision in identifying deepfakes, effectively minimizing false negatives.

## ⚙️ How to Run
1. Open the `.ipynb` file in Google Colab.
2. Mount your Google Drive containing the dataset.
3. Update the dataset paths in the "Data Loading" section.
4. Run the training cells or load the pre-trained weights.
   
## 👥 Credits
**Project Members:**
- Agasthya Todupunoori 
- Poojana Varre
- Anshul Simhadri 
- Ram Saladi

**Under the Guidance of:**

Dr. V Sangeeta, Associate Professor, Department of CSE, GITAM School of Technology.

## 📜 License

This project is released under the MIT License. See the LICENSE file for more details.
