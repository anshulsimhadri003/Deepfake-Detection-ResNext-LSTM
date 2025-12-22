Project Title: Detection of Deepfake Videos using ResNext-LSTM.


Project Overview: A brief summary explaining that the system detects facial manipulations by analyzing both spatial artifacts and temporal inconsistencies.

Key Methodology:


Preprocessing: Video frame extraction (20 frames per sequence) and face centering using face_recognition.
+1


Architecture: A hybrid model using ResNext-50 for spatial feature extraction and LSTM for temporal sequence analysis.


Results: Mention your accuracy (e.g., 88%) and include a screenshot of your Confusion Matrix or ROC Curve from your report to provide visual proof.
+1

How to Run: Simple 3-step instructions:

git clone [your-repo-link]

pip install -r requirements.txt

python app.py
