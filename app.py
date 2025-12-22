import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import Dataset
import numpy as np
import cv2
import gradio as gr

# 1. Define the Model Architecture (Must match the trained model)
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        # Load the ResNext50 backbone
        model = models.resnext50_32x4d(pretrained=True)
        # Remove the last two layers (classification layers) to use as feature extractor
        self.model = nn.Sequential(*list(model.children())[:-2])
        # LSTM for temporal analysis
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        # Reshape for CNN: (batch * seq_length, channels, height, width)
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        # Reshape for LSTM: (batch, seq_length, features)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        # Classify based on the last LSTM output
        return fmap, self.dp(self.linear1(x_lstm[:,-1,:]))

# 2. Setup Transformations
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 3. Load the Model
# NOTE: Ensure 'model.pth' is in the same folder as this script
model = Model(2)
path_to_model = "model.pth" 

try:
    # Load with map_location to ensure it runs even if trained on GPU but run on CPU
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Could not find '{path_to_model}'. Please upload your trained model file.")

# Softmax for probability calculation
sm = nn.Softmax(dim=1)

# 4. Helper Classes & Functions
class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=20, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        # Extract frames
        for i, frame in enumerate(self.frame_extract(video_path)):
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        
        # Pad if fewer than 20 frames (edge case handling)
        if len(frames) < self.count:
             while len(frames) < self.count:
                 frames.append(frames[-1])

        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image

def predict(model, img):
    with torch.no_grad():
        fmap, logits = model(img)
        logits = sm(logits)
        predictionVal = logits[0][1] # Probability of being Real (assuming index 1 is Real)
        confidence = predictionVal * 100
    return confidence

def fakeOrNot(video_path):
    if not video_path:
        return "Please upload a video."
    
    # Preprocess the video
    video_dataset_ processed = validation_dataset([video_path], sequence_length=20, transform=train_transforms)
    
    # Make Prediction
    predictConfidence = predict(model, video_dataset_processed[0])
    
    if predictConfidence > 60:
        return f"Prediction: REAL Video\nConfidence: {predictConfidence:.2f}%"
    else:
        tempX = 100 - predictConfidence
        return f"Prediction: DEEPFAKE Video\nConfidence: {tempX:.2f}%"

# 5. Gradio Interface
def gradio_interface(video):
    return fakeOrNot(video)

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Textbox(label="Result"),
    title="Deepfake Detection System",
    description="Upload an MP4 video to detect if it is Real or a Deepfake using ResNext + LSTM."
)

if __name__ == "__main__":
    iface.launch()