#Model Training

#0Libraries
%%capture
import glob
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
!pip install face_recognition
import face_recognition
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

#1 Validating Video to check if we can send to train or test and coverting the video into frames
def validate_video(vid_path, train_transforms):
    transform = train_transforms
    count = 20
    video_path = vid_path
    frames = []
    a = int(100 / count)
    first_frame = np.random.randint(0, a)
    temp_video = video_path.split("/")[-1]
    for i, frame in enumerate(frame_extract(video_path)):
        frames.append(transform(frame))
        if len(frames) == count:
            break
    frames = torch.stack(frames)
    frames = frames[:count]
    return frames


#extract 'a' from video
def frame_extract(path):
    vidObj = cv2.VideoCapture(path)
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image

#2 Re-size and Normalize the image
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

#3 Loading fake video dataset and counting number of videos
gbmd = {
    "file": [],
    "label": []
}
frame_count = []

video_files1 =  glob.glob('/content/drive/MyDrive/deep/Celeb_fake_face_only-20241230T040101Z-001/Celeb_fake_face_only/*.mp4')
for video_file in video_files1:
    cap = cv2.VideoCapture(video_file)
    if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) < 100:
        video_files1.remove(video_file)
        continue
    frame_count.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    gbmd["file"].append(video_file)
    gbmd["label"].append("FAKE")
print("Total no of video: ", len(frame_count))

#print(gbmd["file"][432], gbmd["label"][432])

#4 Real Video dataset
video_files2 = glob.glob('/content/drive/MyDrive/deep/Celeb_real_face_only-20241230T040206Z-001/Celeb_real_face_only/*.mp4')

for video_file in video_files2:
    cap = cv2.VideoCapture(video_file)
    if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) < 100:
        video_files2.remove(video_file)
        continue
    frame_count.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    gbmd["file"].append(video_file)
    gbmd["label"].append("REAL")

# print("frames are ", frame_count)
print("Total no of video: ", len(frame_count))

#5 plot the image
def im_plot(tensor):
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    b, g, r = cv2.split(image)
    image = cv2.merge((r, g, b))
    image = image * [0.22803, 0.22145, 0.216989] + [0.43216, 0.394666, 0.37645]
    image = image * 255.0
    plt.imshow(image.astype(int))
    plt.show()


def number_of_real_and_fake_videos(data_list):
    header_list = ["file", "label"]
    lab = pd.read_csv("/content/drive/My Drive/Gobal_metadata.csv", names=header_list)
    fake = 0
    real = 0
    if len(lab) > 0:
        for i in data_list:
            temp_video = i.split("/")[-1]
            if labels.loc[labels["file"] == temp_video].empty:
                print("No video with file name " + temp_video + " found.")
            else:
                label = lab.iloc[
                    (labels.loc[labels["file"] == temp_video].index.values[0]), 1
                ]
                if label == "FAKE":
                    fake += 1
                if label == "REAL":
                    real += 1
    # print(real)
    # print(fake)
    return real, fake

# print(gbmd)
header_list = ["file","label"]
df = pd.DataFrame(gbmd).sample(frac=1)
# df = df.sample(frac=1)
# print(df)
df.to_csv('/content/drive/My Drive/Gobal_metadata.csv', header=False, index=False)
labels = pd.read_csv('/content/drive/My Drive/Gobal_metadata.csv',names=header_list)
# print(labels)
video_files = labels["file"]
# labels = labels["label"][1:]
# print(video_files[0])

#6 Train and test split of dataset
from sklearn.model_selection import train_test_split

# random.shuffle(gbmd)
# ---------------------------------------
total_files = len(video_files)
# print(total_files)
train_ratio = 0.7
test_ratio = 0.1

# Calculate the sizes of train, test, and validation sets
train_size = int(train_ratio * total_files)
test_size = int(test_ratio * total_files)
valid_size = total_files - train_size - test_size

# Split the video files into train, test, and validation sets
train_videos = video_files[:train_size]
test_videos = video_files[train_size:train_size+test_size]
valid_videos = video_files[train_size+test_size:]

# Print the sizes of the train, test, and validation sets
print("Train set size:", len(train_videos))
print("Validation set size:", len(valid_videos))
print("Test set size:", len(test_videos))

#7 Called for Veryfing the dataset real or fake
def number_of_real_and_fake_videos(data_list):
  header_list = ["file","label"]
  lab = pd.read_csv('/content/drive/My Drive/Gobal_metadata.csv',names=header_list)
  fake = 0
  real = 0
  # print(data_list)
  if len(lab) > 0:
    for i in data_list:
      temp_video = i#.split('/')[-1]
      if labels.loc[labels["file"] == temp_video].empty:
        print("No video with file name " + temp_video + " found.")
      else:
        label = lab.iloc[(labels.loc[labels["file"] == temp_video].index.values[0]),1]
        if(label == 'FAKE'):
          fake+=1
        if(label == 'REAL'):
          real+=1
  # print(real)
  # print(fake)
  return real, fake

#8 Trasfroming(Resing and normalize) the data using #2
train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])

valid_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])

test_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])
#9 Preparing the dataset to send for train and test
class video_dataset(Dataset):
    def __init__(self, video_names, labels, sequence_length=60, transform=None):
        self.video_names = video_names
        self.labels = labels
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        # print(self.video_names)
        video_path = self.video_names[idx]
        # print(video_path)
        frames = list(self.frame_extract(video_path))
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)
        temp_video = video_path#.split("/")[-1]
        # print(temp_video)
        if len(labels) == 0:
            print("No labels found.")
            return frames, None
        label = self.labels.iloc[
            (labels.loc[labels["file"] == temp_video].index.values[0]), 1
        ]
        if label == "FAKE":
            label = 0
        if label == "REAL":
            label = 1
        for i, frame in enumerate(frames):
            frames[i] = self.transform(frame)
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)
        frames = frames[: self.count]
        # print("length:" , len(frames), "label",label)
        return frames, label

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image
#10 Loading the dataset for train test and validate
train_data = video_dataset(train_videos.reset_index(drop=True),labels,sequence_length = 10,transform = train_transforms)
val_data = video_dataset(valid_videos.reset_index(drop=True),labels,sequence_length = 10,transform = train_transforms)
test_data = video_dataset(test_videos.reset_index(drop=True),labels,sequence_length = 10,transform = train_transforms)

train_loader = DataLoader(train_data,batch_size = 4,shuffle = True,num_workers = 4)
valid_loader = DataLoader(val_data,batch_size = 4,shuffle = True,num_workers = 4)
test_loader = DataLoader(test_data,batch_size = 4,shuffle = True,num_workers = 4)

#11 Cleaning if there any loss of frames
for batch_idx, (data, target) in enumerate(valid_loader):
       if torch.isnan(data).any() or torch.isinf(data).any():
           print(f"Found NaN or Inf in data at batch index {batch_idx}")
#12 Model

from torch import nn
from torchvision import models
class Model(nn.Module):
    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained = True) #Residual Network CNN
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,2048)
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(torch.mean(x_lstm,dim = 1)))

#13 Training Epochs
import torch
from torch.autograd import Variable
import time
import os
import sys
import os
!pip install -Uqq ipdb
import ipdb
def train_epoch(epoch, num_epochs, data_loader, model, criterion, optimizer):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    t = []
    for i, (inputs, targets) in enumerate(data_loader):
        if torch.cuda.is_available():
            targets = targets.type(torch.cuda.LongTensor)
            inputs = inputs.cuda()
        _,outputs = model(inputs)
        # print("asidh", outputs)
        loss  = criterion(outputs,targets.type(torch.cuda.LongTensor))
        acc = calculate_accuracy(outputs, targets.type(torch.cuda.LongTensor))
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # Clip gradients
        optimizer.step()
        sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d / %d] [Loss: %f, Acc: %.2f%%]"
                % (
                    epoch,
                    num_epochs,
                    i,
                    len(data_loader),
                    losses.avg,
                    accuracies.avg))
    torch.save(model.state_dict(),'/content/checkpoint.pt')
    return losses.avg,accuracies.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    return 100* n_correct_elems / batch_size

#14 Confusion Matrix
import seaborn as sn
#Output confusion matrix
def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print('True positive = ', cm[0][0])
    print('False positive = ', cm[0][1])
    print('False negative = ', cm[1][0])
    print('True negative = ', cm[1][1])
    print('\n')
    df_cm = pd.DataFrame(cm, range(2), range(2))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.ylabel('Actual label', size = 20)
    plt.xlabel('Predicted label', size = 20)
    plt.xticks(np.arange(2), ['Fake', 'Real'], size = 16)
    plt.yticks(np.arange(2), ['Fake', 'Real'], size = 16)
    plt.ylim([2, 0])
    plt.show()
    calculated_acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+ cm[1][1])
    print("Calculated Accuracy",calculated_acc*100)

def plot_loss(train_loss_avg,test_loss_avg,num_epochs):
  loss_train = train_loss_avg
  loss_val = test_loss_avg
  print(num_epochs)
  epochs = range(1,num_epochs+1)
  plt.plot(epochs, loss_train, 'g', label='Training loss')
  plt.plot(epochs, loss_val, 'b', label='validation loss')
  plt.title('Training and Validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()
def plot_accuracy(train_accuracy,test_accuracy,num_epochs):
  loss_train = train_accuracy
  loss_val = test_accuracy
  epochs = range(1,num_epochs+1)
  plt.plot(epochs, loss_train, 'g', label='Training accuracy')
  plt.plot(epochs, loss_val, 'b', label='validation accuracy')
  plt.title('Training and Validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()

#15 testing set
def test(epoch,model, data_loader ,criterion):
    print('Testing')
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    pred = []
    true = []
    count = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            if torch.cuda.is_available():
                targets = targets.cuda().type(torch.cuda.FloatTensor)
                inputs = inputs.cuda()
            # ipdb.set_trace()
            abc,outputs = model(inputs)
            # print("abc", abc, "\nOutputs", outputs)
            loss = torch.mean(criterion(outputs, targets.type(torch.cuda.LongTensor)))
            acc = calculate_accuracy(outputs,targets.type(torch.cuda.LongTensor))
            _,p = torch.max(outputs,1)
            true += (targets.type(torch.cuda.LongTensor)).detach().cpu().numpy().reshape(len(targets)).tolist()
            pred += p.detach().cpu().numpy().reshape(len(p)).tolist()
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            sys.stdout.write(
                    "\r[Batch %d / %d]  [Loss: %f, Acc: %.2f%%]"
                    % (
                        i,
                        len(data_loader),
                        losses.avg,
                        accuracies.avg
                        )
                    )
        print('\nAccuracy {}'.format(accuracies.avg))
    return true,pred,losses.avg,accuracies.avg

#16 Training the Model using the dataset
from sklearn.metrics import confusion_matrix

#learning rate
lr = 1e-5#0.001
#number of epochs
num_epochs = 20

optimizer = torch.optim.Adam(model.parameters(), lr= lr,weight_decay = 1e-7)

class_weights = torch.from_numpy(np.asarray([1,15])).type(torch.FloatTensor).cuda()
criterion = nn.CrossEntropyLoss(weight = class_weights).cuda()
# criterion = nn.CrossEntropyLoss().cuda()
# criterion = nn.BCELoss().cuda()

train_loss_avg =[]
train_accuracy = []
test_loss_avg = []
test_accuracy = []
for epoch in range(1, num_epochs + 1):
    l, acc = train_epoch(epoch,num_epochs,train_loader,model,criterion,optimizer)
    train_loss_avg.append(l)
    train_accuracy.append(acc)
    print(train_loss_avg)
    true,pred,tl,t_acc = test(epoch,model,train_loader,criterion)
    test_loss_avg.append(tl)
    print(test_loss_avg)
    test_accuracy.append(t_acc)
plot_loss(train_loss_avg,test_loss_avg,len(train_loss_avg))
plot_accuracy(train_accuracy,test_accuracy,len(train_accuracy))
print(confusion_matrix(true,pred))
print_confusion_matrix(true,pred)

Real_test_loss_avg = []
Real_test_accuracy = []
RealTesttrue,RealTestpred,RealTesttl,RealTestt_acc = test(1,model,test_loader,criterion)
Real_test_loss_avg.append(RealTesttl)
Real_test_accuracy.append(RealTestt_acc)
# plot_loss(train_loss_avg,test_loss_avg,len(train_loss_avg))
# plot_accuracy(train_accuracy,test_accuracy,len(train_accuracy))
print(confusion_matrix(RealTesttrue,RealTestpred))
print_confusion_matrix(RealTesttrue,RealTestpred)

#17 ROC Curve
np.random.seed(42)
y_true = RealTesttrue
y_scores = RealTestpred

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

#18 Saving the model
torch.save(model.state_dict(), '/content/saved_model.pth')
path_to_model = "/content/drive/MyDrive/FF_88_acc_seqLen_20_Confusion_matrix.pth"
torch.save(model.state_dict(), '/content/drive/MyDrive/FF_88_acc_seqLen_20_Confusion_matrix.pth')

#Prediction and User Interface

import gradio as gr
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch import nn
from torchvision import models
import numpy as np
import cv2

# Defining model class
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:,-1,:]))

# Loading model
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

model = Model(2)
path_to_model = "/content/drive/MyDrive/FF_88_acc_seqLen_20_Confusion_matrix.pth"
model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
model.eval()


sm = nn.Softmax()

# Prediction function
def predict(model, img):
    fmap, logits = model(img.to('cpu'))
    logits = sm(logits)
    predictionVal = logits[0][1]
    confidence = predictionVal * 100
    return confidence

# Defining the validation dataset class
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
        a = int(100/self.count)
        first_frame = np.random.randint(0, a)
        for i, frame in enumerate(self.frame_extract(video_path)):
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
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

# Defining the video processing function
def fakeOrNot(video_path):
    video_dataset = validation_dataset([video_path], sequence_length=20, transform=train_transforms)
    predictConfidence = predict(model, video_dataset[0])
    if predictConfidence > 60:
        return f"With {predictConfidence:.2f}% confidence it is a REAL video"
    else:
        tempX = 100 - predictConfidence
        return f"With {tempX:.2f}% confidence it is a Deep Faked video"

# Create Gradio interface
def gradio_interface(video):
    return fakeOrNot(video)

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Textbox(label="Prediction"),
    title="Deepfake Detection",
    description="Upload a video to check if it's real or deepfake."
)

iface.launch(debug=True)
