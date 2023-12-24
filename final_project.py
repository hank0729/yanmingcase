import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from glob import glob
import librosa
import librosa.display
import IPython.display as ipd
from itertools import cycle
import torchvision
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split 
from tqdm import tqdm
from flask import Flask

app = Flask(__name__)
app.route('/')
def doapp():
    # Set theme for plotting
    sns.set_theme(style="white", palette=None)
    color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    # Load audio files
    audio_files_path = glob('./data/*/*/*.wav')
    y_list = []
    sr_list = []
    for audio_file in audio_files_path:
        y, sr = librosa.load(audio_file)
        y_list.append(y)
        sr_list.append(sr)

    # Process audio files
    y_trimmed_list = []
    sr = sr_list[0]
    target_length = 5 * sr
    for y in y_list:
        y_trimmed, _ = librosa.effects.trim(y, top_db=60)
        if len(y_trimmed) > target_length:
            y_trimmed = y_trimmed[:target_length]
        y_trimmed_list.append(y_trimmed)

    # Load ECS files
    ECS_files_path = glob('./audio/*.wav')
    y0_list = []
    sr0_list = []
    for ECS_file in ECS_files_path:
        y, sr = librosa.load(ECS_file)
        y0_list.append(y)
        sr0_list.append(sr)

    # Convert audio to Mel Spectrogram
    S_db_channels_list = []
    num_channels = 3
    hop_size = [128, 256, 512]
    win_size = [100, 200, 400]

    for y_trimmed in y_trimmed_list:
        S_db_channels = []
        for j in range(num_channels):
            S = librosa.feature.melspectrogram(y=y_trimmed, sr=sr, hop_length=hop_size[j], win_length=win_size[j])
            S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
            S_db_mel_resized = np.asarray(torchvision.transforms.Resize((224, 224))(Image.fromarray(S_db_mel)))
            S_db_channels.append(S_db_mel_resized)
        S_db_channels_arr = np.array(S_db_channels)
        S_db_channels_list.append(S_db_channels_arr)

    # Prepare dataset
    dataset = []
    for i in range(len(y_list)):
        new_entry = {'values': S_db_channels_list[i], 'target': 1, 'audio': y_list[i]}
        dataset.append(new_entry)

    for i in range(len(y0_list)):
        new_entry = {'values': S_db_channels_list[i], 'target': 0, 'audio': y0_list[i]}
        dataset.append(new_entry)

    # Split dataset
    X_train, X_test, Y_train, Y_test = train_test_split([entry['values'] for entry in dataset], [entry['target'] for entry in dataset], test_size=0.2, random_state=42)

    # Define dataset class
    class AudioDataset(Dataset):
        def __init__(self, features, targets):
            self.features = features
            self.targets = targets

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            feature = self.features[idx]
            target = self.targets[idx]
            return feature, target

    # Prepare DataLoaders
    trainset = AudioDataset(X_train, Y_train)
    testset = AudioDataset(X_test, Y_test)
    train_Loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
    test_Loader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

    # Define DenseNet model
    class DenseNet(nn.Module):
        def __init__(self):
            super(DenseNet, self).__init__()
            num_classes = 2
            self.model = models.densenet121(pretrained=True)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, num_classes)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            output = self.model(x)
            output = self.softmax(output)
            return output

    # Train model
    net = DenseNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    def train_model(model, dataloader, lossfn, optimizer):
        epochs = 10
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            with tqdm(total=len(dataloader)) as t:
                for i, data in enumerate(train_Loader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = lossfn(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    t.update()
                    running_loss += loss.item()
                    if i % 200 == 199:
                        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                        running_loss = 0.0
        model.eval()
        print('Finished Training')

    # Run training
    train_model(net, train_Loader, criterion, optimizer)

    # Save model
    PATH = './densenet.pth'
    torch.save(net.state_dict(), PATH)

    # Evaluate model
    dataiter = iter(test_Loader)
    images, labels = next(dataiter)

    net = DenseNet()
    net.load_state_dict(torch.load(PATH))
    outputs = net(images)

    # Calculate accuracy
    k = []
    acc = 0.
    total = 0.
    for output in outputs:
        predicted_label = 0 if output[0] > output[1] else 1
        k.append(predicted_label)
    for i, label in enumerate(labels):
        if k[i] == label.item():
            acc += 1
        total += 1
    return ('accuracy: ', acc / total)
    
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))   
    app.run(host='0.0.0.0', port=port) 
