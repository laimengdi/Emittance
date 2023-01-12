import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import torch.optim as optim
# import torch.utils.data
from torch.autograd import Variable
import matplotlib.pyplot as plt


class ImageClassify(nn.Module):
    def __init__(self):
        super(ImageClassify, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.pol1 = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.pol1(x)
        x = torch.sigmoid(x)
        x = x.squeeze(2)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)

        return output


class ImageClassify2(nn.Module):
    def __init__(self):
        super(ImageClassify2, self).__init__()
        self.fc1 = nn.Linear(494, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        output = F.log_softmax(x)
        # output = output.squeeze(1)

        return output


class classifyData(Dataset):
    def __init__(self, list_IDs, labels):
        # self.transform=transform
        self.labels = labels
        self.list_IDs = list_IDs
        IDx = self.list_IDs
        IDy = self.labels
        self.X = torch.load(os.path.join('data', IDx + '.pt'))
        print(os.path.join('data', IDx + '.pt'))
        self.Y = torch.load(os.path.join('data', IDy + '.pt'))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return {'cases': x,
                'label': y}


