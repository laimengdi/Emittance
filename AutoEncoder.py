import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.fc1 = nn.Linear(2000, 128)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2)
        self.convd1 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        # self.fc2 = nn.Linear(2000, 1024)
        # self.fc3 = nn.Linear(1024, 256)
        # self.fc4 = nn.Linear(256, 56)

    def forward(self, x):
        # x = x.reshape(-1, 4, 500)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.convd1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))

        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # self.fc4 = nn.Linear(128, 2000)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.convd1 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(480, 494)
        # self.fc3 = nn.Linear(256, 1024)
        # self.fc2 = nn.Linear(1024, 2000)
        # self.fc1 = nn.Linear(2000, 2000)

    def forward(self, x):
        # x = self.fc4(x)
        # x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv3(x))
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.convd1(x))
        x = F.interpolate(x, scale_factor=2)
        # x = x.reshape(-1, 1, 1952)
        x = self.fc(x)
        # x = F.relu(x)
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc1(x)
        return x

# Class for autoencoder

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.E = Encoder()
        self.D = Decoder()
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, x):
        h = self.E(x)
        self.data_rho = h.mean(0)               # calculates rho from encoder activations
        out = self.D(h)
        return out

    def decode(self, h):
        with torch.no_grad():
            return self.D(h)

    def rho_loss(self, rho, size_average=True):
        dkl = - rho * torch.log(self.data_rho) - (1 - rho) * torch.log(1 - self.data_rho)  # calculates KL divergence
        if size_average:
            self._rho_loss = dkl.mean()
        else:
            self._rho_loss = dkl.sum()
        return self._rho_loss

    def loss(self, x, target, **kwargs):
        assert x.shape == target.shape
        self._loss1 = self.loss_fn(x, target, **kwargs)
        x1 = x * 0
        self._loss2 = self.loss_fn(x1, target, **kwargs)
        self._loss = self._loss1
        return self._loss

## dataloader prepare
class RegressData(Dataset):
    def __init__(self, root, list_IDs, labels):
        # self.transform=transform
        self.rootPath = root
        self.list_IDs = list_IDs
        self.labels = labels
        self.X = torch.load(os.path.join(self.rootPath, self.list_IDs + '.pt'))
        print(os.path.join('data', self.list_IDs + '.pt'))
        self.Y = torch.load(os.path.join(self.rootPath, self.labels + '.pt'))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return {'cases': x,
                'label': y}
#

def RPDLoss(output, target):
  return torch.mean(torch.abs(target - output) / ((torch.abs(target) + torch.abs(output)) / 2))

def RELoss(output, target):
  return torch.mean(torch.abs(target - output) / ((torch.abs(target) + torch.abs(output)) / 2))

# rootPath = 'F:\\hida project\\dataset\\Autoencoder\\'
# xlabels = ['trainCases', 'testCases']
# ylabels = ['labelTrain', 'labelTest']
# train_ds = RegressData(rootPath, xlabels[0],ylabels[0])
# test_ds = RegressData(rootPath, xlabels[1],ylabels[1])