import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torch.utils.data
from torch.autograd import Variable
import matplotlib.pyplot as plt

#matplotlib inline

# np.random.seed(12321)  # for reproducibility
# torch.manual_seed(12321)  # for reproducibility

# batch_size = 16
# rho = 0.05                                                # fixed value
# beta = 2                                                  # 5
# NUM_EPOCHS = 100
# which_GPU = '0'
# # device = torch.device("cuda:"+which_GPU)
#
#
# ## read data
# mainPath = 'E:\\ms\\images rms ML\\cases'                 # data path
#
# X = np.zeros((494))
# Y = np.zeros((494))
# for i in range(0, 1920):
#     foderPath = mainPath + '\\' + str(i)
#     Xfile = '\\X.txt'
#     Ylabel = '\\label2.txt'
#     Xdata = np.loadtxt(foderPath+Xfile)
#     Ydata = np.loadtxt(foderPath+Ylabel)
#     X = np.append(X, values=Xdata[:, 1]/np.max(Xdata[:, 1]), axis=0)      # normalize each case and add
#     Y = np.append(Y, values=Ydata[:]/np.max(Ydata[:]), axis=0)
#
# X = X.reshape(-1, Xdata.shape[0])                        # 1164 cases * 494 features matrix
# Y = Y.reshape(-1, Ydata.shape[0])
# X = np.delete(X, 0, axis=0)                                # delete the original first row
# Y = np.delete(Y, 0, axis=0)
#
# index = [i for i in range(len(X))]
# random.shuffle(index)
# X = X[index, :]
# Y = Y[index, :]
#
# x_train, x_test = np.split(X, [1520], axis=0)                # 850 cases as training cases
# print(x_train.shape)
# print(x_test.shape)
# #x_mean=np.mean(X)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
#
# x_train = np.reshape(x_train, (1520, 1, 494))
# x_test = np.reshape(x_test, (1920-1520, 1, 494))
# train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, num_workers=0)  # package data as batch_size to tensor
# test_loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size)
# Encoder and decoder classes

class Encoder_classification(nn.Module):
    def __init__(self):
        super(Encoder_classification, self).__init__()
        # self.fc1 = nn.Linear(2000, 128)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        # self.fc2 = nn.Linear(2000, 1024)
        # self.fc3 = nn.Linear(1024, 256)
        # self.fc4 = nn.Linear(256, 56)

    def forward(self, x):
        # x = x.reshape(-1, 4, 500)
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))

        return x

class Decoder_classification(nn.Module):
    def __init__(self):
        super(Decoder_classification, self).__init__()
        # self.fc4 = nn.Linear(128, 2000)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=1)

        self.fc = nn.Linear(484, 494)
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
        # x = x.reshape(-1, 1, 1952)
        x = self.fc(x)
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc1(x)
        return x

# Class for autoencoder

class Net_classification(nn.Module):
    def __init__(self, loss_fn=nn.MSELoss(), lr=2e-4, l2=0):  # 3e-4
        super(Net_classification, self).__init__()
        self.E = Encoder_classification()
        self.D = Decoder_classification()
        self.loss_fn = loss_fn
        self._rho_loss = None
        self._loss = None
        self.optim = optim.Adam(self.parameters(), lr=lr, weight_decay=l2)

    def forward(self, x):
        h = self.E(x)
        self.data_rho = h.mean(0)  # calculates rho from encoder activations
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
      #  self._loss = torch.mean((x - target)**2)
      #  self._loss = torch.sum((x - target)**2)/(len(x))
       # self._loss = torch.sum((x - target)**2)/x.shape[0]
        self._loss = self.loss_fn(x, target, **kwargs)
      #  sizes = input.size()
 #       self._loss = ((x-target)**2).mean()
        return self._loss

# Making AEs with 16, 64 and 256 neurons in latent layer

# model =Net().cuda(device)
# model = Net()

# Train function

# def train(epoch, model, beta):
#     loss_total = 0
#     train_size = len(train_loader)
#     # print(train_size)
#     for batch_idx, data in enumerate(train_loader):
#         model.optim.zero_grad()
#         # inputs = data.to(device)  # .clone().detach()
#         inputs = data
#         inputs = Variable(inputs, requires_grad=True)
#         output = model(inputs)
#         # error1 = torch.sum((output[1,0,:] - inputs[1,0,:])**2)
#         # print('train error:' + str(error1))
#         if beta != 0:
#             rho_loss = model.rho_loss(rho)
#             rms_loss = model.loss(output.squeeze(), inputs.squeeze())
#             # print(rms_loss)
#             loss = rms_loss + beta * rho_loss
#         else:
#             loss = model.loss(output, inputs)
#         loss.backward()
#         model.optim.step()
#         loss_total = loss_total + loss.item()
#     return loss_total / float(train_size)
#
# def test(model):
#     loss_total = 0
#     test_size = len(test_loader)
#     # print(test_size)
#     for batch_idx, data in enumerate(test_loader):
#
#         with torch.no_grad():
#             # inputs = data.to(device)
#             inputs = data
#             inputs = Variable(inputs, requires_grad=False)
#             output = model(inputs)
#             # error1 = torch.sum((output[1, 0, :] - inputs[1, 0, :]) ** 2)
#             # print('test error: ' + str(error1))
#             loss = model.loss(output, inputs)
#             loss_total = loss_total + loss.item()
#     return loss_total/float(test_size)
#
# for epoch in range(1, NUM_EPOCHS):
#     model.train()
#     train_loss = train(epoch, model, beta)
#
#     model.eval()
#     test_loss = test(model)
#     print('Epoch {} of {}, Train Loss: {:.3f}, Val Loss: {:.3f}'.format(epoch, NUM_EPOCHS, train_loss, test_loss))
#
#
# savePath = 'E:\\ms\\images rms ML\\state_dict_model.pt'
# torch.save(model.state_dict(), savePath)
#
# def plot_auto(net, testloader, indices):
#     for data in testloader:
#         # data = data.to(device)
#         outputs = net(data).cpu().detach().numpy()
#         break
#
#     output1 = outputs.reshape((-1, 494))
#     data1 = data.reshape((-1, 494))
#     print(np.mean((output1-data1.cpu().detach().numpy()) ** 2))
#     for i in range(0, indices.shape[0]):
#         plt.figure(i)
#         plt.plot(data1[indices[i]].cpu().detach().numpy(), label='GT')
#         plt.plot(output1[indices[i]], label='predicted')
#         plt.legend()
#         plt.show()
# imageShow = np.array([0, 2, 8, 12, 14])
# plot_auto(model, test_loader, imageShow)
