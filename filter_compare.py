import os
import multiprocessing
import time
import seaborn as sns
from PIL import ImageEnhance
from PIL import Image
import scipy.signal as signal
from cv2 import cv2
import glob
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import ntpath
from numpy import *
from scipy.optimize import curve_fit
from time import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
from numpy import linspace, hstack
from AutoEncoder import Encoder, Decoder, Net
from scipy import signal

def gaussian_fit(x, a, b, c, d):
    return a/(np.sqrt(2*np.pi)*c)*np.exp(-(x-b)**2/(2*c**2))+d

def gaussian_fit_show(x,p_fit):
    a, b, c, d = p_fit.tolist()
    return a/(np.sqrt(2*np.pi)*c)*np.exp(-(x-b)**2/(2*c**2)) +d

def calculator(x):
    '''
    calculate x center and rms size
    '''
    index = x.shape[0]
    y = np.zeros((index,2))
    for i in range(x.shape[0]):
        y[i,0] = i*x[i]
    center = np.sum(y[:,0])/np.sum(x[:])
    for i in range(x.shape[0]):
        y[i,1] = (i - center)**2*x[i]
    rms = np.sqrt(np.sum(y[:,1])/np.sum(x[:]))
    return center, rms

def medfilt (x, k):
    """
    Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median(y, axis=1)

def test(model, x_data):
    for batch_idx, data in enumerate(x_data):
        with torch.no_grad():
            inputs = data
            inputs = Variable(inputs, requires_grad=False)
            inputs = inputs.float()
            output = model(inputs)
            output = output.squeeze().numpy()
    for i in range(len(output)):
        if output[i] < 0:
            output[i] = 0
    return output

modelPath = 'E:\\ms\\doctor\\experiments\\scripts\\silt-scan\\hida parameters 2 250 64.pt'
model = Net()
model.load_state_dict(torch.load(modelPath))
model.eval()

image = cv2.imread('F:\\experiments data\\07.20.2021\\19.07.2021\\196 pC\\beam\\subtract dark current\\EM()(7.19.2021_9-44 ) 16.tif', -1)
h, w = image.shape
image1D = np.zeros((h,1))
for i in range(h):
    image1D[i,0] = np.sum(image[i,:])

image1D = image1D - np.min(image1D)
data_max = 1e6
dataTensor = np.reshape(image1D / data_max, (1, 1, 494))
dataTensor = torch.utils.data.DataLoader(dataTensor, batch_size=16, num_workers=0)

## ML filter
newdata = test(model, dataTensor) * data_max
center_ml1, rms_ml1 = calculator(newdata)
limit_ml1 = int(center_ml1 - rms_ml1 * 3)
limit_ml2 = int(center_ml1 + rms_ml1 * 3)
if limit_ml1 < 0:
    limit_ml1 = 0
if limit_ml2 > 494:
    limit_ml2 = 493
newdata[:limit_ml1] = 0
newdata[limit_ml2:] = 0
center_ml1, rms_ml1 = calculator(newdata)
print([center_ml1, rms_ml1])

## median filter
image_tra = image1D[:,0]
data_traditional = medfilt(image_tra,7)
center_tra1, rms_tra1 = calculator(data_traditional)
limit_tra1 = int(center_tra1 - rms_tra1 * 1.5)
limit_tra2 = int(center_tra1 + rms_tra1 * 1.5)
if limit_tra1 < 0:
    limit_tra1 = 0
if limit_tra2 > 494:
    limit_tra2 = 493
# data_traditional[:limit_tra1] = 0
# data_traditional[limit_tra2:] = 0
center_tra1, rms_tra1 = calculator(data_traditional)
print([center_tra1, rms_tra1])

## Butterworth filter
b, a = signal.butter(5, 0.09)
zi = signal.lfilter_zi(b, a)
z, _ = signal.lfilter(b, a, image_tra, zi=zi*image_tra[0])
y_Fir = signal.filtfilt(b, a, image_tra)

image2 = cv2.imread('F:\\experiments data\\07.20.2021\\19.07.2021\\196 pC\\beam\\subtract dark current\\EM()(7.19.2021_9-44 ) 16.tif', -1)
h2, w2 = image.shape
image1D2 = np.zeros((h2,1))
for i in range(h2):
    image1D2[i,0] = np.sum(image2[i,:])
image1D2 = image1D2 - np.min(image1D2)
data_max2 = 1e6
dataTensor2 = np.reshape(image1D2 / data_max2, (1, 1, 494))
dataTensor2 = torch.utils.data.DataLoader(dataTensor2, batch_size=16, num_workers=0)
newdata2 = test(model, dataTensor2) * data_max2
image_tra2 = image1D2[:,0]

## Gaussian fit
x = np.arange(len(image1D2))
popt, pcov = curve_fit(gaussian_fit, x, image1D2[:,0], [image1D2[:,0].max(), 400, 10, 100], maxfev=800000)
y_gaussian = gaussian_fit_show(x,popt)



plt.figure(2)
plt.plot(x,image1D2[:,0],label='Original image intensity')
# plt.plot(x,y_gaussian, label='Gaussian fit')
plt.plot(x,y_Fir, label='Butterworth filter')
# plt.plot(x,data_traditional, label=' Median filter')
# plt.plot(x, newdata, label='ML filter')
# plt.scatter(limit_ml1,0,c='C2', label='3.0 ' + r'$\sigma_{ML}$' + ' cut')
# plt.scatter(limit_ml2,0,c='C2')
# plt.scatter(limit_tra1,0,c='C1', label='1.5 ' + r'$\sigma_{Median}$' + ' cut')
# plt.scatter(limit_tra2,0,c='C1')
plt.xlabel('Pixel index')
plt.ylabel('Intensity')
plt.legend()
plt.show()
