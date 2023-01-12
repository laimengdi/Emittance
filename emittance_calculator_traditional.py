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
from torch.utils.data import Dataset, DataLoader
import matplotlib
# matplotlib.use('Agg')
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
from image_classification import ImageClassify,ImageClassify2, classifyData

start = time()

modelPath = 'E:\\ms\\doctor\\experiments\\scripts\\silt-scan\\hida parameters 2 250 64.pt'
model = Net()
model.load_state_dict(torch.load(modelPath))
model.eval()

modelPath_classification = 'E:\\ms\\doctor\\experiments\\scripts\\silt-scan\\classification.pt'
model_classification = ImageClassify()
model_classification.load_state_dict(torch.load(modelPath_classification))
model_classification.eval()

f = open(os.getcwd() + '\\' + 'imagePath.txt')
line = f.readline()
line = line.strip('\n')
f.close()
main_path = line + '\\'
newFolder = 'after processing\\'

image_background = glob.glob(os.path.join(main_path, '*)  0.tif'))
for path in image_background:
    background = path

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

def test_classification(inputs):
    inputs = inputs.unsqueeze(1)
    output = model_classification(inputs)
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    return pred

## insure the record position numbers equal to the image numbers
def check_numbers():
    imagePath = line
    imageNumber = glob.glob(os.path.join(imagePath, '*.tif'))
    dataPath = line + '\\data.txt'
    inforData = np.genfromtxt(dataPath,skip_footer=1)
    lastData = np.genfromtxt(dataPath, skip_header=inforData.shape[0])
    if len(imageNumber) > inforData.shape[0]:
        os.remove(imageNumber[-1])
    if len(imageNumber) < inforData.shape[0]:
        inforData = np.delete(inforData,-1,axis=0)
    if lastData.shape[0] == 2:
        lastData = np.insert(lastData, 2, values=0, axis=0)
    inforData = np.insert(inforData,inforData.shape[0],values=lastData,axis=0)
    np.savetxt(dataPath, inforData)

    basic_data_path = glob.glob(os.path.join(main_path, '*).dat'))
    for pat in basic_data_path:
        basic_data = np.loadtxt(pat)
    E = basic_data[0]                                                  # beam total energy, unit: MeV
    D = basic_data[1]                                                  # distance from the slit to image, 0.75 m
    pixel_mm = basic_data[2]                                           # one pixel equal 0.0253 mm
    gama = E/0.511
    beta = np.sqrt(gama**2-1)/gama
    corePara = np.loadtxt(line+'\\core scale.txt')
    basicParameters = {'energy': E, 'drift distance': D, 'pixel size': pixel_mm,
                       'gamma':gama, 'beta':beta, 'scale': corePara}
    return basicParameters

## calculate rms radius, center

## create a new folder
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

## Gaussian fitting and draw the curve
def fit(x, a, b, c, d):
    return a/(np.sqrt(2*np.pi)*c)*np.exp(-(x-b)**2/(2*c**2))+d

def image_name(name):
    return int(name[-7:-4])

sigma_factor = 1.5
sigma_file_path = main_path + str(sigma_factor) + ' sigma traditional filter'
mkdir(sigma_file_path)

# limit = np.loadtxt(main_path + '\\1D images data\\limit.txt')
# prediction1 = np.loadtxt(main_path + '\\1D images data\\signal vs noise.txt')

## 2D images rms and center
def image2Dlimit(casePath):
    image2Dpath = os.path.join(casePath,'2D images data')
    mkdir(image2Dpath)
    factor = np.loadtxt(os.path.join(casePath,'core scale.txt'))
    paths = glob.glob(os.path.join(casePath, '*.tif'))
    paths.sort(key=image_name)
    background_imgae = cv2.imread(paths[0], -1)
    background_imgae = background_imgae[::-1,::-1]
    h, w = background_imgae.shape
    allBeam = np.zeros((h, w))

    for i in arange(len(paths)):
        path = paths[i]
        head,tail = ntpath.split(path)
        uint16_img0 = cv2.imread(path, -1)
        uint16_img = uint16_img0[::-1,::-1]   # rotate image 180 degrees
        allBeam = allBeam + uint16_img

    all_sum = np.sum(allBeam)
    h_sum = 0
    for i in range(h):
        h_sum = h_sum + i * np.sum(allBeam[i,:])
    h_center = h_sum/all_sum
    h_rms = 0
    for i in range(h):
        h_rms = h_rms + np.sum(allBeam[i,:]) * (i-h_center)**2
    h_rms = np.sqrt(h_rms/all_sum)

    w_sum = 0
    for i in range(w):
        w_sum = w_sum + i * np.sum(allBeam[:,i])
    w_center = w_sum / all_sum
    w_rms = 0
    for i in range(h):
        w_rms = w_rms + np.sum(allBeam[:,i]) * (i - w_center) ** 2
    w_rms = np.sqrt(w_rms / all_sum)

    h_limit1 = int(h_center - h_rms * factor)
    h_limit2 = int(h_center + h_rms * factor)
    w_limit1 = int(w_center - w_rms * factor)
    w_limit2 = int(w_center + w_rms * factor)

    if h_limit1 < 0:
        h_limit1 = 0
    if h_limit2 >= h:
        h_limit2 = h-2
    if w_limit1 < 0:
        w_limit1 = 0
    if w_limit2 >= w:
        w_limit2 = w-2

    limit = np.array([h_limit1, h_limit2, w_limit1, w_limit2])

    return(limit)

def image2Dclear(casePath):
    image1Dpath = os.path.join(casePath, '1D images data median filter')
    mkdir(image1Dpath)
    h_limit1, h_limit2, w_limit1, w_limit2 = image2Dlimit(main_path)
    paths = glob.glob(os.path.join(casePath, '*.tif'))
    paths.sort(key=image_name)
    allBeam = np.zeros((1, 494))
    image1D_background = np.zeros((494, 1))
    center = np.zeros((len(paths), 1))
    sigma = np.zeros((len(paths), 1))

    for i in arange(len(paths)):
        time_i1 = time()
        path = paths[i]
        head,tail = ntpath.split(path)
        uint16_img0 = cv2.imread(path, -1)
        uint16_img0 = uint16_img0 - np.min(uint16_img0)
        uint16_img1 = cv2.blur(uint16_img0,(11,11))    # median filter
        uint16_img = uint16_img1[::-1,::-1]          # rotate image 180 degrees
        uint16_img[:h_limit1+1,:] = 0
        uint16_img[h_limit2:, :] = 0
        uint16_img[:, :w_limit1+1] = 0
        uint16_img[:, w_limit2:] = 0

        h, w = uint16_img.shape
        image1D0 = np.zeros((h, 1))
        image1D = np.zeros((h, 1))
        centeri = np.zeros((h, 1))
        squari = np.zeros((h, 1))
        if i == 0:
            for j in arange(h):
                image1D_background[j, 0] = np.sum(uint16_img[j, :])
                # image1D = image1D_background
        else:
            for j in arange(h):
                image1D[j, 0] = np.sum(uint16_img[j, :])
                image1D0[j, 0] = np.sum(uint16_img0[j, :])
        image1D = image1D - image1D_background
        image1D0 = image1D0 - np.min(image1D0)

        if np.max(image1D) < 1000:
            image1D = np.zeros((h, 1))

        for n in arange(image1D.shape[0]):
            if image1D[n] < np.max(image1D) * 0.05:              # cut off 5% intensity
                image1D[n] = 0
            centeri[n] = n * image1D[n]
        if np.sum(image1D) != 0:
            center[i, 0] = np.sum(centeri) / np.sum(image1D)
        for m in arange(image1D.shape[0]):
            squari[m, 0] = image1D[m] * ((m - center[i, 0]) ** 2)
        if np.sum(image1D) == 0:
            sigma[i, 0] = 0
        else:
            sigma[i, 0] = np.sqrt(np.sum(squari) / np.sum(image1D))
        time_i2 = time()
        print(time_i2 - time_i1)
        allBeam = np.vstack((allBeam, image1D.T))

        xindex = np.arange(len(image1D))
        plt.figure(i)
        plt.plot(xindex, image1D0, label='original')
        plt.plot(xindex, image1D, label='ML filter')
        plt.scatter(h_limit1, 0, c='r')
        plt.scatter(h_limit2, 0, c='r')
        plt.savefig(sigma_file_path + '\\' + str(i) + '.jpg')
        plt.legend()
        # plt.switch_backend('Agg')
        plt.close()

    allBeam = np.delete(allBeam, 0, axis=0)
    beamIntensity = np.zeros((allBeam.shape[0], 1))
    weight = np.zeros((allBeam.shape[0], 1))
    for k in arange(beamIntensity.shape[0]):
        beamIntensity[k, 0] = np.sum(allBeam[k, :])


    sigma = sigma * 0.0253 * 1e-3  # unit: m
    basicParameters = check_numbers()
    # x = np.arange(beamIntensity.shape[0])
    # plt.plot(x,beamIntensity)
    # plt.show()
    center = center * 0.0253 * 1e-3
    np.savetxt(os.path.join(image1Dpath, 'center.txt'), center)
    np.savetxt(os.path.join(image1Dpath, 'allBeam.txt'), allBeam)
    np.savetxt(os.path.join(image1Dpath, 'sigma.txt'), sigma)
    np.savetxt(os.path.join(image1Dpath, 'beamIntensity.txt'), beamIntensity)


    # slit position
    original_data = np.genfromtxt(os.path.join(casePath, 'data.txt'), skip_footer=1)
    original_data = np.delete(original_data, 0, axis=0)

    distance_step = np.zeros((original_data.shape[0] - 1, 1))  # obtain the slit real position when take the image
    time_image_step = np.zeros((original_data.shape[0] - 1, 1))
    time_demand_step = np.zeros((original_data.shape[0] - 1, 1))
    time_delay = np.zeros((original_data.shape[0], 1))
    velocity_step = np.zeros((original_data.shape[0] - 1, 1))
    shift_step = np.zeros((original_data.shape[0] - 1, 1))
    distance_after_shift = np.zeros((original_data.shape[0], 1))
    for i in range(0, original_data.shape[0]):
        if i <= original_data.shape[0] - 2:
            distance_step[i, 0] = original_data[i + 1, 1] - original_data[i, 1]
            time_image_step[i, 0] = original_data[i + 1, 2] - original_data[i, 2]
            time_demand_step[i, 0] = original_data[i + 1, 0] - original_data[i, 0]
            velocity_step[i, 0] = distance_step[i, 0] / time_demand_step[i, 0]
            time_delay[i, 0] = original_data[i, 2] - original_data[i, 0]
            shift_step[i, 0] = velocity_step[i, 0] * time_delay[i, 0]
            distance_after_shift[i, 0] = original_data[i, 1] + shift_step[i, 0]
        if i > original_data.shape[0] - 2:
            time_delay[i, 0] = original_data[i, 2] - original_data[i, 0]
            distance_after_shift[i, 0] = original_data[i, 1] + np.average(shift_step[:, 0]) * time_delay[i, 0]

    distance_after_shift = distance_after_shift * 1e-3  # unit: m
    np.savetxt(os.path.join(casePath, '1D images data', 'data_exact.txt'), distance_after_shift)
    imageParameters = {'beamlet intensity': allBeam, 'beamlet center': center, 'slit position': distance_after_shift,
                       'beam integrate intensity': beamIntensity, 'beamlet rms': sigma, 'shift step': shift_step}
    return imageParameters

## calculate emittance
def emittanceCalculation(basicParameters, imageParameters):
    E = basicParameters['energy']
    D = basicParameters['drift distance']
    pixel_size = basicParameters['pixel size']
    gamma = basicParameters['gamma']
    beta = basicParameters['beta']
    beamlet = imageParameters['beamlet intensity'][:-1]
    beamlet_particles_number = imageParameters['beam integrate intensity'][:-1]
    beamlet_position = imageParameters['beamlet center'][:-1]
    slit_position = imageParameters['slit position']
    beamlet_rms = imageParameters['beamlet rms'][:-1]
    shift_step = imageParameters['shift step']
    print(beamlet.shape)

    distance_after_shift = slit_position

    intensity = np.sum(beamlet_particles_number)

    # print(fit_data.shape, distance_after_shift.shape)
    y_center = np.sum(beamlet_particles_number[:,0]*distance_after_shift[:,0])/intensity
    y_center_i = distance_after_shift[:,0]-y_center
    y_sigma = np.sum(beamlet_particles_number[:,0]*(distance_after_shift[:,0]-y_center)**2)/intensity
    # print(beamlet_position.shape,distance_after_shift.shape)
    beamlet_dp = ((beamlet_position[:,0]-distance_after_shift[:,0])/D).reshape(-1,1)
    beamlet_dp_average = np.sum(beamlet_particles_number[:,0]*beamlet_dp[:,0])/intensity
    sigma_dp_square = ((beamlet_rms[:,0]**2)/D**2).reshape(-1,1)
    y_dp_square = np.sum(beamlet_particles_number[:,0]*(sigma_dp_square[:,0]+(beamlet_dp[:,0]-beamlet_dp_average)**2))/intensity
    y_ydp = np.sum(beamlet_particles_number[:,0]*distance_after_shift[:,0]*beamlet_dp[:,0])/intensity-y_center*beamlet_dp_average
    emittance = np.sqrt(y_sigma*y_dp_square-y_ydp**2)
    normalize_emittance = beta*gamma*emittance * 1e6                          # unit: um
    print(normalize_emittance,y_sigma,y_dp_square,y_ydp,beamlet_dp_average)
    k = np.array([normalize_emittance,y_sigma,y_dp_square,y_ydp])
    np.savetxt(sigma_file_path+'\\detail parameters from images.txt',k)


    ## calculate the influence of error from beamlet center and rms jitter on emittance
    y_center = np.sum(beamlet_particles_number[:,0]*distance_after_shift[:,0])/intensity
    y_center_i = distance_after_shift[:,0]-y_center
    y_sigma = np.sum(beamlet_particles_number[:,0]*(distance_after_shift[:,0]-y_center)**2)/intensity
    beamlet_position_with_error = np.zeros((len(beamlet_position),1))
    beamlet_rms_with_error = np.zeros((len(beamlet_rms),1))
    beamlet_position_with_error = beamlet_position * (1-0.01)
    beamlet_rms_with_error = beamlet_rms* (1-0.04)
    beamlet_dp_with_error = ((beamlet_position_with_error[:,0]-distance_after_shift[:,0])/D).reshape(-1,1)
    beamlet_dp_average_with_error = np.sum(beamlet_particles_number[:,0]*beamlet_dp_with_error[:,0])/intensity
    sigma_dp_square_with_error = ((beamlet_rms_with_error[:,0]**2)/D**2).reshape(-1,1)
    y_dp_square_with_error = np.sum(beamlet_particles_number[:,0]*(sigma_dp_square_with_error[:,0]+(beamlet_dp_with_error[:,0]-beamlet_dp_average_with_error)**2))/intensity
    y_ydp_with_error = np.sum(beamlet_particles_number[:,0]*distance_after_shift[:,0]*beamlet_dp_with_error[:,0])/intensity-y_center*beamlet_dp_average_with_error
    emittance_with_error = np.sqrt(y_sigma*y_dp_square_with_error-y_ydp_with_error**2)
    normalize_emittance_with_error = beta*gamma*emittance_with_error * 1e6                          # unit: um
    print('normalize_emittance_with_error: ' + str(normalize_emittance_with_error))


    ## calculate cross emittance
    Y_screen = np.sum(beamlet_particles_number[:, 0] * beamlet_position[:, 0]) / intensity
    Y_screen_squre = np.sum(beamlet_particles_number[:, 0] * (beamlet_position[:, 0] - Y_screen) ** 2) / intensity
    Xc_dx_Yc = np.sum(beamlet_particles_number[:,0]*(beamlet_dp[:,0]-beamlet_dp_average)*(beamlet_position[:, 0] - Y_screen))/intensity
    XY = np.sum(beamlet_particles_number[:,0]*distance_after_shift[:,0]*beamlet_position[:,0])/intensity
    cross_emittance = np.sqrt((y_sigma*Xc_dx_Yc-y_ydp*XY)/D)
    sigma_emittance = np.sqrt(np.sum(beamlet_particles_number[:, 0] * y_sigma * sigma_dp_square[:,0]) / (intensity * D ** 2))
    print('cross emittance: ' + str(cross_emittance))
    print('rms emittance: ' + str(sigma_emittance))
    print('geometric emittance: ' + str(emittance))

    twinss_beta = y_sigma/emittance
    twinss_gamma = y_dp_square/emittance
    twinss_alpha = -y_ydp/emittance

    positioni = distance_after_shift[:,0] - y_center
    position = np.zeros((beamlet.shape[0],beamlet.shape[1]))
    for i in range(beamlet.shape[0]):
        position[i,:] = positioni[i]
    position = position * 1e3

    beam_dp = (np.arange(0.0, beamlet.shape[1], 1.0) * pixel_size / D).reshape(1,-1)
    for j in range(beamlet.shape[0]-1):
        beam_dpi = (np.arange(0.0, beamlet.shape[1], 1.0) * pixel_size / D).reshape(1,-1) - np.sum(shift_step[:j]) / D
        beam_dp = np.vstack((beam_dp,beam_dpi))

    maxindex_beam_dp = np.unravel_index(beamlet.argmax(), beamlet.shape)  # find the original point in dp in phase space
    beam_dp = beam_dp - beam_dp[maxindex_beam_dp[0], maxindex_beam_dp[1]]  # chang the vertical original point

    np.savetxt(sigma_file_path+'\\position.txt', position,
               delimiter='   ')
    np.savetxt(sigma_file_path+'\\dy.txt', beam_dp,
               delimiter='   ')
    np.savetxt(sigma_file_path+'\\beam.txt', beamlet,
               delimiter='   ')

basicParameters = check_numbers()
imageParameters = image2Dclear(main_path)
emittanceCalculation(basicParameters, imageParameters)
end = time()
print(end-start)

