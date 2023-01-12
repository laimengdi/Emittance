import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

uniform = np.loadtxt('F:\\experiments data\\10.15.2021\\aperture 1 mm\\slit scan\\indis_input uniform.ini',skiprows=7)
gauss = np.loadtxt('F:\\experiments data\\10.15.2021\\aperture 1 mm\\slit scan\\indis_input gauss.ini',skiprows=7)
# Generate fake data
x = uniform[:,0]
y = uniform[:,1]
# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

plt.figure()
plt.scatter(x, y, c=z, s=30)
plt.xlim(-0.002,0.002)
plt.ylim(-0.002,0.002)
plt.xlabel('x / mm')
plt.ylabel('y / mm')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.show()