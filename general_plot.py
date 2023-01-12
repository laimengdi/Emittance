import numpy as np
from matplotlib import pyplot as plt

## energy and energy spread vs gun phase
data = np.loadtxt('F:\\experiments data\\02.11.2021\\beam energy and energy spread.txt',skiprows=3)
phase = data[:,0] - 8
energy =data[:,1]
sigma_s5 = data[:-2,2]
sigma_s4 = data[:-2,3]
energySpread = np.sqrt(sigma_s5**2-sigma_s4**2)*13.51             # 13.51 keV/mm

simuData = np.loadtxt('E:\\ms\\doctor\\simulation\\aperture 1.0 mm\\1\\ep.Scan.001')
simuPhase = simuData[:,0]
simuEnery = simuData[:,7] + 0.511
simuEnerySp = simuData[:,6]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(simuPhase, simuEnery, label='Beam Energy in Simulation', c='r')
ax1.scatter(phase, energy, label='Beam Energy in Experiment', c='r')
ax2.plot(simuPhase, simuEnerySp, label='Beam Energy Spread in Simulation', c='b')
ax2.scatter(phase[:-2], energySpread, label='Beam Energy Spread in Experiment', c='b')

ax1.set_xlabel('Phase / deg')
ax1.set_ylabel('Beam Energy / MeV', color='r')
ax2.set_ylabel('Beam Energy Spread / keV', color='b')
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
lines = lines_1 + lines_2
labels = labels_1 + labels_2
ax1.legend(lines, labels, loc='center left')
plt.show()


## Normalized emittance vs solenoid current
def e1(emittance):
    return np.exp(-2.2*emittance)/(1-np.exp(-2.2*emittance))

e2 = 0.005
e3 = 0.059
e4 = 0.05

emit1 = np.loadtxt('F:\\experiments data\\01.11.2021\\results.txt',skiprows=1)
emit2 = np.loadtxt('F:\\experiments data\\02.11.2021\\results.txt',skiprows=1)

experEmit1_error1 = e1(emit1[:,1])
experEmit1_error = np.sqrt(experEmit1_error1**2+e2**2+e3**2+e4**2) * emit1[:,1]

experEmit2_error1 = e1(emit2[:,1])
experEmit2_error = np.sqrt(experEmit2_error1**2+e2**2+e3**2+e4**2) * emit2[:,1]

plt.figure()
plt.errorbar(emit1[:,0], emit1[:,1], yerr=experEmit1_error, c='r',label='Experiment', fmt="o",)
plt.errorbar(emit2[:,0], emit2[:,1], yerr=experEmit2_error, c='r', fmt="o",)
plt.plot(emit2[:,0],emit2[:,2],c='b',label='Simulation Gaussian distribution')
plt.plot(emit2[:,0],emit2[:,3],'b--',label='Simulation Uniform distribution')
plt.xlabel('Solenoid current / A')
plt.ylabel('Normalized emittance / um')
plt.legend()
plt.show()
