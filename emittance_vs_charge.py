import numpy as np
from matplotlib import pyplot as plt

def e1(emittance):
    return np.exp(-2.2*emittance)/(1-np.exp(-2.2*emittance))

e2 = 0.005
e3 = 0.059
e4 = 0.05

simulationData = np.loadtxt('F:\\experiments data\\07.20.2021\\20.07.2021\\solenoid scan\\simulation.txt')
simuBunch = simulationData[:,0]
simEmittance = simulationData[:,-2]
simEmittance1 = simulationData[:,-1]
bunchCharge1 = np.array([13.62,19.8,38,50,74,90,120,150,180,214,240])
experEmit1 = np.array([2.782666667,3.5908,3.741666667,3.340033333,3.499561667,3.996833333,4.433133333,
                                4.45265,5.191866667,6.1549,6.297766667])
experEmit1_error1 = e1(experEmit1)
experEmit1_error = np.sqrt(experEmit1_error1**2+e2**2+e3**2+e4**2) * experEmit1

bunchCharge2 = np.array([16,22,36,43,53,63,73,83,100])
experEmit2 = np.array([1.757663333,1.665286436,1.923358414,1.911657287,2.340795196,2.334367973,
                                 2.722485877,2.764937253,2.732161917])
experEmit2_error1 = e1(experEmit2)
experEmit2_error = np.sqrt(experEmit2_error1**2+e2**2+e3**2+e4**2) * experEmit2

experEmit3 = np.loadtxt('F:\\experiments data\\07.20.2021\\20.07.2021\\solenoid scan\\bunch charge vs emittance.txt')
experEmit3_error1 = e1(experEmit3[:,1])
experEmit3_error = np.sqrt(experEmit3_error1**2+e2**2+e3**2+e4**2) * experEmit3[:,1]

plt.figure()
plt.plot(simuBunch,simEmittance,'--',label='simulation as ideal conditions')
plt.plot(simuBunch,simEmittance1,label='simulation as real conditions')
# plt.errorbar(bunchCharge1,experEmit1, yerr=experEmit1_error, fmt="o", c='r', label='experiment measurement')
plt.errorbar(bunchCharge2*1.2,experEmit2, yerr=experEmit2_error, fmt="o", c='r', label='experiment measurement')
plt.errorbar(experEmit3[:,0]*1.2,experEmit3[:,1], yerr=experEmit3_error, fmt="o", c='r')
plt.xlabel('Bunch charge / pC')
plt.ylabel('Normalized emittance / um')
plt.legend()
plt.show()


## aperture 1 mm
simu = np.loadtxt('F:\\experiments data\\10.15.2021\\aperture 1 mm\\slit scan\\plot data simu sol 3.8 A.txt',skiprows=1)
exp = np.loadtxt('F:\\experiments data\\10.15.2021\\aperture 1 mm\\slit scan\\plot data exp.txt',skiprows=1)
bunch = simu[:,0]
sol38gauss = simu[:,1]
sol38unif = simu[:,2]

exp_error0 = e1(exp[:,1])
exp_error = np.sqrt(exp_error0**2+e2**2+e3**2+e4**2) * exp[:,1]

plt.figure()
plt.plot(bunch, sol38gauss, '--', label='Aperture 1 mm, Gaussian distribution simulation')
plt.plot(bunch, sol38unif, '--',label='Aperture 1 mm, Uniform distribution simulation')
plt.errorbar(exp[18:,0],exp[18:,1],yerr=exp_error[18:], fmt="o", c='b',label='Aperture 1 mm, Experiment data')
plt.plot(simuBunch,simEmittance,'--',label='Aperture 1.5 mm, Gaussian distribution simulation')
plt.plot(simuBunch,simEmittance1,label='Aperture 1.5 mm, Real distribution simulation')
plt.errorbar(bunchCharge2*1.2,experEmit2, yerr=experEmit2_error, fmt="o", c='r', label='Aperture 1.5 mm, Experiment data')
plt.errorbar(experEmit3[:,0]*1.2,experEmit3[:,1], yerr=experEmit3_error, fmt="o", c='r')
plt.xlabel('Bunch charge / pC')
plt.ylabel('Normalized emittance / um')
plt.legend()
plt.show()

data_sol_35 = np.loadtxt('F:\\experiments data\\10.15.2021\\aperture 1 mm\\slit scan\\plot data sol 3.5 A.txt', skiprows=1)
charge = data_sol_35[:,0]
exp_35 = data_sol_35[:,1]
simu_35_gauss = data_sol_35[:,2]
simu_35_uniform = data_sol_35[:,3]
exp_error0 = e1(exp_35[:])
exp_error35 = np.sqrt(exp_error0**2+e2**2+e3**2+e4**2) * exp_35[:]
plt.figure()
# plt.errorbar(charge,exp_35,yerr=exp_error35, fmt="o", c='r', label='Aperture 1 mm, Experiment data')
# plt.plot(charge,simu_35_gauss, '--',label='Aperture 1 mm Gaussian distribution simulation')
# plt.plot(charge,simu_35_uniform, '--',label='Aperture 1 mm Uniform distribution simulation')
plt.plot(bunch, sol38gauss, '--', label='Aperture 1 mm, Gaussian distribution simulation')
plt.plot(bunch, sol38unif, '--',label='Aperture 1 mm, Uniform distribution simulation')
plt.errorbar(exp[18:,0],exp[18:,1],yerr=exp_error[18:], fmt="o", c='r',label='Aperture 1 mm, Experiment data')
plt.xlabel('Bunch charge / pC')
plt.ylabel('Normalized emittance / um')
plt.legend()
plt.show()



## 1.5 mm aperture
simu1 = np.loadtxt('F:\\experiments data\\07.20.2021\\19.07.2021\\simulation results.txt',skiprows=1)
exp1 = np.loadtxt('F:\\experiments data\\07.20.2021\\19.07.2021\\experiment results.txt', skiprows=1)
factor = exp1[:,2]*1e-3/np.sqrt(exp1[:,3])
expResult = exp1[:,1]*factor
Rb_error = exp1[:,4]
beamlet_jitter = 0.059
energy_jitter = 0.05
error = np.sqrt(Rb_error**2+beamlet_jitter**2+energy_jitter**2) * expResult

## 1.0 mm aperture 3.8 A
simul38 = np.loadtxt('F:\\experiments data\\10.15.2021\\aperture 1 mm\\slit scan\\solenoid 3.8 A\\plot data simu sol 3.8 A.txt',skiprows=1)
exp38 = np.loadtxt('F:\\experiments data\\10.15.2021\\aperture 1 mm\\slit scan\\solenoid 3.8 A\\experiment results.txt',skiprows=1)
factor38 = exp38[:,2]*1e-3 / np.sqrt(exp38[:,3])
expResult38 = exp38[:,1] * factor38
Rb_error38 = exp38[:,4]
error38 = np.sqrt(Rb_error38**2+beamlet_jitter**2+energy_jitter**2) * expResult38

simu1[-1,1] = simEmittance[0]
plt.figure()
plt.errorbar(exp1[:,0],expResult,yerr=error, fmt="o", c='r', label='Aperture 1.5 mm, Experiment data')
plt.plot(simuBunch,simEmittance,'--',label='Aperture 1.5 mm, Gaussian distribution simulation')
plt.plot(simu1[:,0],simu1[:,1],label='Aperture 1.5 mm, Real distribution simulation')

plt.plot(simul38[:,0], simul38[:,1], '--', label='Aperture 1 mm, Gaussian distribution simulation')
plt.plot(simul38[:,0], simul38[:,2], '--',label='Aperture 1 mm, Uniform distribution simulation')
plt.errorbar(exp38[:,0], expResult38, yerr=error38, fmt="o", c='b',label='Aperture 1 mm, Experiment data')

plt.xlabel('Bunch charge / pC')
plt.ylabel('Normalized emittance / um')
plt.legend()
plt.show()

# particle_number = np.loadtxt('D:\\slit-scan simulation\\HPC\\run\\particles number vs normalized emittance.txt')
# number = particle_number[:,0]
# emit_ratio = particle_number[:,1]/1.19
# emit_ratio_linear = particle_number[:,2]/1.19
# x = np.arange(50000, 1600000)
# y = np.ones(len(x))
# plt.figure()
# plt.scatter(number,emit_ratio, label='with space charge')
# plt.scatter(number,emit_ratio_linear, label='without space charge')
# plt.plot(x,y,'--',c='r')
# plt.plot(x,y*1.1,'--',c='r')
# plt.xlabel('Particles number')
# plt.ylabel(r'$\frac{\epsilon_c}{\epsilon_o}$',rotation=0,fontsize=20)
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.ylim(0.9,1.2)
# plt.legend()
# plt.show()
