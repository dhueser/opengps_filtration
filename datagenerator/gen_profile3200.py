import numpy as np
import scipy.io as sio
#
numdata = 8000
dx = 0.5
amplitude = 0.35
lambdap = 3200.0
rms_noise = 0.08
#
x = np.arange(0, numdata)*dx - 0.5*dx*numdata
zsin = np.cos(2*np.pi*x/lambdap)
zdat = zsin + np.random.normal(loc = 0.0, scale= rms_noise, size= numdata)
sio.savemat('../datasets/profile_Lc3200.mat',\
 {'dx': dx, 'lambdap': lambdap, 'zsin': zsin, 'zdat': zdat})
