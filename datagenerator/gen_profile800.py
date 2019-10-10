import numpy as np
import scipy.io as sio
#
numdata = 8000
dx = 0.5
amplitude = 0.35
lambdap = 800.0
rms_noise = 0.08
#
x = np.arange(0, numdata)*dx
zsin = np.sin(2*np.pi*x/lambdap)
zdat = zsin + np.random.normal(loc = 0.0, scale= rms_noise, size= numdata)
sio.savemat('../datasets/profile_Lc800.mat',\
 {'dx': dx, 'lambdap': lambdap, 'zsin': zsin, 'zdat': zdat})
