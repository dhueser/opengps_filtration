import numpy as np
import scipy
from scipy import special
#
def gaussian_reg_p1(dx, heights, lamb):
#
# the profiles have to be the rows
#
# filter constant multiplied with cut off wavelength
    alpha_lamb = np.sqrt(np.log(2)/np.pi) * lamb
#
    numh = len(heights)
    nlambda = int(np.floor(lamb/dx)) + 1
    x_1 = np.array(np.arange(0, numh)*dx)
    x_2 = np.concatenate((x_1, np.array(np.arange(-nlambda, 0, 1)*dx)))/alpha_lamb
#
# (1) Fourier trafos fuer die Inhomogenitaet des Gl-sys.:
# (1.a) z-Werte:
#     dazu fuer die periodische Fortsetzung verlaengern
    heightextended = np.zeros(len(x_2))
    heightextended[0:numh] = heights
    fourier_height = np.fft.fft(heightextended)
# (1.b)
    sgauss = np.exp(-np.pi*np.square(x_2))/(alpha_lamb)
    fourier_sgauss = np.fft.fft(sgauss)
    fourier_sgaussx = np.fft.fft(x_2*sgauss)
    r_1 = np.fft.ifft(fourier_height*fourier_sgauss) * dx
    r_2 = np.fft.ifft(fourier_height*fourier_sgaussx) * dx
#
# (2) Gleichungssystem des RG-Filters loesen
# (1.a) Momentenmatrix, hier die Gaussglocke
#       mit dem Maximum und nicht mit der
#       Flaeche auf 1 normiert verwenden
    x_3 = x_1
    x_3[numh-nlambda:numh] = np.flip(x_1[0:nlambda], axis=0)
    sgauss3 = np.exp(-np.pi*np.square(x_3/alpha_lamb))
    mu0 = 0.5*(special.erf(np.sqrt(np.pi)*x_3/(alpha_lamb)) + 1)
    mu1 = -(alpha_lamb / (2.0*np.pi)) * sgauss3
    mu2 = (alpha_lamb**2 / (2.0*np.pi)) * (mu0 - (x_3/alpha_lamb) * sgauss3) 
#
# (1.b) 2 \times 2 Gl-Sys. aufgeloest
    determi = mu2*mu0 - mu1**2
    b_0 = mu2/determi
    sig = np.ones(numh)
    sig[0:nlambda] = -1
    b_1 = sig*mu1/determi
    wout = np.zeros(numh)
    wout = b_0*np.real(r_1[0:numh]) + b_1*np.real(r_2[0:numh])
    return wout

