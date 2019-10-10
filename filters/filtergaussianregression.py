import numpy as np
import scipy
import scipy.special
#
def gaussian_regression_p1(dx, heights, lamb):
#
# the profiles have to be the rows
#
# filter constant
    alpha = np.sqrt(np.log(2)/np.pi)
#
    n = len(heights)
    nh = int(n/2) + 1
    nlambda = int(np.floor(lamb/dx)) + 1
    x = np.array(np.arange(0,n)*dx)
    x2 = np.concatenate((x,np.array(np.arange(-nlambda,0,1)*dx)))
    n2 = len(x2)
    heightextended = np.zeros(n2)
    heightextended[0:n] = heights
#
    sgauss = np.exp(-np.pi*np.square(x2)/(alpha*lamb)**2)/(alpha*lamb)
# (1) Fourier trafos fuer die Inhomogenitaet des Gl-sys.:
# (1.a) z-Werte:
#     dazu fuer die periodische Fortsetzung verlaengern
    Fz = np.fft.fft(heightextended)
# (1.b)
    Fsgauss = np.fft.fft(sgauss)
    sgaussx = np.multiply(x2,sgauss)
    Fsgaussx = np.fft.fft(sgaussx)
    spR1 = np.multiply(Fz,Fsgauss)
    spR2 = np.multiply(Fz,Fsgaussx)
    R1 = np.fft.ifft(spR1)*dx;
    R2 = np.fft.ifft(spR2)*dx;
#
# (2) Gleichungssystem des RG-Filters loesen
# (1.a) Momentenmatrix, hier die Gaussglocke
#       mit dem Maximum und nicht mit der
#       Flaeche auf 1 normiert verwenden
    x3 = x
    x3[n-nlambda:n] = np.flip(x[0:nlambda],axis=0)
    sgauss = np.exp(-np.pi*np.square(x3)/((alpha*lamb)**2))
    mu0 = scipy.special.erf(np.sqrt(np.pi)*x3/(alpha*lamb))
    mu0 = 0.5*( mu0 + 1 )
    mu1 = -( (lamb*alpha)/(2.0*np.pi) ) * sgauss
    mu2 = ((lamb*alpha)**2/(2.0*np.pi) ) * ( mu0 - (x3/(lamb*alpha)) * sgauss) 
#
# (1.b) 2 \times 2 Gl-Sys. aufgeloest
    d = mu2*mu0 - mu1**2
    b0 = mu2/d
    sig = np.ones(n)
    sig[0:nlambda] = -1
    b1 = sig*mu1/d
    wout = np.zeros(n)
    wout = b0*np.real(R1[0:n]) + b1*np.real(R2[0:n])
    return wout

