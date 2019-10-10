import unittest
import numpy as np
import scipy.io as sio
import filtergaussianregression as gr
#
class TestFilter(unittest.TestCase):
    def test_gaussian_regression_p1(self):
        lambda_c = 800
        #
        content = sio.loadmat('../datasets/profile_Lc3200.mat')
        lambdap = content['lambdap'][0]
        dx = content['dx'][0]
        zsin = content['zsin'][0]
        zdat = content['zdat'][0]
        #
        wavi = gr.gaussian_regression_p1(dx, zdat, lambda_c)
        #
        numdata = len(zdat)
        mid = int(numdata/2)
        testvalue = np.mean(zsin[mid-10:mid+10]-wavi[mid-10:mid+10])
        self.assertLessEqual(testvalue, 0.05)

