# -*- coding: utf-8 -*-
import numpy as np
from scipy import misc
import time

import cuconv as cu
import cpuconv as cp
from scipy import signal as sg


# Get numpy array from image
def from_img(fname):
    return np.asarray(misc.imread(fname, flatten=True), dtype=np.float32)


# Write numpy array to image
def to_img(m):
    return np.clip(np.absolute(m), 0, 255)


# Create a box blue kernel of radius r
def k_boxblur(r):
    return np.ones([r+(0 if r % 2 is 1 else 1), r+(0 if r % 2 is 1 else 1)])


# Normalize kernel matrix
def nrm(m):
    m = np.array(m)
    return m/np.sum(np.abs(m))


# CREATE KERNELS
k_sv = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
k_sh = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
k_b5 = k_boxblur(5)

# LOAD IMAGE
a = from_img('img.png').astype(np.float32)


# GPU
start = time.time()
c = cu.convolve(a, k_sv)
misc.imsave('results/g_result_sv.png', to_img(c))
c = cu.convolve(a, k_sh)
misc.imsave('results/g_result_sh.png', to_img(c))
c = cu.convolve(a, k_b5)
misc.imsave('results/g_result_b5.png', to_img(c))
end = time.time()
print("GPU time: %.5f s" % (end-start))

# SCIPY
start = time.time()
c = sg.convolve(a, nrm(k_sv))
misc.imsave('results/s_result_sv.png', to_img(c))
c = sg.convolve(a, nrm(k_sh))
misc.imsave('results/s_result_sh.png', to_img(c))
c = sg.convolve(a, nrm(k_b5))
misc.imsave('results/s_result_b5.png', to_img(c))
end = time.time()
print("Scipy time: %.5f s" % (end-start))

# CPU
start = time.time()
c = cp.convolve(a, k_sv)
misc.imsave('results/c_result_sv.png', to_img(c))
c = cp.convolve(a, k_sh)
misc.imsave('results/c_result_sh.png', to_img(c))
c = cp.convolve(a, k_b5)
misc.imsave('results/c_result_b5.png', to_img(c))
end = time.time()
print("CPU time: %.5f s" % (end-start))
