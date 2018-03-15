"""
Ceates a device object on the default (null) stream, and 
then runs a few cuFFT routines.

Notes:
-    Plan overhead is large on GPUs, so you likely won't 
     see much of speedup doing just a single fft. Batched ffts that can 
     reuse the same plan will see much more speedup.

-    cuFFT routines do not scale the C2C FFTs by the array 
     size. That must either be done with a subsequent cuBLAS 
     call (cublas_scal) or via Python after copying back to host.
"""

import sys
import numpy as np
from scipy import misc
from matplotlib.pylab import plt


sys.path.append("..")
from device import Device



f = misc.face()                                         # Load an image
gray = np.dot(f,[0.2989,0.5870,0.1140]).astype('f4')    # Convert to grayscale
ny, nx = gray.shape
fft_extent = np.array([nx,ny,1],dtype='i4')             # Needed as fft plan argument later
result = np.empty((ny,nx), dtype='c8')                  # Pre allocated space for fft result




with Device() as d:

    d.r2c_plan = d.cufft.plan(fft_extent, 'cufft_r2c')              #fft r2c plan single precision (double = cufft_d2z)
    d.c2c_plan = d.cufft.plan(fft_extent, 'cufft_c2c')              #fft c2c plan single precision (double = cufft_z2z)

    d.gray = d.malloc(gray.shape,gray.dtype)                        #Space of input data
    d.r2c_res = d.malloc((ny,(nx//2+1)),'c8')                       #Space for r2c result
    d.r2c_full= d.malloc((ny,nx),'c8')                              #Space for full rest
    d.c2c_res = d.malloc((ny,nx),'c8')                              #space for c2c result
     
    d.memcpy_h2d(d.gray, gray)                                      #Copy input data to device
    d.cufft.r2c(d.r2c_plan, d.gray, d.r2c_res)                      #r2c fft
    d.cufft.add_redundants(d.r2c_plan, d.r2c_res, d.r2c_full)       #Custom kernel that fills in the redundant values
    d.cufft.c2c(d.c2c_plan, d.r2c_full, d.c2c_res, 'cufft_inverse') #c2c fft
    
    alpha = np.array([1./result.size], dtype=gray.dtype)            
    d.cublas.scal(2*result.size, alpha, d.c2c_res)                  #Scale c2c result using cuBLAS
    
    d.memcpy_d2h(result, d.c2c_res)                                 #Copy scaled c2c result back to host



## Using the device calls are all synchronous so the result will be ready 
## by the time numpy operates on it.
fig = plt.figure()
plt.imshow(result.real)
plt.colorbar()
plt.show()