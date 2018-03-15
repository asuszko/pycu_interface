"""
This a simple sample code that creates the device object with 
2 streams, and runs the cuBLAS routine on both streams. Each stream 
handles half of the input data.

Note that the main benefit of streams is to hide  
device to host, and host to device memory transfers.
The compute heavy algorithm itself is parallelized at 
the kernel level, which in this case would be the 
cublas routine itself.

The numpy equivalent is:
nrms = np.linalg.norm(alpha*(alpha*a+b), axis=1)

Correct result is:
Norm from stream 0: 72.938332
Norm from stream 1: 193.183853
"""

import sys
import numpy as np

sys.path.append("..")
from device import Device

vec_len = 20
nstreams = 2

# Sample data
a = np.arange(0,vec_len,1).reshape(nstreams,vec_len//nstreams).astype('f4')
b = np.ones(a.shape,a.dtype)
n = len(a[0])

# Pre allocated space for the result
c = np.empty(a.shape,a.dtype)
nrms = np.empty(nstreams,a.dtype)

# Scaling factor that the cuBLAS routine called later uses
alpha = np.full(1,2,dtype=a.dtype)

# Initialize the Device object on the default device(0) with 2 streams
with Device(n_streams=nstreams) as d:
    
    # If using streams, relevant host memory needs to be pinned for async operation 
    d.require_streamable(a,b,c)
    
    # Allocate device memory for each stream to use, and return their references
    ## Mallocs are always synchronous, and should be done in their own loop to
    ## avoid synchorizing the otherwise async stream methods
    for s in d.streams:
        s.a = s.malloc(a[0].shape, a.dtype)
        s.b = s.malloc(b[0].shape, b.dtype)

    # Running the stream async operations    
    for stream_id,s in enumerate(d.streams):
        s.memcpy_h2d_async(s.a,a[stream_id])
        s.memcpy_h2d_async(s.b,b[stream_id])
        s.cublas.axpy(n, alpha, s.a, s.b)                        #ax plus y
        s.cublas.scal(n, alpha, s.b)                             #scale matrix by alpha
        s.memcpy_d2h_async(c[stream_id], s.b)                    #copy result back to host
        nrms[stream_id] = s.cublas.nrm2(n, s.b, dtype='f4')      #norm (result is automatically copied back to host)
        s.sync()


# Print result
for i,nrm in enumerate(nrms):
    print('Norm from stream %i: %f'%(i,nrm))