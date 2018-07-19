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

from functools import reduce
from operator import mul
import numpy as np
import sys


sys.path.append("..")
from device import Device

# Sample data
a_shape = (8,8,8)
b_shape = (8,8,8)
c_shape = (8,8,8)

a = np.arange(0,reduce(mul,a_shape),1).reshape(a_shape).astype('f4')
b = np.arange(0,reduce(mul,b_shape),1).reshape(b_shape).astype('f4')

# Initialize the Device object on the default device(0) with 2 streams
with Device() as d:
    
    d_a = d.malloc(a_shape, dtype=a.dtype, fill=a)
    d_b = d.malloc(b_shape, dtype=b.dtype, fill=b)
    d_c = d.malloc(c_shape, dtype=b.dtype)
    
    d.cublas.gemm_strided_batched(d_a,d_b,d_c)
    
    c = d_c.to_host()


test_c = np.empty(c_shape, dtype=c.dtype)
for i in range(len(test_c)):
    test_c[i] = np.dot(a[i],b[i])


print(np.array_equal(c,test_c))
