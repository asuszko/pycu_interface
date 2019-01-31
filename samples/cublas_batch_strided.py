"""

"""

from functools import reduce
from operator import mul
import numpy as np
import sys
import os


dir_path = os.path.dirname(os.path.realpath(__file__))
rdr_path = os.path.dirname(dir_path)
sys.path.append(rdr_path)
from device import Device

# Sample data
a_shape = (64,128,128)
b_shape = (64,128,128)
c_shape = (64,128,128)

a = np.arange(0,reduce(mul,a_shape),1).reshape(a_shape).astype('c16')
b = np.arange(0,reduce(mul,b_shape),1).reshape(b_shape).astype('c16')

# Initialize the Device object on the default device(0) with 2 streams
with Device() as d:
    
    d_a = d.malloc(a_shape, dtype=a.dtype, fill=a)
    d_b = d.malloc(b_shape, dtype=b.dtype, fill=b)
    d_c = d.malloc(c_shape, dtype=b.dtype)
    
    d.cublas.gemm_strided_batched(d_a, d_b, d_c, OPA='T', OPB='C')
    
    c = d_c.to_host()


test_c = np.empty(c_shape, dtype=c.dtype)
for i in range(len(test_c)):
    test_c[i] = np.dot(a[i].T,b[i].conj().T)


print("GPU result is CPU result: %s" % str(np.allclose(c,test_c)))
