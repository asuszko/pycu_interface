import numpy as np
import sys
import os

# Imports come from here
dir_path = os.path.dirname(os.path.realpath(__file__))
rdr_path = os.path.dirname(dir_path)
sys.path.append(rdr_path)
from device import Device

# Generate fake test data
h_A = np.random.random((40, 256)).astype('c8')
h_B = np.random.random((40, 40)).astype('c8')

# Add to imaginary part if complex
if np.iscomplexobj(h_A):
    h_A.imag += np.random.random(h_A.shape)
if np.iscomplexobj(h_B):
    h_B.imag += np.random.random(h_B.shape)

# @ h_A.conj().T @ h_R
h_C = np.zeros((h_A.shape[-1], h_B.shape[0]), dtype=h_B.dtype)


with Device() as d:
    d_A = d.malloc(h_A.shape, fill=h_A, dtype=h_A.dtype)
    d_B = d.malloc(h_B.shape, fill=h_B, dtype=h_B.dtype)
    d_C = d.malloc(h_C.shape, dtype=h_C.dtype)

    d_hA = d_A.to_host()
    d_hB = d_B.to_host()
    d_hC = d_C.to_host()
    
    # Memory checks out
    print("Memory d_A OK: %s" % np.array_equal(d_hA, h_A))
    print("Memory d_R OK: %s" % np.array_equal(d_hB, h_B))
    print("Memory d_C OK: %s" % np.array_equal(d_hC, h_C))

    # Run on GPU
    d.cublas.gemm(d_A, d_B, d_C, OPA='C', m3m=False)
    d_hC = d_C.to_host()
    
    # Run on CPU
    h_C = h_A.conj().T @ h_B

    # Relative error of magnitudes
    rel_error = np.abs(np.abs(d_hC) - np.abs(h_C)) / np.abs(h_C)

    print("Mean Error %.8f" % np.mean(rel_error))
    print("Max Error %.8f" % np.max(rel_error))
