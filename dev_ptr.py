# -*- coding: utf-8 -*-
__all__ = [
    "Device_Ptr",
]

from functools import reduce
from operator import mul
import numpy as np
import warnings
from cuda_helpers import (cu_memcpy_d2d,
                          cu_memcpy_d2h,
                          cu_memcpy_h2d,
                          cu_memset,
                          cu_memcpy_d2d_async,
                          cu_memcpy_d2h_async,
                          cu_memcpy_h2d_async,
                          cu_memset_async,
                          cu_transpose)


dtype_map={np.dtype('f4') :0,
           np.dtype('f8') :1,
           np.dtype('c8') :2,
           np.dtype('c16'):3}


def check_contiguous(arr):
    if not arr.flags['C_CONTIGUOUS'] and not arr.flags['F_CONTIGUOUS']:
        warnings.warn("Non-contiguous host memory detected, unexpected behavior/results may occur")



class Device_Ptr(object):
    
    def __init__(self, ptr, shape, dtype, stream=None):
        
        self.ptr = ptr
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.stream = stream
        
        try:
            self.size = reduce(mul,shape)
        except:
            self.size = int(shape)
            
        self.nbytes = self.size*np.dtype(dtype).itemsize
    
    
    def __call__(self):
        return self.ptr
    
    
    def __len__(self):
        return self.size


    def __repr__(self):
        return repr(self.__dict__)
    
    
    def _harray(self):
        return np.empty(self.shape, self.dtype)


    def T(self, stream=None):
        """
        Transpose the matrix on the device. This currently only works 
        with square matrices. Rectangular will be implemented in a 
        future update.
        """
        try:
            nrows, ncols = self.shape
            cu_transpose(self.ptr,
                         nrows,
                         ncols,
                         dtype_map[self.dtype],
                         stream)
            self.shape = self.shape[::-1]
        except:
            warnings.warn("Transpose failed on array with ndim=%i."%len(self.shape))


    def d2d(self, dst_arr, nbytes=None):
        """
        Copy memory from device to device. This works both 
        for copying memory to a separate device, or creating a 
        copy of memory on the same device.

        Parameters
        ----------
        dst_arr : c_void_p
            Pointer of the destination array.

        nbytes : int
            Size to copy/transfer in bytes.
        """
        nbytes = nbytes or self.nbytes
        cu_memcpy_d2d(self.ptr, dst_arr, nbytes)
        

    def d2h(self, arr=None, nbytes=None):
        """
        Copy contiguous memory from the device to the host.

        Parameters
        ----------
        arr : list or np.ndarray
            Host array.

        nbytes : int, optional
            Size to transfer in bytes.
            
        Returns, optional
        -------
        tmp_arr : np.ndarray
            Returns the array from the device to newly allocated 
            existing memory, if none was passed in. Otherwise 
            the copy is done to arr.
            
        Notes
        -----
        Having arr created and pinned long beforehand will improve 
        overall performance. Using arr=None should only be used for 
        development, testing, and debugging purposes.
        """
        nbytes = nbytes or self.nbytes
        if arr is not None:
            check_contiguous(arr)
            cu_memcpy_d2h(self.ptr, arr, nbytes)
        else:
            tmp_arr = self._harray()
            cu_memcpy_d2h(self.ptr, tmp_arr, nbytes)
            return tmp_arr
            
    
    def h2d(self, arr, nbytes=None):
        """
        Copy memory from the host to the device.

        Parameters
        ----------
        arr : list or np.ndarray
            Host array.

        nbytes : int, optional
            Size to transfer in bytes.
        """
        nbytes = nbytes or self.nbytes
        check_contiguous(arr)
        cu_memcpy_h2d(self.ptr, arr, nbytes)
        
        
    def d2d_async(self, dst_arr, stream=None, nbytes=None):
        """
        Copy memory from device to device. This works both 
        for copying memory to a separate device, or creating a 
        copy of memory on the same device.

        Parameters
        ----------
        dst_arr : c_void_p
            Pointer of the destination array.
            
        stream : c_void_p
            CUDA stream pointer.

        nbytes : int
            Size to copy/transfer in bytes.
        """
        nbytes = nbytes or self.nbytes
        stream = stream or self.stream
        cu_memcpy_d2d_async(self.ptr, dst_arr, nbytes, stream)


    def d2h_async(self, arr=None, stream=None, nbytes=None):
        """
        Copy contiguous memory from the device to the host.

        Parameters
        ----------
        arr : list or np.ndarray
            Host array.
            
        stream : c_void_p
            CUDA stream pointer.
        
        nbytes : int, optional
            Size to transfer in bytes.
            
        Returns, optional
        -------
        tmp_arr : np.ndarray
            Returns the array from the device to newly allocated 
            existing memory, if none was passed in. Otherwise 
            the copy is done to arr.
            
        Notes
        -----
        Having arr created and pinned long beforehand will improve 
        overall performance. Using arr=None should only be used for 
        development, testing, and debugging purposes.
        """
        nbytes = nbytes or self.nbytes
        stream = stream or self.stream
    
        if arr is not None:
            check_contiguous(arr)
            cu_memcpy_d2h_async(self.ptr, arr, nbytes, stream)
        else:
            tmp_arr = self._harray()
            cu_memcpy_d2h_async(self.ptr, arr, nbytes, stream)
            return tmp_arr

                
    def h2d_async(self, arr, stream=None, nbytes=None):
        """
        Copy contiguous memory from the host to the device.

        Parameters
        ----------
        arr : list or np.ndarray
            Host array.

        stream : c_void_p
            CUDA stream pointer.
               
        nbytes : int, optional
            Size to transfer in bytes.
        """
        nbytes = nbytes or self.nbytes
        stream = stream or self.stream
        check_contiguous(arr)
        cu_memcpy_h2d_async(self.ptr, arr, nbytes, stream)
  
      
    def zero(self, nbytes=None):
        """
        Zero out the values in the array.

        Parameters
        ----------
        nbytes : int
            Size to set in bytes.
        """
        nbytes = nbytes or self.nbytes
        cu_memset(self.ptr, 0, nbytes)
        
    
    def zero_async(self, stream=None, nbytes=None):
        """
        Zero out the values in the array.

        Parameters
        ----------
        stream : c_void_p
            CUDA stream pointer.
        
        nbytes : int
            Size to set in bytes.
        """
        stream = stream or self.stream
        nbytes = nbytes or self.nbytes
        cu_memset_async(self.ptr, 0, nbytes, stream)