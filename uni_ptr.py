# -*- coding: utf-8 -*-
__all__ = [
    "Unified_Ptr",
]

from ctypes import cast, c_void_p
from functools import reduce
from operator import mul
import numpy as np
import warnings

from cuda_helpers import (cu_conj,
                          cu_free,
                          cu_iadd_val,
                          cu_iadd_vec,
                          cu_idiv_val,
                          cu_idiv_vec,
                          cu_imul_val,
                          cu_imul_vec,
                          cu_isub_val,
                          cu_isub_vec,
                          cu_malloc_managed,
                          cu_memcpy_d2d,
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

c2f_map={np.dtype('f4') : 0,
         np.dtype('f8') : 1,
         np.dtype('c8') : 0,
         np.dtype('c16'): 1}


def check_contiguous(arr):
    if not arr.flags['C_CONTIGUOUS'] and not arr.flags['F_CONTIGUOUS']:
        warnings.warn("Non-contiguous host memory detected, unexpected behavior/results may occur.")


class Unified_Ptr(object):
    
    def __init__(self, shape, dtype, stream=None, fill=None):
        """
        Allocates device memory, holds important information, 
        and provides useful operations.

        Parameters
        ----------
        shape : tuple
            The shape of the array to allocate.
            
        dtype : np.dtype
            That data type of the array.
            
        stream : c_void_p
            CUDA stream to associate the returned object with.
            
        fill : scalar, np.ndarray, or Device_Ptr, optional
            Default value to fill in allocated memory space. If 
            None, then the memory is allocated with zeros.
        """
        
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.stream = stream
        
        try:
            self.size = reduce(mul,shape)
        except:
            self.size = int(shape)
        
        self.nbytes = self.size*self.dtype.itemsize
        dev_ptr = cu_malloc_managed(self.nbytes)
        self.ptr = cast(dev_ptr, c_void_p)
        self.h = np.ctypeslib.as_array(cast(dev_ptr,
                                            np.ctypeslib.ndpointer(dtype,
                                                                   shape=shape,
                                                                   flags='C')))
        
        if fill is not None:
            if isinstance(fill, (int, float, complex)):
                self.h[:] = fill
            elif type(fill) in [list,tuple]:
                self.h[:] = fill
            elif type(fill) == np.ndarray:
                self.h[:] = fill.reshape(self.shape)
            elif type(fill) == type(self):
                self.d2d(src=fill, dst=self)
            else:
                raise TypeError('Unsupported fill value or type input')


    def __call__(self):
        return self.ptr
    
    
    def __len__(self):
        return self.size


    def __repr__(self):
        return repr(self.__dict__)

    
    def __getitem__(self, slice):
        return self.h[slice]
        

    def __setitem__(self, slice, b):
        self.h[slice] = b
        

    def __iadd__(self, b):
        """
        Perform in-place element-wise addition.
        
        Parameters
        ----------
        b : np.ndarray or scalar
            Value(s) to add by
        
        Returns
        -------
        self : Device_Ptr
            Returns self with updated values in self.ptr
        """
        self.h += b
        return self


    def __imul__(self, b):
        """
        Perform in-place element-wise multiplication.
        
        Parameters
        ----------
        b : np.ndarray or scalar
            Value(s) to multiply by
        
        Returns
        -------
        self : Device_Ptr
            Returns self with updated values in self.ptr
        """
        self.h *= b
        return self

    
    def __isub__(self, b):
        """
        Perform in-place element-wise subtraction.
        
        Parameters
        ----------
        b : np.ndarray or scalar
            Value(s) to subtract by
        
        Returns
        -------
        self : Device_Ptr
            Returns self with updated values in self.ptr
        """
        self.h -= b
        return self


    def __itruediv__(self, b):
        """
        Perform in-place element-wise division.
        
        Parameters
        ----------
        b : np.ndarray or scalar
            Value(s) to divide by
        
        Returns
        -------
        self : Device_Ptr
            Returns self with updated values in self.ptr
        """
        self.h /= b
        return self


    def T(self, stream=None):
        """
        Transpose the matrix on the device. This currently only works 
        with square matrices. Rectangular will be implemented in a 
        future update.
        """
        self.h = self.h.T
        self.shape = self.h.shape
        return self


    def conj(self, inplace=True, stream=None):
        """
        Take and return the complex conjugate.
        """
        self.h = self.h.conj
        return self
    

    @classmethod
    def d2d(self, src, dst, nbytes=None):
        """
        Copy memory from 'device to device'. This works both 
        for copying memory to a separate device, or for creating a 
        copy of memory on the same device.
        
        Parameters
        ----------
        src : Device_Ptr
            Device_Ptr object containing the src ptr.
        
        dst : Device_Ptr
            Device_Ptr object containing the dst ptr.

        nbytes : int
            Size to copy/transfer in bytes.
        """
        nbytes = min([src.nbytes, nbytes or src.nbytes])
        if nbytes > dst.nbytes:
            raise ValueError('Attempted to copy a src with size greater than dst.')
        cu_memcpy_d2d(src.ptr, dst.ptr, nbytes)
            

    @property
    def dtype_depth(self):
        if self.dtype in ['f4', 'f8']:
            return 1
        if self.dtype in ['c8', 'c16']:
            return 2
    
    
    def __enter__(self):
        return self


    def __exit__(self, *args, **kwargs):
        """
        Frees the memory used by the object, and then 
        deletes the object.
        """
        cu_free(self.ptr)
        del self