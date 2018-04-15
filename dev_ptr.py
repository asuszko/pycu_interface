# -*- coding: utf-8 -*-
__all__ = [
    "Device_Ptr",
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
                          cu_malloc,
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


def check_input(a,b):
    if not a.dtype == b.dtype:
        warnings.warn("Attempting arithmetic on arrays with dtypes that are not equal, unexpected behavior/results may occur.")
    if not a.shape == b.shape:
        warnings.warn("Attempting arithmetic on arrays with shapes that are not equal, unexpected behavior/results may occur.")


class Device_Ptr(object):
    
    def __init__(self, shape, dtype, fill=None, stream=None):
        """
        Allocates device memory, holds important information, 
        and provides useful operations.

        Parameters
        ----------
        shape : tuple
            The shape of the array to allocate.
            
        dtype : np.dtype
            That data type of the array.

        fill : scalar, np.ndarray, or Device_Ptr, optional
            Default value to fill in allocated memory space. If 
            None, then the memory is allocated with zeros.
            
        stream : c_void_p
            CUDA stream to associate the returned object with.
        """
        
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.stream = stream
        
        try:
            self.size = reduce(mul,shape)
        except:
            self.size = int(shape)
        
        self.nbytes = self.size*self.dtype.itemsize
        dev_ptr = cu_malloc(self.nbytes)
        self.ptr = cast(dev_ptr, c_void_p)
        
        if fill is not None:
            if isinstance(fill, (int, float, complex)):
                tmp_arr = np.full(shape, fill, dtype=self.dtype)
                self.to_device(tmp_arr)
                del tmp_arr
            elif type(fill) in [list,tuple]:
                tmp_arr = np.array(fill, dtype=self.dtype)
                self.to_device(tmp_arr, tmp_arr.nbytes)
                del tmp_arr
            elif type(fill) == type(self):
                self.d2d(src=fill, dst=self)
            elif type(fill) == np.ndarray:
                fill = np.require(fill, dtype=self.dtype, requirements='C')
                self.to_device(fill, fill.nbytes)
            else:
                raise TypeError('Unsupported fill value or type input')

    
    def __call__(self):
        return self.ptr
    
    
    def __len__(self):
        return self.size


    def __repr__(self):
        return repr(self.__dict__)
        

    def __iadd__(self, b):
        """
        Perform in-place element-wise addition.
        
        Parameters
        ----------
        b : Device_Ptr
            Device pointer object to add to self. This object 
            contains the reference to the device memory 
            where the values to be added are stored.
        
        Returns
        -------
        self : Device_Ptr
            Returns self with updated values in self.ptr
        """
        if type(b) == type(self):
            check_input(self,b)
            cu_iadd_vec(self.ptr,
                        b.ptr,
                        self.size,
                        c2f_map[self.dtype],
                        self.dtype_depth,
                        self.stream)
        elif isinstance(b, (int, float, complex)):
            cu_iadd_val(self.ptr,
                        np.array([b], dtype=self.dtype),
                        self.size,
                        c2f_map[self.dtype],
                        self.dtype_depth,
                        self.stream)
        else:
            raise TypeError("Invalid type in _iadd_")
        return self


    def __imul__(self, b):
        """
        Perform in-place element-wise multiplication.
        
        Parameters
        ----------
        b : Device_Ptr
            Device pointer object to multiply to self. This 
            object contains the reference to the device memory 
            where the values to be multiplied are stored.
        
        Returns
        -------
        self : Device_Ptr
            Returns self with updated values in self.ptr
        """
        if type(b) == type(self):
            check_input(self,b)
            cu_imul_vec(self.ptr,
                        b.ptr,
                        self.size,
                        c2f_map[self.dtype],
                        self.dtype_depth,
                        self.stream)
        elif isinstance(b, (int, float, complex)):
            cu_imul_val(self.ptr,
                        np.array([b], dtype=self.dtype),
                        self.size,
                        c2f_map[self.dtype],
                        self.dtype_depth,
                        self.stream)
        else:
            raise TypeError("Invalid type in _imul_")
        return self

    
    def __isub__(self, b):
        """
        Perform in-place element-wise subtraction.
        
        Parameters
        ----------
        b : Device_Ptr
            Device pointer object to subtract to self. This 
            object contains the reference to the device memory 
            where the values to be subtract are stored.
        
        Returns
        -------
        self : Device_Ptr
            Returns self with updated values in self.ptr
        """
        if type(b) == type(self):
            check_input(self,b)
            cu_isub_vec(self.ptr,
                        b.ptr,
                        self.size,
                        c2f_map[self.dtype],
                        self.dtype_depth,
                        self.stream)
        elif isinstance(b, (int, float, complex)):
            cu_isub_val(self.ptr,
                        np.array([b], dtype=self.dtype),
                        self.size,
                        c2f_map[self.dtype],
                        self.dtype_depth,
                        self.stream)
        else:
            raise TypeError("Invalid type in _isub_")
        return self


    def __itruediv__(self, b):
        """
        Perform in-place element-wise multiplication.
        
        Parameters
        ----------
        b : Device_Ptr
            Device pointer object to divide to self. This 
            object contains the reference to the device memory 
            where the values to be divided are stored.
        
        Returns
        -------
        self : Device_Ptr
            Returns self with updated values in self.ptr
        """
        if type(b) == type(self):
            check_input(self,b)
            cu_idiv_vec(self.ptr,
                        b.ptr,
                        self.size,
                        c2f_map[self.dtype],
                        self.dtype_depth,
                        self.stream)
        elif isinstance(b, (int, float, complex)):
            cu_idiv_val(self.ptr,
                        np.array([b], dtype=self.dtype),
                        self.size,
                        c2f_map[self.dtype],
                        self.dtype_depth,
                        self.stream)
        else:
            raise TypeError("Invalid type in _imul_")
        return self


    def T(self, stream=None):
        """
        Transpose the matrix on the device. This currently only works 
        with square matrices. Rectangular will be implemented in a 
        future update.
        """
        stream = stream or self.stream
#        try:
        nrows, ncols = self.shape
        cu_transpose(self.ptr,
                     nrows,
                     ncols,
                     dtype_map[self.dtype],
                     stream)
        self.shape = self.shape[::-1]
#        except:
#            warnings.warn("Transpose failed on array with ndim=%i."%len(self.shape))


    def conj(self, inplace=True, stream=None):
        """
        Take and return the complex conjugate.
        """
        if self.dtype == (np.dtype('c8') or np.dtype('c16')):
            stream = stream or self.stream
            if inplace:
                cu_conj(self.ptr, self.size, dtype_map[self.dtype], stream)
                return self
            else:
                new_Device_Ptr = Device_Ptr(self.shape,
                                            self.dtype,
                                            stream=stream,
                                            fill=self)
                new_Device_Ptr.conj()
                return new_Device_Ptr
    

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
        

    def to_host(self, arr=None, nbytes=None):
        """
        Copy contiguous memory from the device to the host.
        'device to host'

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
        nbytes = min([self.nbytes, nbytes or self.nbytes])
        if arr is not None:
            check_contiguous(arr)
            cu_memcpy_d2h(self.ptr, arr, nbytes)
        else:
            tmp_arr = np.empty(self.shape, self.dtype)
            cu_memcpy_d2h(self.ptr, tmp_arr, nbytes)
            return tmp_arr
            
    
    def to_device(self, arr, nbytes=None):
        """
        Copy memory from the host to the device.
        'host to device'

        Parameters
        ----------
        arr : list or np.ndarray
            Host array.

        nbytes : int, optional
            Size to transfer in bytes.
        """
        nbytes = min([self.nbytes, nbytes or self.nbytes])
        check_contiguous(arr)
        cu_memcpy_h2d(self.ptr, arr, nbytes)
        
    
    @classmethod
    def d2d_async(self, src, dst, stream=None, nbytes=None):
        """
        Copy memory from device to device. This works both 
        for copying memory to a separate device, or creating a 
        copy of memory on the same device.

        Parameters
        ----------
        src : Device_Ptr
            Device_Ptr object containing the src ptr.
            
        dst : Device_Ptr
            Device_Ptr object containing the dst ptr.
            
        stream : c_void_p
            CUDA stream pointer.

        nbytes : int
            Size to copy/transfer in bytes.
        """
        nbytes = min([src.nbytes, nbytes or src.nbytes])
        stream = stream or src.stream
        if nbytes > dst.nbytes:
            raise ValueError('Attempted to copy a src with size greater than dst.')
        cu_memcpy_d2d_async(src.ptr, dst.ptr, nbytes, stream)
            

    def to_host_async(self, arr=None, stream=None, nbytes=None):
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
        nbytes = min([self.nbytes, nbytes or self.nbytes])
        stream = stream or self.stream
    
        if arr is not None:
            check_contiguous(arr)
            cu_memcpy_d2h_async(self.ptr, arr, nbytes, stream)
        else:
            tmp_arr = np.empty(self.shape, self.dtype)
            cu_memcpy_d2h_async(self.ptr, arr, nbytes, stream)
            return tmp_arr

                
    def to_device_async(self, arr, stream=None, nbytes=None):
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
        nbytes = min([self.nbytes, nbytes or self.nbytes])
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
        nbytes = min([self.nbytes, nbytes or self.nbytes])
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
        nbytes = min([self.nbytes, nbytes or self.nbytes])
        stream = stream or self.stream
        cu_memset_async(self.ptr, 0, nbytes, stream)


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