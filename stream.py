# -*- coding: utf-8 -*-
__all__ = [
    "Stream",
]

import numpy as np
from ctypes import cast, c_void_p

# Path must be appended for Python 3.x
import os
import sys
sys.path.append(os.path.join(os.getcwd(),"cuda_helpers"))
sys.path.append(os.path.join(os.getcwd(),"cublas_helpers"))
sys.path.append(os.path.join(os.getcwd(),"cufft_helpers"))

# Local imports
from shared import Shared
from cublas_helpers import cublas
from cufft_helpers.cufft import cufft
from cuda_helpers import (cu_memcpy_3d_async,
                          cu_memcpy_d2d_async,
                          cu_memcpy_d2h_async,
                          cu_memcpy_h2d_async,
                          cu_memset_async,
                          cu_stream_create,
                          cu_sync_stream)


class Stream(Shared):

    def __init__(self, device, stream_id):
        """
        CUDA stream object.
        
        Parameters
        ----------
        device : object
            The CUDA device object.

        stream_id : int
            The CUDA set stream id.


        Attributes
        ----------
        id : int
            Stream ID.

        stream : c_void_p (cudaStream_t*)
            Pointer to the CUDA Stream handle.
            
        cublas: object
            The callable cuBLAS object.
            
        cufft : object
            The callable cuFFT object.
        """
        super(Stream, self).__init__() 

        self._device = device
        self._stream = cu_stream_create()
        self._id = stream_id
        self._cublas = cublas(self.stream)
        self._cufft = cufft(self.stream)
  

    def memcpy_3d_async(self, src, dst, extent, size):
        """
        Copy a CUDA array.

        Parameters
        ----------
        src_arr : c_void_p
            Pointer of the source array to copy.

        d_arr : c_void_p
            Pointer of the destination array.
            
        extent : list or ndarray
            Dimensions of the array to copy [x,y,z].

        size : int
            Size of the array in bytes to copy.
        """
        if type(extent) is (list or tuple):
            extent = np.array(extent, dtype='i4')
        cu_memcpy_3d_async(src, dst, extent, size, self.stream)
        return


    def memcpy_d2d_async(self, src_arr, dst_arr, nbytes):
        """
        Copy memory from device to device. This works both 
        for copying memory to a separate device, or creating a 
        copy of memory on the same device.

        Parameters
        ----------
        src_arr : c_void_p
            Pointer of the source array to copy.

        d_arr : c_void_p
            Pointer of the destination array.

        nbytes : int
            Size to copy/transfer in bytes.
        """
        cu_memcpy_d2d_async(src_arr, dst_arr, nbytes, self.stream)


    def memcpy_d2h_async(self, arr, d_arr, nbytes=None):
        """
        Copy contiguous memory from the device to the host.

        Parameters
        ----------
        arr : list or np.ndarray
            Host array.

        d_arr : c_void_p
            Device pointer reference.
        
        nbytes : int, optional
            Size to transfer in bytes.
        """
        nbytes = nbytes or arr.nbytes
        cu_memcpy_d2h_async(d_arr, arr, nbytes, self.stream)
        

    def memcpy_h2d_async(self, d_arr, arr, nbytes=None):
        """
        Copy contiguous memory from the host to the device.

        Parameters
        ----------
        d_arr : c_void_p
            Device pointer reference.

        arr : list or np.ndarray
            Host array.

        nbytes : int, optional
            Size to transfer in bytes.
        """
        nbytes = nbytes or arr.nbytes
        cu_memcpy_h2d_async(d_arr, arr, nbytes, self.stream)

  
    def memset_async(self, d_arr, value, nbytes):
        """
        Set the value of the memory. Currently, the only values that 
        are 'obvious' to set are 0, and -1. Otherwise, other values 
        must set in hex.

        Parameters
        ----------
        d_arr : c_void_p
            Pointer of the memory on the device to set.

        nbytes : int
            Size to set in bytes.
        """
        cu_memset_async(d_arr, value, nbytes, self.stream)
        return


    def sync(self):
        """
        Block the host thread until the stream has completed its task.
        """
        cu_sync_stream(self.stream)
        return


    @property
    def cublas(self):
        return self._cublas        


    @property
    def cufft(self):
        return self._cufft


    @property
    def device(self):
        return self._device
    
     
    @property
    def id(self):
        return self.stream_id


    @property
    def stream(self):
        return self._stream


    def __contains__(self, key):
        return key in self.__dict__


    def __getitem__(self, key):
        return self.__dict__[key]
    
    
    def __len__(self):
        return len(self.__dict__)


    def __repr__(self):
        return repr(self.__dict__)


    def __setitem__(self, key, value):
        self.__dict__[key] = value