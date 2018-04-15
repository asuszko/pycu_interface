# -*- coding: utf-8 -*-
__all__ = [
    "Stream",
]

import numpy as np


# Local imports
from shared import (get_nbytes,
                    Shared)            #Shared calls between Device and Stream
from cublas_helpers import cublas
from cufft_helpers.cufft import cufft
from cuda_helpers import (cu_memcpy_3d_async,
                          cu_stream_create,
                          cu_sync_stream)


class Stream(Shared, object):

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


    def malloc(self, shape, dtype, stream=None, fill=None):
        """
        Allocates device memory.

        Parameters
        ----------
        shape : tuple
            The shape of the array to allocate.
            
        dtype : np.dtype
            That data type of the array.

        fill : scalar or np.ndarray, optional
            Default value to set allocated array to.
            
        stream : c_void_p, optional
            CUDA stream to associate the returned object with.
            
        Returns
        -------
        dev_ptr: c_void_p
            Pointer to allocated device memory.
        """
        stream = stream or self.stream
        return self.device.malloc(shape, dtype, fill, stream)


    def sync(self):
        """
        Block the host thread until the stream has completed its task.
        """
        cu_sync_stream(self.stream)


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