# -*- coding: utf-8 -*-
__all__ = [
    "Device_DblPtr",
]

from ctypes import cast, c_void_p
import numpy as np

from cuda_helpers import (cu_free, cu_malloc_dblptr)

dtype_map={np.dtype('f4') :0,
           np.dtype('f8') :1,
           np.dtype('c8') :2,
           np.dtype('c16'):3}


class Device_DblPtr(object):
    
    def __init__(self, device_ptr, n, batch_size):
        """
        Allocates space for a double pointer on the device.

        Parameters
        ----------
        device_ptr : Device_Ptr
            Original Device_Ptr object to map to double pointer.
        """
        dev_dblptr = cu_malloc_dblptr(device_ptr.ptr,
                                      n*n, batch_size,
                                      dtype_map[device_ptr.dtype])
        self.ptr = cast(dev_dblptr, c_void_p)
        self.batch_size = batch_size
        self.dtype = device_ptr.dtype

    
    def __call__(self):
        return self.ptr
    
    
    def __len__(self):
        return self.batch_size


    def __repr__(self):
        return repr(self.__dict__)
        
    
    def __enter__(self):
        return self


    def __exit__(self, *args, **kwargs):
        """
        Frees the memory used by the object, and then 
        deletes the object.
        """
        cu_free(self.ptr)
        del self