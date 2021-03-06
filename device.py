# -*- coding: utf-8 -*-
__all__ = [
    "Device",
]

import numpy as np
from ctypes import (cast,
                    c_void_p,
                    pointer,
                    sizeof)


# Local imports
from cuctx import cuCtx                #Context specific calls
from shared import (get_nbytes,
                    Shared)            #Shared calls between Device and Stream
from stream import Stream              #Stream specific calls
from dev_ptr import Device_Ptr
from uni_ptr import Unified_Ptr

from cublas_helpers import cublas
from cufft_helpers.cufft import cufft
from cuda_helpers import (cu_device_reset,
                          cu_device_props,
                          cu_get_mem_info,
                          cu_mempin,
                          cu_memunpin,
                          cu_sync_device)


class Device(Shared, object):

    def __init__(self, device_id=0, n_streams=0,
                 default_dtype='f4'):
        """
        CUDA device object. This object opens up, stores, and 
        controls a CUDA context. When the object is destroyed, 
        the context is destroyed, freeing up all device 
        resources that were used by the object.

        Parameters
        ----------
        device_id : int, optional
            The CUDA device ID.

        n_streams : int or list of ints, optional
            Number of CUDA streams per device object.
            
        default_dtype : np.dtype
            Default data type to use in mallocs.

        Attributes
        ----------
        context : c_void_p (CUcontext*)
            Pointer reference to the device context.

        streams : list of c_void_p
            List of pointer references to each CUDA stream.

        pinned_arrs : list of np.ndarray
            This object keeps track of the pinned host arrays.
            
        cublas: object
            The callable cuBLAS object.
            
        cufft : object
            The callable cuFFT object.
            
        props : deviceProps ctypes Structure
            The device properties structure defined by:
            http://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp
            
        Notes
        --------
        Not officially supported for CUDA devices with sm < 3.0.
        """
        super(Device, self).__init__()  
        
        self._id = device_id
        self._context = cuCtx(self)
        self._cublas = cublas()
        self._cufft = cufft()
        self._default_dtype = np.dtype(default_dtype)
        self._pinned_arrs = []
#        self._props = cu_device_props(self._id) Titan V broken
        self._props = None
        self._streams = [Stream(self, i) for i in range(n_streams)]


    def malloc(self, shape, dtype=None, fill=None, stream=None):
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
        Device_Ptr : Device_Ptr
            The object that holds the pointer to the memory.
        """
        dtype = dtype or self._default_dtype
        return Device_Ptr(shape, dtype, fill, stream)


    def malloc_unified(self, shape, dtype=None, fill=None, stream=None):
        """
        Allocates unified memory.

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
        Device_Ptr : Device_Ptr
            The object that holds the pointer to the unified memory.
        """
        dtype = dtype or self._default_dtype
        return Unified_Ptr(shape, dtype, stream, fill)


    def host_pin(self, arr, nbytes=None):
        """
        Page-lock the host memory.
        
        Parameters
        ----------
        arr : np.ndarray or c_void_p
            Host pointer or numpy array to page lock.
        
        nbytes : int, optional
            Size to pin in bytes. If None, the whole array is pinned.
        """   
        nbytes = get_nbytes(arr, nbytes)
        if type(arr) in [list,np.ndarray]:
            cu_mempin(arr.ctypes.data_as(c_void_p), nbytes)
        else:
            cu_mempin(cast(pointer(arr), c_void_p), nbytes)
        
        try:
            self._pinned_arrs.append(arr)
        except BaseException as e:
            print("Unknown error page-locking memory:")
            print(e)


    def host_unpin(self, arr):
        """
        Remove the page-lock from pinned host memory.

        Parameters
        ----------
        arr : list or np.ndarray
            The array the unpin. 
        """
        for i, pinned_arr in enumerate(self._pinned_arrs):
            if arr is pinned_arr:
                cu_memunpin(arr)
                self._pinned_arrs.pop(i)
                break
            else:
                print("Exception: Array not found in pinned memory")

   
    def host_unpin_all(self):
        """
        Remove the page-lock from all pinned host memory.
        """
        for i in range(len(self._pinned_arrs)):
            self.host_unpin(self._pinned_arrs[0])
     
   
    def query(self):
        """
        Query the device, and print information about the 
        device name, and the amount of free and used memory.
        
        Notes
        -----
        The operating system will use device memory, which this 
        routine reflects. Thus, seeing Free Mem < Total Mem even 
        without any memory allocations is expected.
        """
        free = np.array([1], dtype=np.uintp)
        total = np.array([1], dtype=np.uintp)
        cu_get_mem_info(free, total)
        free_f = float(free[0])/(1024.**2)
        total_f = float(total[0])/(1024.**2)
        name = self._props.name if self._props != None else "NVIDIA GPU"
        print("%s\n------------\n  Total Mem : %.2f (mb)\n  Free  Mem : %.2f (mb)"%(name,total_f,free_f))
        
    
    def require_streamable(self, *args):
        """
        Set a space of memory to be streamable. The ensures that 
        streaming works properly with the array by forcing it to 
        be C-contiguous, and pinned.

        Parameters
        ---------
        *args : nd.arrays
            Arrays passed in as separate args.
        """
        for arr in args:
            if type(arr) is np.ndarray:
                if not arr.flags['C_CONTIGUOUS']:
                    arr = np.require(arr, requirements="C")
                self.host_pin(arr)
            else:
                self.host_pin(arr, sizeof(arr)) #c-types struct/object

        
    def reset(self):
        """
        Reset the device context.
        """
        cu_device_reset()
        

    def set_context(self):
        """
        Sets the current active device to the one specified in this
        object's context.
        """
        self.context.push()


    def sync(self):
        """
        Block the host thread until the device has completed all tasks.
        """
        cu_sync_device()


    @property
    def id(self):
        return self._id
    

    @property
    def context(self):
        return self._context


    @property
    def cublas(self):
        return self._cublas        


    @property
    def cufft(self):
        return self._cufft
     
     
    @property
    def props(self):
        return self._props
    
     
    @property
    def streams(self):
        return self._streams


    def __enter__(self):
        return self


    def __exit__(self, *args, **kwargs):
        """
        Cleans up and frees the resources used by the object. 
        The CUDA context is destroyed, and any arrays that were 
        pinned are unpinned.
        """
        self.sync()
        self.host_unpin_all()
        self.context.__exit__()
        self.clear()