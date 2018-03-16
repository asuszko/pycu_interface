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


from cublas_helpers import cublas
from cufft_helpers.cufft import cufft
from cuda_helpers import (cu_device_reset,
                          cu_device_props,
                          cu_get_mem_info,
                          cu_memcpy_d2d,
                          cu_memcpy_d2h,
                          cu_memcpy_h2d,
                          cu_memcpy_3d,
                          cu_mempin,
                          cu_memunpin,
                          cu_memset,
                          cu_sync_device)


class Device(Shared, object):

    def __init__(self, device_id=0, n_streams=0):
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
        self._pinned_arrs = []
        self._props = cu_device_props(self._id)
        self._streams = [Stream(self, i) for i in range(n_streams)]


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
        
 
    def memcpy_h2d(self, d_arr, arr, nbytes=None):
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
        nbytes = get_nbytes(arr, nbytes)
        cu_memcpy_h2d(d_arr, arr, nbytes)


    def memcpy_3d(self, src, dst, extent, size):
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
        cu_memcpy_3d(src, dst, extent, size)
        return

 
    def memcpy_d2h(self, arr, d_arr, nbytes=None):
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
        nbytes = get_nbytes(arr, nbytes)
        cu_memcpy_d2h(d_arr, arr, nbytes)


    def memcpy_d2d(self, src_arr, dst_arr, nbytes):
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
        cu_memcpy_d2d(src_arr, dst_arr, nbytes)
        
    
    def memset(self, d_arr, value, nbytes):
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
        cu_memset(d_arr, value, nbytes)

   
    def query(self):
        """
        Query the device, and print information about the 
        device name, and the amount of free and used memory.
        """
        free = np.array([1], dtype=np.uintp)
        total = np.array([1], dtype=np.uintp)
        cu_get_mem_info(free, total)
        free_f = float(free[0])/(1024.**2)
        total_f = float(total[0])/(1024.**2)
        print("%s\n------------\n  Total Mem : %.2f (mb)\n  Free Mem  : %.2f (mb)"%(self._props.name,total_f,free_f))
        
    
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
        for i in range(len(self._pinned_arrs)):
            self.host_unpin(self._pinned_arrs[0])
        self.context.__exit__()
        self.__dict__.clear()