# -*- coding: utf-8 -*-
__all__ = [
    "get_nbytes",
    "Shared",
]

from functools import reduce
from operator import mul
import numpy as np
from ctypes import cast, c_void_p
import warnings

# Local imports
from cuda_helpers import (cu_create_channel_char,
                          cu_create_channel_short,
                          cu_create_channel_float,
                          cu_malloc,
                          cu_malloc_3d,
                          cu_malloc_managed)
from dev_ptr import Device_Ptr
from uni_ptr import Unified_Ptr
from shared_utils import Mapping




def get_nbytes(arr, nbytes=None):
    if type(arr) in [list or tuple]:
        nbytes = nbytes or np.array(arr).nbytes
    else:
        nbytes = nbytes or arr.nbytes
    return nbytes


class Shared(Mapping, object):

    def __init__(self):
        """
        This object holds methods that are callable by both the 
        Device and stream Objects.
        
        Notes
        -----
        Calling these methods from the Stream object may cause 
        thread host blocking behavior and break the asynchronous 
        stream operation.
        """
        super(Mapping, self).__init__()


    def create_channel(self, dtype, components=1, unsigned=False):
        """
        Create channel information used by a CUDA texture.
        
        Parameters
        ----------
        dtype : int
            dtype identifier.
            
        components : int, optional
            Number of components to create in the channel.
            
        unsigned : bool, optional
            Unsigned flag.
            
        Returns
        -------
        channelDesc : ctypes Structure
            CUDA channel object that is passed into the CUDA 2/3d
            memcpy functions.
        """
        return {
            0: cu_create_channel_char(components, unsigned),
            1: cu_create_channel_short(components, unsigned),
            2: cu_create_channel_float(components, unsigned),
        }.get(dtype, 0)


    def malloc_3d(self, channel, extent, layered=False):

        """
        Allocates device memory as a CUDA array.

        Parameters
        ----------
        nbytes : int
            Size to allocate in bytes.
        
        extent : list, 1d np.ndarray, or tuple
            The extent or dimensions of the array.
            Format: (nx,ny,nz,...)
            
        layered : bool, optional
            Layered array flag.
            
        Returns
        -------
        dev_ptr: c_void_p
            Pointer to allocated device memory.
            
        Notes
        -----
        Setting the layered flag to True will turn a 3D 
        array into a 2D layered array.
        """
        if type(extent) in [list, tuple]:
            extent = np.array(extent, dtype='i4')
        dev_ptr = cu_malloc_3d(channel, extent, layered)
        dev_ptr = cast(dev_ptr, c_void_p)
        return dev_ptr