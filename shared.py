# -*- coding: utf-8 -*-
__all__ = [
    "Shared",
]

import numpy as np
from ctypes import cast, c_void_p

# Path must be appended for Python 3.x
import os
import sys
sys.path.append(os.path.join(os.getcwd(),"cuda_helpers"))

# Local imports
from cuda_helpers import (cu_create_channel_char,
                          cu_create_channel_short,
                          cu_create_channel_float,
                          cu_malloc,
                          cu_malloc_3d)


class Shared(object):

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
        pass


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


    def malloc(self, nbytes):
        """
        Allocates device memory.

        Parameters
        ----------
        nbytes : int
            Size to allocate in bytes.
            
        Returns
        -------
        dev_ptr: c_void_p
            Pointer to allocated device memory.
        """
        dev_ptr = cu_malloc(nbytes)
        dev_ptr = cast(dev_ptr, c_void_p)
        return dev_ptr


    def malloc_3d(self, channel, extent, layered=False):

        """
        Allocates device memory as a CUDA array.

        Parameters
        ----------
        nbytes : int
            Size to allocate in bytes.
        
        extent : list or np.ndarray
            The extent or dimensions of the array [x,y,z].
            
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
        if extent is list:
            extent = np.array(extent, dtype='i4')
        dev_ptr = cu_malloc_3d(channel, extent, layered)
        dev_ptr = cast(dev_ptr, c_void_p)
        return dev_ptr