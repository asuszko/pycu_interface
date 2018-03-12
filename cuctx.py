# -*- coding: utf-8 -*-
__all__ = [
    "cuCtx",
]

from ctypes import cast, c_void_p

# Path must be appended for Python 3.x
import os
import sys
sys.path.append(os.path.join(os.getcwd(),"cuda_helpers"))

# Local imports
from cuda_helpers import (cu_context_create,
                          cu_context_destroy,
                          cu_context_pop,
                          cu_context_push)


class cuCtx(object):

    def __init__(self, device):
        """
        CUDA context object.

        Parameters
        ----------
        device : object
            CUDA device object.

        Attributes
        ----------
        context : c_void_p (CUcontext*)
            Pointer to the CUDA device context.
            
        Notes
        -----
        General use of this framework will never require directly calling 
        or using object, and should only be explicitly called for 
        special and advanced use cases.
        """
        self._context = cu_context_create(device.id)

        
    def __call__(self, method, *args):
        """
        Push the current CUDA context to the stack, call a method, 
        and then pop the context from the stack.
        
        Parameters
        ----------
        method : func
            The function to call.
            
        *args : list
            The parameters to pass into method
            
        Returns
        -------
        """
        self.push()
        tmp = method(*args)
        self.pop()
        return tmp
        
  
    def pop(self):
        """
        Pop the current CUDA context from the stack.
        """
        cu_context_pop(self.context)
        
        
    def push(self):
        """
        Push the current CUDA context to the stack.
        """
        cu_context_push(self.context)
 

    @property
    def context(self):
        return self._context

        
    def __enter__(self):
        return self


    def __exit__(self, *args, **kwargs):
        cu_context_destroy(self._context)