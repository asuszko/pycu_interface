# -*- coding: utf-8 -*-

__all__ = [
    "load_lib"
]

import platform
from numpy.ctypeslib import load_library


def load_lib(lib_path, lib_fname):
    """
    Load a shared library.
    
    Parameters
    ----------
    lib_path : str
        Absolute path of the shared library.
    
    lib_fname : str
        File name of the shared library.
        
    Returns
    -------
    """   
    if platform.system() == 'Linux':
        c_lib = load_library(lib_fname+".so", lib_path)
    elif platform.system() == 'Windows':
        c_lib = load_library(lib_fname+".dll", lib_path)
    return c_lib