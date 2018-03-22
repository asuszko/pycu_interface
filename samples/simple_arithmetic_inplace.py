# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 13:03:37 2018

@author: pj8pgp
"""

import sys
sys.path.append("..")
from device import Device

import numpy as np


with Device() as d:

    # Allocating two complex arrays, with default fill value
    d.a = d.malloc((768,512), dtype='c16', default=3j+5)
    d.b = d.malloc(d.a.shape, dtype='c16', default=1j+16)
    
    # Some simple in-place arithmetic
    d.b += d.a
    d.b *= d.a
    d.a -= d.b
    d.a /= d.b
    
    #Query memory usage for fun
    d.query()
    
    result = d.a.to_host()