# -*- coding: utf-8 -*-
"""
Sample code that does in-place basic arithmetic of 
addition, multiplication, subtraction, and division.
"""

import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
upone_path = os.path.dirname(dir_path)
sys.path.append(upone_path)

from device import Device


with Device() as d:

    # Allocating two complex arrays, with default fill value
    d.a = d.malloc((768,512), dtype='f8', fill=3)
    d.b = d.malloc(d.a.shape, dtype='f8', fill=1)
    
    # Some simple in-place arithmetic
    d.b += d.a
    d.b *= d.a
    d.a -= d.b
    d.a /= d.b
    
    pow(d.a, 2)
    
    #Query memory usage for fun
    d.query()
    
    result = d.a.to_host()