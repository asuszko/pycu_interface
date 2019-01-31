# -*- coding: utf-8 -*-

import argparse
import os

from shared_utils.build import build

__compile_dirs = {"cuda_helpers"   : "cuda",
                  "cublas_helpers" : "cublas",
                  "cufft_helpers"  : "cufft"}



def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-arch', '--arch',
                        action="store", dest="arch",
                        help="CUDA hardware architecture version",
                        default="sm_30")
    
    parser.add_argument('-cc_bin', '--cc_bin',
                        action="store", dest="cc_bin",
                        help="Path to the cl.exe bin folder on Windows",
                        default=None)
    
    args = parser.parse_args()
    
    base_path = os.path.abspath(os.path.dirname(__file__))
    
    for _dir_name, _so_name in __compile_dirs.items():
        module_path = os.path.join(base_path, _dir_name)
        build(module_path, _so_name, args.arch, args.cc_bin)
    
    
if __name__ == "__main__":
    main()