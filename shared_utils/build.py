# -*- coding: utf-8 -*-

__all__ = [
    "build"
]

import glob
import os
import platform
import subprocess


def nvcc_call(wd, name, sources,
              arch="sm_30",
              compile_args=["-m64", "-Xcompiler", "-std=c++l4"],
              cc_bin=None,
              include_dirs=[],
              library_dirs=[],
              libraries=[],
              extra_compile_args=[]):
    """
    Compile sources to shared library using nvcc.
    """
    
    if platform.system() == "Linux":
        ext = ".so"
        cc_bin = None
    elif platform.system() == "Windows":
        ext = ".dll"
    
    cwd = os.getcwd()
    os.chdir(wd)

    as_list = lambda str_or_list: [str_or_list] if isinstance(str_or_list, str) else str_or_list

    sources = as_list(sources)
    compile_args = as_list(compile_args)
    include_dirs = as_list(include_dirs)
    library_dirs = as_list(library_dirs)
    libraries = as_list(libraries)
    extra_compile_args = as_list(extra_compile_args)

    call_list = ["nvcc"]
    call_list += ["-arch="+arch]
    call_list += compile_args
    call_list += ["-shared", "-o", name+ext]
    call_list += sources
    call_list += ["-I"+I for I in include_dirs]
    call_list += ["-L"+L for L in library_dirs]
    if cc_bin is not None:
        call_list += [as_list('-ccbin "'+os.path.normpath(cc_bin)+'"')]
    call_list += ["-l"+l for l in libraries]
    call_list += extra_compile_args

    print(subprocess.list2cmdline(call_list))
    subprocess.call(call_list)

    os.chdir(cwd)
    


def build(module_path, so_name, arch="sm_30", cc_bin=None):
    
    if os.path.exists(module_path):
        
        src_path = os.path.join(module_path,"src")
        lib_path = os.path.join(module_path,"lib")
        
        # Create the folder if not exists to put shared library in
        if not os.path.exists(lib_path):
            os.makedirs(lib_path)
        
        # Compile into shared library
        os.chdir(src_path)
        nvcc_call(wd=src_path,
                  name=os.path.join(lib_path,so_name),
                  sources=glob.glob("*.cu"),
                  arch=arch,
                  cc_bin=cc_bin,
                  libraries=["cuda", "cublas", "cufft"])
        
        #Cleanup extra compile files
        filelist = glob.glob(os.path.join(lib_path, "*.exp"))
        for f in filelist:
            os.remove(f)
        filelist = glob.glob(os.path.join(lib_path, "*.lib"))
        for f in filelist:
            os.remove(f)