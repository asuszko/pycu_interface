# pycu_interface

This repo is for a Python framework that provides a user with CUDA **GPU resource management, performance primitives, and custom CUDA kernel calls to accelerate Python code**.

# cuBLAS and cuFFT Support

Functions in the cuBLAS and cuFFT objects are supported as they are needed by the user. That is, not all functions are available out of the box. When a user needs a new function, it may be added to the source in the same manner as the others are. This involves writing up a C++11 compatible wrapper (if templates are useful), that is wrapped by an 'extern C' exported function. This exported function is called in Python using the [ctypes](https://docs.python.org/3/library/ctypes.html) function library.

If adding functionality to pycu_interface, fork the repo, add the new code, compile, test, and then merge request if you wish to help update master. Note that, [cuda_helpers](https://github.com/asuszko/cuda_helpers), [cublas_helpers](https://github.com/asuszko/cublas_helpers), and [cufft_helpers](https://github.com/asuszko/cufft_helpers) have their own repos, and are included as sub_modules in pycu_interface. For the current available [cuBLAS](https://github.com/asuszko/cublas_helpers) and [cuFFT](https://github.com/asuszko/cufft_helpers) routines, see their readme in their respective repos.

For reference to official Nvidia documentation:
- [cuBLAS Documentation](http://docs.nvidia.com/cuda/cublas/index.html)
- [cuFFT Documentation](http://docs.nvidia.com/cuda/cufft/index.html)

## Setup

To clone the repo and all subrepos with it, in your terminal or git console, run the following command:
> git clone --recursive git@github.com:asuszko/pycu_interface.git

To compile the shared libraries needed, run the **setup.py** file found in the root folder from the command line, with optional argument(s) -arch, and -cc_bin if on Windows. On Windows, the NVCC compiler looks for cl.exe to compile the C/C++ code. cl.exe comes with Visual Studio. On Linux, it uses the built in gcc compiler. An example of a command line run (on Windows) to compile the code is given below:
> python setup.py -arch=sm_50 -cc_bin="C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin"

On Linux, the command would be the same, with the -cc_bin argument omitted. If you are unable to compile the libraries, you may [download the latest precompiled libraries here](https://github.com/asuszko/pycu_interface_libs).

## Compiler Requirements

- Python 3.6.x (2.7 compatibility not yet tested - so YMMV with 2.7 for now) 
- The latest  version of the [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- (Windows) Visual Studio 2013 Professional, or any compatible version with your version of CUDA Toolkit. Note: You can download a trial of Professional to obtain cl.exe. Compilation via the command line will still work after the trial period has ended.

## Testing

After obtaining the shared libraries either by compilation or by [download](https://github.com/asuszko/pycu_interface_libs), run any of the sample scripts located from within the samples folder, for example:
> python simple_cufft.py

## Troubleshooting

Known error messages and causes:
> GPUassert: invalid device symbol

You are using a shared library that was compiled using a compute architecture that your hardware does not support. Recompile or download libraries that use an older architecture.

## Notes

For sample scripts or further documentation on how to use this framework to implement your own custom CUDA kernels, view the code in the repos below that import and utilize pycu_interface. The codes below are simple examples that show how to utilize the device object for **optimal** GPU resource management, and custom CUDA kernel calls to accelerate the Python. More details and tutorials on how to use this framework optimally are in the works. 

- [Image Phase Correlation](https://github.com/asuszko/phase_correlation)
- [Signal Cross Correlation](https://github.com/asuszko/signal_cross_correlation)

## License
 
The MIT License (MIT)

Copyright (c) 2018 Arthur Suszko (art.suszko@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
