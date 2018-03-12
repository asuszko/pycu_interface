# pycu_interface

This repo is for a Python framework that allows a user to **GPU accelerate their Python code**. Within the container are controls to create, access, and modify GPU device memory, and callable cuBLAS and cuFFT objects. 


# cuBLAS and cuFFT Support

Functions in the cuBLAS and cuFFT objects are supported as they are needed by the user. That is, not all functions are available out of the box. When a user needs a new function, it may be added to the source in the same manner others are. When doing this, start a new branch, add the function, compile, test, and then merge request after verifying the added function works. For current available [cuBLAS](https://github.com/asuszko/cublas_helpers) and [cuFFT](https://github.com/asuszko/cufft_helpers) routines, see the readme in their respective repos linked.

For reference to official Nvidia documentation:
- [cuBLAS Documentation](http://docs.nvidia.com/cuda/cublas/index.html)
- [cuFFT Documentation](http://docs.nvidia.com/cuda/cufft/index.html)

## Setup

To compile the shared libraries needed, run the **setup.py** file found in the root folder from the command line, with optional argument(s) -arch, and -cc_bin if on Windows. On Windows, the NVCC compiler looks for cl.exe to compile the C/C++, which comes with Visual Studio. On Linux, it uses the built in GCC compiler. An example of a command line run to compile the code is given below:
> python setup.py -arch=sm_50 -cc_bin="C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin"

If you are unable to compile, you may [download precompiled libraries here](https://github.com/asuszko/pycu_interface_libs).

## Compiler Requirements

- Python 3.6.x (2.7 compatibility not yet tested) 
- The latest  version of the [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- (Windows) Visual Studio 2013 Professional, or any compatible version with your version of CUDA Toolkit. Note: You can download a trial of Professional to obtain cl.exe. Compilation via the command line will still work after the trial period has ended.

## Testing

After obtaining the shared libraries either by compilation or by  download, run any of the sample scripts located in the samples folder, for example:
> python simple_cufft.py

## Notes
For sample scripts or further documentation on how to use this framework to implement your own custom CUDA kernels, view the code in the repos below that import and utilize pycu_interface, and/or view the PowerPoint presentation [here](link) which provides some basic examples on how utilize this framework.

- [Image Phase Correlation](https://github.com/asuszko/phase_correlation)
- [Signal Cross Correlation](https://github.com/asuszko/signal_cross_correlation)

## License
 
The MIT License (MIT)

Copyright (c) 2018 Arthur Suszko (art.suszko@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.