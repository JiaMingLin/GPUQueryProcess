/*
Source code and executable packages for GPUQP
http://www.cse.ust.hk/gpuqp/
July, 2008.
*/

***GENEERAL INFORMATION***
There are two directories in the package, GPUPrimitive_CUDA and Test. The GPUPrimitive_CUDA contains all the source codes and Test contains the test scripts for our experiments. The project can be either compiled into a DLL or executable. The header file for the exported DLL is GPU_Dll.h. Note that this GPU code package should be put into the CUDA source code directory when compiling.

***ENVIRONMENT***
- Hardware: CUDA-Enabled GPU. The list of the supported GPU products is available here: http://www.nvidia.com/object/cuda_learn_products.html
- Operating System: Windows XP or compatible OS
- Software: CUDA 1.0 or above
- Programming IDE: Microsoft Visual Studio C++ 2005 or compatible compilers.

***PACKAGE DESCRIPTION***
- Header File:
GPU_Dll.h: There are C/C++ encapsulated interfaces for exported dll. There are two prefix for most functions, namely GPIOnly_ and GPUCopy_. GPUOnly_ assumes the data is already in the GPU memory, while GPUCopy_ assumes the data is in the main memory. Thus there are extra memory transfers in the GPUCopy_ functions to copy the data from main memory to GPU memory.

- Primitives:
The package contains the primitives map, scatter, gather, reduce prefix scan, split fitler and sort.

- Join Algorithm:
There are four joins algorithms are implemented: Non-Indexed Nested Loop Joins, Indexed Nested Loop Joins, Sort-Merge Joins and Hash Joins.

***TEST CODE***
The TestAll.cu and TestJoin.cu include the test codes for all primitives and joins.