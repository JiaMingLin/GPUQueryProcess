#ifndef GPU_MEMORY_CU
#define GPU_MEMORY_CU

#include "GPU_Dll.h"

extern "C"
void CopyCPUToGPU( void* d_destData, void* h_srcData, int sizeInBytes )
{
	CUDA_SAFE_CALL( cudaMemcpy( d_destData, h_srcData, sizeInBytes, cudaMemcpyHostToDevice ) );
}

extern "C"
void CopyGPUToCPU( void * h_destData, void* d_srcData, int sizeInBytes)
{
	CUDA_SAFE_CALL( cudaMemcpy( h_destData, d_srcData, sizeInBytes, cudaMemcpyDeviceToHost ) );
}

extern "C"
void GPUAllocate( void** d_data, int sizeInBytes )
{
	GPUMALLOC( d_data, sizeInBytes );
}

extern "C"
void CPUAllocateByCUDA( void** h_data, int sizeInBytes )
{
	CPUMALLOC( h_data, sizeInBytes );
}


extern "C"
void GPUFree( void* d_data)
{
	CUDA_SAFE_CALL( cudaFree( d_data) );
}

extern "C"
void CPUFreeByCUDA( void* h_data)
{
	CUDA_SAFE_CALL( cudaFreeHost( h_data) );
}

extern "C"
void resetGPU()
{
	CUDA_SAFE_CALL( cudaThreadExit() );
}






#endif