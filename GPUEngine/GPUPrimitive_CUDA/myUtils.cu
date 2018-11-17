#pragma once

#include <cutil.h>
#include <GPUPrimitive_Def.cu>

void mirrorHostDataOnDevice(void* h_src, void** d_dest, unsigned int size) {
    //CUDA_SAFE_CALL(cudaMalloc((void**) d_dest, size));
    //CUDA_SAFE_CALL(cudaMemcpy(*d_dest, h_src, size, cudaMemcpyHostToDevice) );
	GPUMALLOC((void**) d_dest, size);
	TOGPU(*d_dest, h_src, size);
}

void copyBackToHost(void* d_src, void** h_dest, unsigned int size, int allocateOnHost, int destroyDeviceCopy) {
	if(allocateOnHost) {
		//*h_dest = malloc(size);
		CPUMALLOC((void**)&(*h_dest),size);
	}

    //CUDA_SAFE_CALL(cudaMemcpy(*h_dest, d_src, size, cudaMemcpyDeviceToHost));
	FROMGPU(*h_dest, d_src, size);

	if(destroyDeviceCopy) {
		//CUDA_SAFE_CALL(cudaFree(d_src));
		GPUFREE(d_src);
	}
}

void startMyUtilTimer(unsigned int* t)
{
	CUT_SAFE_CALL( cutCreateTimer(t) );
    CUT_SAFE_CALL( cutStartTimer(*t) );
}

float endMyUtilTimer(unsigned int t)
{
	CUT_SAFE_CALL(cutStopTimer(t));
	float result = cutGetTimerValue(t);
    CUT_SAFE_CALL(cutDeleteTimer(t));

	return result;
}
