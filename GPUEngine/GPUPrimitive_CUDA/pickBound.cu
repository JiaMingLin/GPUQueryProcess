#ifndef PICK_CHUNK_CU
#define PICK_CHUNK_CU

#include "pickBound_kernel.cu"

int computeLargeBound(Record *d_iBound, int rLen, int threshold, int* d_flag, int* d_flagSum, Record* d_oBound)
{
	unsigned int timer=0;
	startTimer(&timer);
	int numThreadsPerBlock_x=256;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int numThread=numBlock_x*numThreadsPerBlock_x;
	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	countLargeBound_kernel<<<grid,thread>>>(d_iBound, numThread, rLen, threshold, d_flag);
	endTimer("countLargeBound_kernel",&timer);
	startTimer(&timer);
	scanImpl(d_flag, rLen, d_flagSum);
	int lastValue=0;
	int partialSum=0;
	FROMGPU(&lastValue, (d_flag+(rLen-1)), sizeof(int));
	FROMGPU(&partialSum, (d_flagSum+(rLen-1)), sizeof(int));
	int numLargeBound=lastValue+partialSum;
	printf("larger than %d, #numLargeBound, %d\n", threshold, numLargeBound);
	//scatter the large bound to the front:)
	writeLargeBound_kernel<<<grid,thread>>>(d_iBound, numThread, rLen, d_flag, d_flagSum, d_oBound);
	//reverse the small bound
	reverseFlag_kernel<<<grid,thread>>>(d_flag, numThread, rLen);
	//prefix sum on the small bound
	scanImpl(d_flag, rLen, d_flagSum);
	writeSmallBound_kernel<<<grid,thread>>>(d_iBound, numThread, rLen, numLargeBound, d_flag, d_flagSum, d_oBound);
	endTimer("computeLargeBound",&timer);
	return numLargeBound;
}

#endif

