#ifndef SORT_IMPL_CU
#define SORT_IMPL_CU

#include "bitonicSort.cu"
#include "radixSortImpl.cu"
#include "getMax.cu"

#include "GPU_Dll.h"

void sortImpl(Record *d_R, int rLen, Record *d_S, 
			  int numThreadPB=NUM_BLOCKS_CHUNK, int numBlock=NUM_BLOCKS_CHUNK)
{
	bitonicSortGPU(d_R, rLen, d_S,numThreadPB,numBlock);
	//GPUFREE(d_R);
}

void radixSortImpl(Record *d_R, int rLen, Record* d_S)
{
	int maxValue=getMax(d_R,rLen);
	int startBits=log2Ceil(maxValue);
	int totalBitsUsed=log2Ceil(rLen)-8;
	totalBitsUsed=(totalBitsUsed>startBits)?startBits:totalBitsUsed;
	int bitPerPass=0;
	if(totalBitsUsed>15)//larger than 15
		bitPerPass=6;
	else
		if(totalBitsUsed>12)//(12,15]
			bitPerPass=5;
		else
			if(totalBitsUsed>=12)//[6,12]
				bitPerPass=6;
			else 
				bitPerPass=totalBitsUsed;
	printf("totalBitsUsed, %d, bitPerPass, %d, startBits, %d\n", totalBitsUsed, bitPerPass, startBits);
	unsigned int timer=0;
	startTimer(&timer);
    radixSort(d_R,rLen,startBits,totalBitsUsed,bitPerPass,  d_S);
	double processingTime=endTimer("radix",&timer);
}

extern "C"
void GPUOnly_RadixSort( Record* d_Rin, int rLen, Record* d_Rout )
{
	radixSortImpl(d_Rin, rLen, d_Rout);
}

extern "C"
void GPUCopy_RadixSort( Record* h_Rin, int rLen, Record* h_Rout )
{
	int memSize = sizeof(Record)*rLen;

	Record* d_Rin;
	Record* d_Rout;
	GPUMALLOC( (void**)&d_Rin, memSize);
	GPUMALLOC( (void**)&d_Rout, memSize );

	TOGPU( d_Rin, h_Rin, memSize );

	GPUOnly_RadixSort(d_Rin, rLen, d_Rout);

	FROMGPU( h_Rout, d_Rout, memSize );

	GPUFREE( d_Rin );
	GPUFREE( d_Rout );
}


#endif

