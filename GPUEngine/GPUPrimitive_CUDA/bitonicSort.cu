#ifndef _BITONIC_SORT_
#define _BITONIC_SORT_

#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "bitonicSort_kernel.cu"
#include "math.h"
#include "assert.h"
#include "mapImpl.cu"

#include "GPU_Dll.h"

/*
@ d_Rin, the input pointer array.
@ rLen, the number of tuples.
@ d_Rout, the output pointer array.
*/
void bitonicSortGPU(Record* d_Rin, int rLen, Record *d_Rout, 
					int numThreadPB=NUM_BLOCKS_CHUNK, int numBlock=NUM_BLOCKS_CHUNK)
{
#ifdef OUTPUT_INFO
	#ifdef SHARED_MEM
		printf("YES, SHARED_MEM in bitonic sort\n");
	#else
		printf("NO, SHARED_MEM in bitonic sort\n");
	#endif
#endif
	unsigned int numRecordsR;

	unsigned int size = rLen;
	unsigned int level = 0;
	while( size != 1 )
	{
		size = size/2;
		level++;
	}

	if( (1<<level) < rLen )
	{
		level++;
	}

	numRecordsR = (1<<level);
	
	if( rLen <= 256*1024 )
	{
		//unsigned int numRecordsR = rLen;
		
		unsigned int numThreadsSort = numThreadPB;
		if(numRecordsR<numThreadPB)
			numThreadsSort=numRecordsR;
		unsigned int numBlocksXSort = numRecordsR/numThreadsSort;
		unsigned int numBlocksYSort = 1;
		dim3 gridSort( numBlocksXSort, numBlocksYSort );		
		unsigned int memSizeRecordsR = sizeof( Record ) * numRecordsR;
		//copy the <offset, length> pairs.
		Record* d_R;
		GPUMALLOC( (void**) &d_R, memSizeRecordsR) ;
		Record tempValue;
		tempValue.x=tempValue.y=TEST_MAX;
		mapInit(d_R, rLen, numRecordsR, tempValue);//[rLen, numRecordsR)
		CUDA_SAFE_CALL( cudaMemcpy( d_R, d_Rin, rLen*sizeof(Record), cudaMemcpyDeviceToDevice) );
		

		for( int k = 2; k <= numRecordsR; k *= 2 )
		{
			for( int j = k/2; j > 0; j /= 2 )
			{
				bitonicKernel<<<gridSort, numThreadsSort>>>(d_R, numRecordsR, k, j);
			}
		}
		//CUDA_SAFE_CALL( cudaMemcpy( d_Rout, d_R+(numRecordsR-rLen), sizeof(Record)*rLen, cudaMemcpyDeviceToDevice) );
		CUDA_SAFE_CALL( cudaMemcpy( d_Rout, d_R, sizeof(Record)*rLen, cudaMemcpyDeviceToDevice) );
		cudaFree( d_R );
		cudaThreadSynchronize();
	}
	else
	{
		unsigned int numThreadsSort = numThreadPB;
		unsigned int numBlocksYSort = 1;
		unsigned int numBlocksXSort = (numRecordsR/numThreadsSort)/numBlocksYSort;
		if(numBlocksXSort>=(1<<16))
		{
			numBlocksXSort=(1<<15);
			numBlocksYSort=(numRecordsR/numThreadsSort)/numBlocksXSort;			
		}
		unsigned int numBlocksChunk = numBlock;
		unsigned int numThreadsChunk = numThreadPB;
		
		unsigned int chunkSize = numBlocksChunk*numThreadsChunk;
		unsigned int numChunksR = numRecordsR/chunkSize;

		dim3 gridSort( numBlocksXSort, numBlocksYSort );
		unsigned int memSizeRecordsR = sizeof( Record ) * numRecordsR;

		Record* d_R;
		GPUMALLOC( (void**) &d_R, memSizeRecordsR) ;
		Record tempValue;
		tempValue.x=tempValue.y=TEST_MAX;
		mapInit(d_R, rLen, numRecordsR, tempValue);
		unsigned int timer=0;
		//startTimer(&timer);
		CUDA_SAFE_CALL( cudaMemcpy( d_R, d_Rin, rLen*sizeof(Record), cudaMemcpyDeviceToDevice) );
		//endTimer("copy GPUtoGPU", &timer);
		int sharedMemSize=numThreadPB*sizeof(Record);
		//startTimer(&timer);
		for( int chunkIdx = 0; chunkIdx < numChunksR; chunkIdx++ )
		{
			unitBitonicSortKernel<<< numBlocksChunk, numThreadsChunk, sharedMemSize>>>(  d_R, numRecordsR, 
				chunkIdx,numThreadsChunk, chunkSize, numBlocksChunk);
		}
		//endTimer("unit", &timer);
		int j;
		for( int k = numThreadsChunk*2; k <= numRecordsR; k *= 2 )
		{
			for( j = k/2; j > numThreadsChunk/2; j /= 2 )
			{
				bitonicKernel<<<gridSort, numThreadsSort>>>( d_R, numRecordsR, k, j);
			}

			for( int chunkIdx = 0; chunkIdx < numChunksR; chunkIdx++ )
			{
				partBitonicSortKernel<<< numBlocksChunk, numThreadsChunk, sharedMemSize>>>(d_R, numRecordsR, 
					chunkIdx, k/numThreadsSort,numThreadsChunk, chunkSize, numBlocksChunk );
			}
		}
		//startTimer(&timer);
		//CUDA_SAFE_CALL( cudaMemcpy( d_Rout, d_R+(numRecordsR-rLen), sizeof(Record)*rLen, cudaMemcpyDeviceToDevice) );
		CUDA_SAFE_CALL( cudaMemcpy( d_Rout, d_R, sizeof(Record)*rLen, cudaMemcpyDeviceToDevice) );
		//endTimer("copy GPUtoGPU result", &timer);
		cudaFree( d_R );
		cudaThreadSynchronize();
	}
}

//////////////////////////////////////////////////////////////////////
//the export interface

extern "C"
void GPUOnly_bitonicSort( Record* d_Rin, int rLen, Record* d_Rout, 
						 int numThreadPB, int numBlock)
{
	bitonicSortGPU( d_Rin, rLen, d_Rout, numThreadPB, numBlock );
}


extern "C"
void GPUCopy_bitonicSort( Record* h_Rin, int rLen, Record* h_Rout, 
						 int numThreadPB, int numBlock )
{	
	int memSize = sizeof(Record)*rLen;
	
	Record* d_Rin;
	Record* d_Rout;
	GPUMALLOC( (void**)&d_Rin, memSize );
	GPUMALLOC( (void**)&d_Rout, memSize );
	
	TOGPU( d_Rin, h_Rin, memSize );

	bitonicSortGPU( d_Rin, rLen, d_Rout, numThreadPB, numBlock);

	FROMGPU( h_Rout, d_Rout, memSize );

	GPUFREE( d_Rin );
	GPUFREE( d_Rout );
}

#endif
