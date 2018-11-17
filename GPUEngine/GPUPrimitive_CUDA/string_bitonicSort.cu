#ifndef _STRING_BITONIC_SORT_
#define _STRING_BITONIC_SORT_

#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "string_bitonicSort_kernel.cu"
#include "math.h"
#include "StringCmp.cu"
#include "string_bitonicProc.cu"
#include "assert.h"
/*
@ d_rawData, the array for the keys.
@ d_Rin, the input pointer array.
@ totalLenInBytes, is not used.
@ rLen, the number of tuples.
@ d_Rout, the output pointer array.
*/
void string_bitonicSortGPU(void* d_rawData, int totalLenInBytes, cmp_type_t* d_Rin, int rLen, void *d_Rout)
{
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
		
		unsigned int numThreadsSort = STRING_NUM_THREADS_CHUNK;
		if(numRecordsR<STRING_NUM_THREADS_CHUNK)
			numThreadsSort=numRecordsR;
		unsigned int numBlocksXSort = numRecordsR/numThreadsSort;
		unsigned int numBlocksYSort = 1;
		dim3 gridSort( numBlocksXSort, numBlocksYSort );		
		unsigned int memSizeRecordsR = sizeof( cmp_type_t ) * numRecordsR;
		//copy the <offset, length> pairs.
		cmp_type_t* d_R;
		GPUMALLOC( (void**) &d_R, memSizeRecordsR) ;
		cmp_type_t tempValue;
		tempValue.x=tempValue.y=-1;
		initialize(d_R, numRecordsR, tempValue);
		CUDA_SAFE_CALL( cudaMemcpy( d_R, d_Rin, rLen*sizeof(cmp_type_t), cudaMemcpyDeviceToDevice) );
		

		for( int k = 2; k <= numRecordsR; k *= 2 )
		{
			for( int j = k/2; j > 0; j /= 2 )
			{
				string_bitonicKernel<<<gridSort, numThreadsSort>>>((void*)d_rawData, totalLenInBytes, d_R, numRecordsR, k, j);
			}
		}
		CUDA_SAFE_CALL( cudaMemcpy( d_Rout, d_R+(numRecordsR-rLen), sizeof(cmp_type_t)*rLen, cudaMemcpyDeviceToDevice) );
		cudaFree( d_R );
		cudaThreadSynchronize();
	}
	else
	{
		unsigned int numThreadsSort = STRING_NUM_THREADS_CHUNK;
		unsigned int numBlocksYSort = 1;
		unsigned int numBlocksXSort = (numRecordsR/numThreadsSort)/numBlocksYSort;
		if(numBlocksXSort>=(1<<16))
		{
			numBlocksXSort=(1<<15);
			numBlocksYSort=(numRecordsR/numThreadsSort)/numBlocksXSort;			
		}
		unsigned int numBlocksChunk = NUM_BLOCKS_CHUNK;
		unsigned int numThreadsChunk = STRING_NUM_THREADS_CHUNK;
		
		unsigned int chunkSize = numBlocksChunk*numThreadsChunk;
		unsigned int numChunksR = numRecordsR/chunkSize;

		dim3 gridSort( numBlocksXSort, numBlocksYSort );
		unsigned int memSizeRecordsR = sizeof( cmp_type_t ) * numRecordsR;

		cmp_type_t* d_R;
		GPUMALLOC( (void**) &d_R, memSizeRecordsR) ;
		cmp_type_t tempValue;
		tempValue.x=tempValue.y=-1;
		initialize(d_R, numRecordsR, tempValue);
		CUDA_SAFE_CALL( cudaMemcpy( d_R, d_Rin, rLen*sizeof(cmp_type_t), cudaMemcpyDeviceToDevice) );

		for( int chunkIdx = 0; chunkIdx < numChunksR; chunkIdx++ )
		{
			string_unitBitonicSortKernel<<< numBlocksChunk, numThreadsChunk>>>( (void*)d_rawData, totalLenInBytes, d_R, numRecordsR, chunkIdx );
		}

		int j;
		for( int k = numThreadsChunk*2; k <= numRecordsR; k *= 2 )
		{
			for( j = k/2; j > numThreadsChunk/2; j /= 2 )
			{
				string_bitonicKernel<<<gridSort, numThreadsSort>>>( (void*)d_rawData, totalLenInBytes, d_R, numRecordsR, k, j);
			}

			for( int chunkIdx = 0; chunkIdx < numChunksR; chunkIdx++ )
			{
				string_partBitonicSortKernel<<< numBlocksChunk, numThreadsChunk>>>((void*)d_rawData, totalLenInBytes, d_R, numRecordsR, chunkIdx, k/numThreadsSort );
			}
		}
		CUDA_SAFE_CALL( cudaMemcpy( d_Rout, d_R+(numRecordsR-rLen), sizeof(cmp_type_t)*rLen, cudaMemcpyDeviceToDevice) );
		cudaFree( d_R );
		cudaThreadSynchronize();
	}
}

/*
totalLenInBytes is in terms of bytes.
*/
void string_bitonicSort( void* rawData, int totalLenInBytes, cmp_type_t *Rin, int rLen, void** Rout)
{
	cmp_type_t *d_Rout;
	GPUMALLOC( (void**) (&d_Rout), sizeof(cmp_type_t)*rLen) ;
	char* d_rawData;
	GPUMALLOC( (void**) &d_rawData, totalLenInBytes) ;
	CUDA_SAFE_CALL( cudaMemcpy( d_rawData, rawData, totalLenInBytes, cudaMemcpyHostToDevice) );
	cmp_type_t* d_Rin;
	GPUMALLOC( (void**) &d_Rin, sizeof(cmp_type_t)*rLen) ;
	CUDA_SAFE_CALL( cudaMemcpy( d_Rin, Rin, sizeof(cmp_type_t)*rLen, cudaMemcpyHostToDevice) );

	string_bitonicSortGPU(d_rawData, totalLenInBytes, d_Rin, rLen, d_Rout);
	*Rout=(cmp_type_t*)malloc(sizeof(cmp_type_t)*rLen);
	CUDA_SAFE_CALL( cudaMemcpy( *Rout, d_Rout, sizeof(cmp_type_t)*rLen, cudaMemcpyDeviceToHost) );
}


void string_bitonicSort_AllocatedData( void* d_rawData, int totalLenInBytes, cmp_type_t *Rin, int rLen, cmp_type_t* Rout)
{
	cmp_type_t *d_Rout;
	GPUMALLOC( (void**) (&d_Rout), sizeof(cmp_type_t)*rLen) ;
	cmp_type_t* d_Rin;
	GPUMALLOC( (void**) &d_Rin, sizeof(cmp_type_t)*rLen);
	CUDA_SAFE_CALL( cudaMemcpy( d_Rin, Rin, sizeof(cmp_type_t)*rLen, cudaMemcpyHostToDevice) );

	string_bitonicSortGPU(d_rawData, totalLenInBytes, d_Rin, rLen, d_Rout);
	CUDA_SAFE_CALL( cudaMemcpy( Rout, d_Rout, sizeof(cmp_type_t)*rLen, cudaMemcpyDeviceToHost) );
}


/*void bitonicSort_AllocatedData( void* d_rawData, int totalLenInBytes, Record *d_Rin, int rLen, Record* d_Rout)
{
	bitonicSortGPU(d_rawData, totalLenInBytes, d_Rin, rLen, d_Rout);
}*/

#endif
