#ifndef BITONIC_PROC_H
#define BITONIC_PROC_H
#include "bitonicProc_kernel.cu"
#define NUM_BLOCK_PER_CHUNK_BITONIC_SORT 8192//b256
#define MAX_LARGE_CHUNK_SIZE (8*8*1024)
#define LARGE_CHUNK_PER_TIME 128

/*
@totalLenInBytes, is not used. 
*/
void bitonicSortMultipleBlocks( Record * d_values, Record* d_bound, int numBlock, Record * d_output)
{
	int numThreadsPerBlock_x=SHARED_MEM_INT2;
	int numThreadsPerBlock_y=1;
	int numBlock_x=NUM_BLOCK_PER_CHUNK_BITONIC_SORT;
	int numBlock_y=1;
	int numChunk=numBlock/numBlock_x;
	if(numBlock%numBlock_x!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	printf("bitonicSortMultipleBlocks_kernel: total, %d, numBlock_x, %d, numChunk, %d",numBlock, numBlock_x,numChunk);
	for(i=0;i<numChunk;i++)
	{
		start=i*numBlock_x;
		end=start+numBlock_x;
		if(end>numBlock)
			end=numBlock;
		//printf("bitonicSortMultipleBlocks_kernel: %d, range, %d, %d\n", i, start, end);
		bitonicSortMultipleBlocks_kernel<<<grid,thread>>>(d_values, d_bound, start, end-start, d_output);
		cudaThreadSynchronize();
	}
//	cudaThreadSynchronize();
}



void bitonicSortMultipleLargeBlocks( Record * d_values, Record* d_iBound, int numLargeBound, Record * d_output)
{
	int numThreadsPerBlock_x=SHARED_MEM_INT2;
	int numThreadsPerBlock_y=1;
	int numBlock_x=MAX_LARGE_CHUNK_SIZE/SHARED_MEM_INT2;
	int numBlock_y=LARGE_CHUNK_PER_TIME;
	int totalBufSize=numBlock_y*MAX_LARGE_CHUNK_SIZE;
	Record *d_Buf;
	GPUMALLOC((void**)&d_Buf, sizeof(Record)*totalBufSize);
	
	int numChunk=numLargeBound/numBlock_y;
	if(numLargeBound%numBlock_y!=0)
		numChunk++;
	
	Record* h_iBound;
	CPUMALLOC((void**)&h_iBound, sizeof(Record)*numLargeBound);
	FROMGPU(h_iBound, d_iBound, sizeof(Record)*numLargeBound);
	//gpuPrintInt2(d_iBound, numLargeBound, "d_iBound");

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0,j=0;
	int start=0;
	int end=0;
	int realMAXSize=0;
	int trueMaxSize = 0;
	int curSize=0;
	//int curNumChunk=0;
	printf("total, %d, LARGE_CHUNK_PER_TIME, %d,\n",numChunk, LARGE_CHUNK_PER_TIME);
	for(i=0;i<numChunk;i++)
	{
		CUDA_SAFE_CALL( cudaMemset( d_Buf, TEST_MAX, totalBufSize*sizeof(int) ) );

		realMAXSize=0;
		start=i*numBlock_y;
		end=start+numBlock_y;
		if(end>numLargeBound)
		{
			end=numLargeBound;
		}

		int* flag = (int*)malloc( sizeof(int)*(end - start) );
		for( int k = 0; k < (end - start); k++ )
		{
			flag[k] = -1;
		}
			
		//curNumChunk=end-start;
		//copy data to the chunk
		Record tempValue;
		tempValue.x=tempValue.y=TEST_MAX;
		mapInit(d_Buf, 0, totalBufSize, tempValue);

		int smallSizeIdx = 0;
		for( j = start; j < end; j++ )
		{
			//printf("C: %d, %d]   ", h_iBound[j].x, h_iBound[j].y);
			curSize=h_iBound[j].y-h_iBound[j].x;
			if( (curSize > realMAXSize) && (curSize < MAX_LARGE_CHUNK_SIZE) )
			{
				realMAXSize = curSize;
			}				

			if( curSize <= MAX_LARGE_CHUNK_SIZE )
			{
				flag[j - start] = smallSizeIdx;

				GPUTOGPU(d_Buf + flag[j - start]*MAX_LARGE_CHUNK_SIZE, d_values+h_iBound[j].x, (curSize)*sizeof(Record));
				smallSizeIdx++;
			}
			
		}
		//printf("\n");
		realMAXSize = (1<<log2Ceil(realMAXSize));
		assert(realMAXSize<=MAX_LARGE_CHUNK_SIZE);

		//sorting.
		unsigned int timer = 0;
		startTimer( &timer );
		numBlock_x = realMAXSize/SHARED_MEM_INT2;// end - start;
		grid.x = numBlock_x;
		printf("bitonicSortMultipleLargeBlocks: total, %d, numBlock_x, %d, numBlock_y, %d, curChunk, %d, realMAXSize, %d\n",numLargeBound, numBlock_x,numBlock_y, i, realMAXSize);
		for( int k = 2; k <= realMAXSize; k *= 2 )
		{
			for( int j = k/2; j > 0; j /= 2 )
			{
				bitonicMultipleLargeBlocks_kernel<<<grid, thread>>>(d_Buf, MAX_LARGE_CHUNK_SIZE, numBlock_y , k, j);
			}
		}
		endTimer( "bitonicMultipleLargeBlocks", &timer );
		
		//copy back
		unsigned int chunkSize = 0;
		for(j = start; j < end; j++)
		{
			if( flag[j - start] >= 0 )
			{
				GPUTOGPU(d_output+h_iBound[j].x,d_Buf+flag[j - start]*MAX_LARGE_CHUNK_SIZE,(h_iBound[j].y-h_iBound[j].x)*sizeof(Record));
			}
		}

		//for the larger block
		unsigned int largeSize = 0;
		for( j = start; j < end; j++ )
		{
			curSize=h_iBound[j].y-h_iBound[j].x;

			if( curSize > MAX_LARGE_CHUNK_SIZE )
			{
				GPUOnly_bitonicSort( d_values+h_iBound[j].x, curSize, d_Buf);
				cudaThreadSynchronize();
				GPUTOGPU( d_values+h_iBound[j].x, d_Buf, sizeof(Record)*curSize );

				largeSize += curSize;
			}			
		}

	}

	cudaThreadSynchronize();
	GPUFREE(d_Buf);
	CPUFREE(h_iBound);
}

#endif
