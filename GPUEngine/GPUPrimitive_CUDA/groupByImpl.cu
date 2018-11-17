
#ifndef GROUP_BY_CU
#define GROUP_BY_CU

#ifdef COALESCED
	__global__ void
	scanGroupLabel_kernel( Record* d_Rin, int rLen, int* d_groupLabel )
	{
		int bx = blockIdx.x;
		int tx = threadIdx.x;
		int gridSize = blockDim.x*gridDim.x;
		int currentValue;
		int nextValue;

		for( int idx = bx*blockDim.x + tx; idx < rLen - 1; idx += gridSize )
		{
			currentValue = d_Rin[idx].y;
			nextValue = d_Rin[idx + 1].y;

			if( currentValue != nextValue )
			{
				d_groupLabel[idx + 1] = 1;
			}
		}

		//write the first position
		if( (bx == 0) && (tx == 0) )
		{
			d_groupLabel[0] = 1;
		}
	}

	__global__ void
	groupByImpl_write_kernel( int* d_startPos, int* d_groupLabel, int* d_writePos, int rLen )
	{
		int bx = blockIdx.x;
		int tx = threadIdx.x;
		int gridSize = blockDim.x*gridDim.x;

		for( int idx = bx*blockDim.x + tx; idx < rLen; idx += gridSize )
		{
			if( d_groupLabel[idx] == 1 )
			{
				d_startPos[d_writePos[idx]] = idx;
			}
		}	
	}
#else// no coalesced

	__global__ void
	scanGroupLabel_kernel( Record* d_Rin, int rLen, int* d_groupLabel )
	{
		int bx = blockIdx.x;
		int tx = threadIdx.x;
		int gridSize = blockDim.x*gridDim.x;
		int currentValue;
		int nextValue;

		int numThread = blockDim.x;
		int numBlock = gridDim.x;
		int totalTx = bx*numThread + tx;
		int threadSize = rLen/( numThread*numBlock );
		int start = totalTx*threadSize;
		int end = start + threadSize;

		if( (tx == (numThread - 1)) && (bx == (numBlock - 1))  )
		{
			end--;
		}

		//for( int idx = bx*blockDim.x + tx; idx < rLen - 1; idx += gridSize )
		for( int idx = start; idx < end; idx++ )
		{
			currentValue = d_Rin[idx].y;
			nextValue = d_Rin[idx + 1].y;

			if( currentValue != nextValue )
			{
				d_groupLabel[idx + 1] = 1;
			}
		}

		//write the first position
		if( (bx == 0) && (tx == 0) )
		{
			d_groupLabel[0] = 1;
		}
	}

	__global__ void
	groupByImpl_write_kernel( int* d_startPos, int* d_groupLabel, int* d_writePos, int rLen )
	{
		int bx = blockIdx.x;
		int tx = threadIdx.x;	
		int numThread = blockDim.x;
		int numBlock = gridDim.x;
		int totalTx = bx*numThread + tx;
		int threadSize = rLen/( numThread*numBlock );
		int start = totalTx*threadSize;
		int end = start + threadSize;

		for( int idx = start; idx < end; idx++ )
		{
			if( d_groupLabel[idx] == 1 )
			{
				d_startPos[d_writePos[idx]] = idx;
			}
		}	
	}
	
#endif

__global__ void
groupByImpl_outSize_kernel( int* d_outSize, int* d_mark, int* d_markOutput, int rLen )
{
	*d_outSize = d_mark[rLen-1] + d_markOutput[rLen-1];
}

int groupByImpl(Record* d_Rin, int rLen, Record * d_Rout, int** d_startPos, int numThread = 512, int numBlock = 256)
{
#ifdef COALESCED
	printf( "YES, COALESCED, GROUP BY\n" );
#else
	printf( "NO, COALESCED, GROUP BY\n" );
#endif

	int numGroup = 0;
	unsigned int timer = 0;

	//sort
	startTimer( &timer );
	bitonicSortGPU( d_Rin, rLen, d_Rout, numThread, numBlock );
	//GPUOnly_QuickSort( d_Rin, rLen, d_Rout );
	endTimer( "bitonic sort", &timer );

	//first scan to get the group position labels
	//startTimer( &timer );
	int* d_groupLabel;
	GPUMALLOC( (void**)&d_groupLabel, sizeof(int)*rLen ) ;
	CUDA_SAFE_CALL( cudaMemset( d_groupLabel, 0, sizeof(int)*rLen ) );
	scanGroupLabel_kernel<<<numBlock, numThread>>>( d_Rout, rLen, d_groupLabel );
	cudaThreadSynchronize();
	//endTimer( "first scan", &timer );


	//prefex sum
	//startTimer(&timer);
	int* d_writePos;
	GPUMALLOC( (void**)&d_writePos, sizeof(int)*rLen ) ;
	scanImpl( d_groupLabel, rLen, d_writePos );
	cudaThreadSynchronize();
	//endTimer( "prefex sum", &timer );

	//get the number of groups
	//startTimer( &timer );
	int* d_numGroup;
	GPUMALLOC( (void**)&d_numGroup, sizeof(int) ) ;
	groupByImpl_outSize_kernel<<<1, 1>>>( d_numGroup, d_groupLabel, d_writePos, rLen );
	cudaThreadSynchronize();
	CUDA_SAFE_CALL( cudaMemcpy( &numGroup, d_numGroup, sizeof(int), 
		cudaMemcpyDeviceToHost) );
	//endTimer( "get the number of groups", &timer );

	//second scan to get the group positions
	//startTimer( &timer );
	GPUMALLOC( (void**)&(*d_startPos), sizeof(int)*numGroup ) ;
	groupByImpl_write_kernel<<<numBlock, numThread>>>( (*d_startPos), d_groupLabel, d_writePos, rLen );
	cudaThreadSynchronize();
	//endTimer( "second scan", &timer );

	//for debug
	/*Record* h_Rout = (Record*)malloc( sizeof(Record)*rLen ) ;
	CUDA_SAFE_CALL( cudaMemcpy( h_Rout, d_Rout, sizeof(Record)*rLen, cudaMemcpyDeviceToHost ) );
	int* h_groupLabel = (int*)malloc( sizeof(int)*rLen );
	CUDA_SAFE_CALL( cudaMemcpy( h_groupLabel, d_groupLabel, sizeof(int)*rLen, cudaMemcpyDeviceToHost ) );
	int* h_startPos = (int*)malloc( sizeof(int)*numGroup );
	CUDA_SAFE_CALL( cudaMemcpy( h_startPos, *d_startPos, sizeof(int)*numGroup, cudaMemcpyDeviceToHost ) );*/

	CUDA_SAFE_CALL( cudaFree( d_groupLabel ) );
	CUDA_SAFE_CALL( cudaFree( d_writePos ) );
	CUDA_SAFE_CALL( cudaFree( d_numGroup ) );

	return numGroup;
}

#endif
