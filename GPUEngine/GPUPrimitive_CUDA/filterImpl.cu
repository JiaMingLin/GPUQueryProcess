#ifndef FILTER_IMPL_CU
#define FILTER_IMPL_CU


__global__ void
filterImpl_map_kernel( Record* d_Rin, int beginPos, int rLen, int* d_mark, int smallKey, int largeKey, int* d_temp )
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int gridSize = blockDim.x*gridDim.x;
	int value;

	for( int idx = bx*blockDim.x + tx + beginPos; idx < beginPos + rLen; idx += gridSize )
	{		
		d_temp[idx] = d_Rin[idx].x;
		value = d_Rin[idx].y;
		//the filter condition
		if( ( value >= smallKey ) && ( value <= largeKey ) )
		{
			d_mark[idx] = 1;
		}
	}
}

__global__ void 
filterImpl_map_noCoalesced_kernel( Record* d_Rin, int beginPos, int rLen, int* d_mark, int smallKey, int largeKey, int* d_temp )
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;	
	int numThread = blockDim.x;
	int numBlock = gridDim.x;
	int totalTx = bx*numThread + tx;
	int threadSize = rLen/( numThread*numBlock );
	int start = totalTx*threadSize;
	int end = start + threadSize;
	int value;

	__syncthreads();
	for( int idx = start; idx < end; idx++ )
	{
		value = d_Rin[idx].y;
		//the filter condition
		if( ( value >= smallKey ) && ( value <= largeKey ) )
		{
			d_mark[idx] = 1;
		}
	}
}


__global__ void
filterImpl_outSize_kernel( int* d_outSize, int* d_mark, int* d_markOutput, int rLen )
{
	*d_outSize = d_mark[rLen-1] + d_markOutput[rLen-1];
}


__global__ void
filterImpl_write_noCoalesced_kernel( Record* d_Rout, Record* d_Rin, int* d_mark, int* d_markOutput, int beginPos, int rLen )
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
		if( d_mark[idx] == 1 )
		{
			d_Rout[d_markOutput[idx]].x = d_Rin[idx+beginPos].x;
			d_Rout[d_markOutput[idx]].y = d_Rin[idx+beginPos].y;
		}
	}	
}

__global__ void
filterImpl_write_kernel( Record* d_Rout, Record* d_Rin, int* d_mark, int* d_markOutput, int beginPos, int rLen )
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int gridSize = blockDim.x*gridDim.x;

	for( int idx = bx*blockDim.x + tx; idx < rLen; idx += gridSize )
	{
		if( d_mark[idx] == 1 )
		{
			d_Rout[d_markOutput[idx]].x = d_Rin[idx+beginPos].x;
			d_Rout[d_markOutput[idx]].y = d_Rin[idx+beginPos].y;
		}
	}	
}


void filterImpl( Record* d_Rin, int beginPos, int rLen, Record** d_Rout, int* outSize, 
				int numThread, int numBlock, int smallKey, int largeKey)
{
#ifdef COALESCED 
	printf( "YES, COALESCED, FILTER\n" );
#else
	printf( "NO COALESCED, FILTER\n" );
#endif

	int* d_mark;
	GPUMALLOC( (void**)&d_mark, sizeof(int)*rLen ) ;
	CUDA_SAFE_CALL( cudaMemset( d_mark, 0, sizeof(int)*rLen ) );
	int* d_markOutput;
	GPUMALLOC( (void**)&d_markOutput, sizeof(int)*rLen ) ;

	unsigned int timer = 0;

	//map to 0 or 1
	//startTimer( &timer );
	int* d_temp;
	GPUMALLOC( (void**)&d_temp, sizeof(int)*rLen ) ;
#ifdef COALESCED
	filterImpl_map_kernel<<<numBlock, numThread>>>( d_Rin, beginPos, rLen, d_mark, smallKey, largeKey, d_temp );
#else
	filterImpl_map_noCoalesced_kernel<<<numBlock, numThread>>>( d_Rin, beginPos, rLen, d_mark, smallKey, largeKey, d_temp );
#endif
	cudaThreadSynchronize();
	GPUFREE( d_temp );
	//endTimer( "map", &timer );

	//prefex sum
	scanImpl( d_mark, rLen, d_markOutput );

	//get the outSize
	int* d_outSize;
	GPUMALLOC( (void**)&d_outSize, sizeof(int) ) ;
	filterImpl_outSize_kernel<<<1, 1>>>( d_outSize, d_mark, d_markOutput, rLen );
	cudaThreadSynchronize();
	CUDA_SAFE_CALL( cudaMemcpy( outSize, d_outSize, sizeof(int), 
		cudaMemcpyDeviceToHost) );

	//write the reduced result
	GPUMALLOC( (void**)&(*d_Rout), sizeof(Record)*(*outSize) ) ;
#ifdef COALESCED
	filterImpl_write_kernel<<<numBlock, numThread>>>( *d_Rout, d_Rin, d_mark, d_markOutput, beginPos, rLen );
#else
	filterImpl_write_noCoalesced_kernel<<<numBlock, numThread>>>( *d_Rout, d_Rin, d_mark, d_markOutput, beginPos, rLen );
#endif
	cudaThreadSynchronize();

	CUDA_SAFE_CALL( cudaFree(d_mark) );
	CUDA_SAFE_CALL( cudaFree(d_markOutput) );
}

#endif
