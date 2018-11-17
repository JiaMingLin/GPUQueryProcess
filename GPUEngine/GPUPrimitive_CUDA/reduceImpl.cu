#ifndef	REDUCE_IMPL_CU
#define REDUCE_IMPL_CU

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

// Define this to more rigorously avoid bank conflicts, 
// even at the lower (root) levels of the tree
// Note that due to the higher addressing overhead, performance 
// is lower with ZERO_BANK_CONFLICTS enabled.  It is provided
// as an example.
//#define ZERO_BANK_CONFLICTS 

#include "GPU_Dll.h"

//#define DEBUG 1

// 16 banks on G80
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

/*#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif*/

//#define REDUCE_MAX_NUM_BLOCK (65535)

int** d_scanBlockSums;
int** h_scanBlockSums;
int* levelSize;
unsigned int d_numLevelsAllocated = 0;

#ifdef COALESCED

	inline __device__
	int CONFLICT_FREE_OFFSET_REDUCE( int index )
	{
		//return ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS));

		return ((index) >> LOG_NUM_BANKS);
	}
  
	//with shared memory, with coalesced
	__global__
	void perscanFirstPass_kernel( int* d_temp, int* d_odata, Record* d_idata, int numElementsPerBlock, 
								 bool isFull, int base, int d_odataOffset, int OPERATOR )
	{
		extern __shared__ int temp[];
		
		int thid = threadIdx.x;
		int offset = 1;
		int baseIdx = blockIdx.x*(blockDim.x*2) + base; 

		int mem_ai = baseIdx + thid;
		int mem_bi = mem_ai + blockDim.x;

		int ai = thid;
		int bi = thid + blockDim.x;

		int bankOffsetA = CONFLICT_FREE_OFFSET_REDUCE( ai );
		int bankOffsetB = CONFLICT_FREE_OFFSET_REDUCE( bi );

		d_temp[mem_ai] = d_idata[mem_ai].x;
		temp[ai + bankOffsetA] = d_idata[mem_ai].y;
		
		if( OPERATOR == REDUCE_SUM || OPERATOR == REDUCE_AVERAGE )
		{
			if( isFull )
			{
				d_temp[mem_bi] = d_idata[mem_bi].x;
				temp[bi + bankOffsetB] = d_idata[mem_bi].y;
			}
			else
			{
				temp[bi + bankOffsetB] = (bi < numElementsPerBlock) ? (d_idata[mem_bi].y) : (0);
			}
		}
		else if( OPERATOR == REDUCE_MAX )
		{
			if( isFull )
			{
				d_temp[mem_bi] = d_idata[mem_bi].x;
				temp[bi + bankOffsetB] = d_idata[mem_bi].y;
			}
			else
			{
				temp[bi + bankOffsetB] = (bi < numElementsPerBlock) ? (d_idata[mem_bi].y) : ( TEST_MIN );
			}
		}
		else if( OPERATOR == REDUCE_MIN )
		{
			if( isFull )
			{
				d_temp[mem_bi] = d_idata[mem_bi].x;
				temp[bi + bankOffsetB] = d_idata[mem_bi].y;
			}
			else
			{
				temp[bi + bankOffsetB] = (bi < numElementsPerBlock) ? (d_idata[mem_bi].y) : ( TEST_MAX );
			}
		}


		__syncthreads();

		//build sum in place up the tree
		if( OPERATOR == REDUCE_SUM || OPERATOR == REDUCE_AVERAGE )
		{
			for( int d = (blockDim.x*2)>>1; d > 0; d >>= 1 )
			{
				__syncthreads();

				if( thid < d )
				{
					int ai = offset*( 2*thid + 1 ) - 1;
					int bi = offset*( 2*thid + 2 ) - 1;
					ai += CONFLICT_FREE_OFFSET_REDUCE( ai );
					bi += CONFLICT_FREE_OFFSET_REDUCE( bi );

					temp[bi] += temp[ai];
				}

				offset *= 2;
			}
		}
		else if( OPERATOR == REDUCE_MAX )
		{
			for( int d = (blockDim.x*2)>>1; d > 0; d >>= 1 )
			{
				__syncthreads();

				if( thid < d )
				{
					int ai = offset*( 2*thid + 1 ) - 1;
					int bi = offset*( 2*thid + 2 ) - 1;
					ai += CONFLICT_FREE_OFFSET_REDUCE( ai );
					bi += CONFLICT_FREE_OFFSET_REDUCE( bi );

					temp[bi] = (temp[bi] > temp[ai])?(temp[bi]):(temp[ai]);				
				}

				offset *= 2;
			}
		}
		else if( OPERATOR == REDUCE_MIN )
		{
			for( int d = (blockDim.x*2)>>1; d > 0; d >>= 1 )
			{
				__syncthreads();

				if( thid < d )
				{
					int ai = offset*( 2*thid + 1 ) - 1;
					int bi = offset*( 2*thid + 2 ) - 1;
					ai += CONFLICT_FREE_OFFSET_REDUCE( ai );
					bi += CONFLICT_FREE_OFFSET_REDUCE( bi );

					temp[bi] = (temp[bi] > temp[ai])?(temp[ai]):(temp[bi]);				
				}

				offset *= 2;
			}
		}


		__syncthreads();

		//write out the reduced block sums to d_odata
		if( thid == (blockDim.x - 1)  )
		{
			d_odata[blockIdx.x + d_odataOffset] = temp[bi+bankOffsetB];
		}
	}


	//with shared memory, with coalesced
	__global__
	void perscan_kernel( int* d_odata, int* d_idata, int numElementsPerBlock, bool isFull, int base, int d_odataOffset, int OPERATOR )
	{
		extern __shared__ int temp[];
		
		int thid = threadIdx.x;
		int offset = 1;
		int baseIdx = blockIdx.x*(blockDim.x*2) + base; 

		int mem_ai = baseIdx + thid;
		int mem_bi = mem_ai + blockDim.x;

		int ai = thid;
		int bi = thid + blockDim.x;

		int bankOffsetA = CONFLICT_FREE_OFFSET_REDUCE( ai );
		int bankOffsetB = CONFLICT_FREE_OFFSET_REDUCE( bi );

		temp[ai + bankOffsetA] = d_idata[mem_ai];

		if( OPERATOR == REDUCE_SUM || OPERATOR == REDUCE_AVERAGE )
		{
			if( isFull )
			{
				temp[bi + bankOffsetB] = d_idata[mem_bi];
			}
			else
			{
				temp[bi + bankOffsetB] = (bi < numElementsPerBlock) ? (d_idata[mem_bi]) : (0);
			}
		}
		else if( OPERATOR == REDUCE_MAX )
		{
			if( isFull )
			{
				temp[bi + bankOffsetB] = d_idata[mem_bi];
			}
			else
			{
				temp[bi + bankOffsetB] = (bi < numElementsPerBlock) ? (d_idata[mem_bi]) : (TEST_MIN);
			}
		}
		else if( OPERATOR == REDUCE_MIN )
		{
			if( isFull )
			{
				temp[bi + bankOffsetB] = d_idata[mem_bi];
			}
			else
			{
				temp[bi + bankOffsetB] = (bi < numElementsPerBlock) ? (d_idata[mem_bi]) : (TEST_MAX);
			}
		}


		__syncthreads();

		//build sum in place up the tree
		if( OPERATOR == REDUCE_SUM || OPERATOR == REDUCE_AVERAGE )
		{
			for( int d = (blockDim.x*2)>>1; d > 0; d >>= 1 )
			{
				__syncthreads();

				if( thid < d )
				{
					int ai = offset*( 2*thid + 1 ) - 1;
					int bi = offset*( 2*thid + 2 ) - 1;
					ai += CONFLICT_FREE_OFFSET_REDUCE( ai );
					bi += CONFLICT_FREE_OFFSET_REDUCE( bi );

					temp[bi] += temp[ai];
				}

				offset *= 2;
			}
		}
		else if( OPERATOR == REDUCE_MAX )
		{
			for( int d = (blockDim.x*2)>>1; d > 0; d >>= 1 )
			{
				__syncthreads();

				if( thid < d )
				{
					int ai = offset*( 2*thid + 1 ) - 1;
					int bi = offset*( 2*thid + 2 ) - 1;
					ai += CONFLICT_FREE_OFFSET_REDUCE( ai );
					bi += CONFLICT_FREE_OFFSET_REDUCE( bi );

					temp[bi] = (temp[bi] > temp[ai])?(temp[bi]):(temp[ai]);
				}

				offset *= 2;
			}
		}
		else if( OPERATOR == REDUCE_MIN )
		{
			for( int d = (blockDim.x*2)>>1; d > 0; d >>= 1 )
			{
				__syncthreads();

				if( thid < d )
				{
					int ai = offset*( 2*thid + 1 ) - 1;
					int bi = offset*( 2*thid + 2 ) - 1;
					ai += CONFLICT_FREE_OFFSET_REDUCE( ai );
					bi += CONFLICT_FREE_OFFSET_REDUCE( bi );

					temp[bi] = (temp[bi] > temp[ai])?(temp[ai]):(temp[bi]);
				}

				offset *= 2;
			}
		}

		__syncthreads();

		//write out the reduced block sums to d_odata
		if( thid == (blockDim.x - 1)  )
		{
			d_odata[blockIdx.x + d_odataOffset] = temp[bi+bankOffsetB];
		}	
	}
#else
	inline __device__
	int CONFLICT_FREE_OFFSET_REDUCE( int index )
	{
		return 0;
	}
  
	//with shared memory, without coalesced
	__global__
	void perscanFirstPass_kernel( int* d_temp, int* d_odata, Record* d_idata, int numElementsPerBlock, 
								 bool isFull, int base, int d_odataOffset, int OPERATOR )
	{
		extern __shared__ int temp[];
		
		int thid = threadIdx.x;
		int offset = 1;
		int baseIdx = blockIdx.x*(blockDim.x*2) + base; 

		int mem_ai = baseIdx + 2*thid;
		int mem_bi = mem_ai + 1;

		int ai = 2*thid;
		int bi = ai + 1;

		int bankOffsetA = CONFLICT_FREE_OFFSET_REDUCE( ai );
		int bankOffsetB = CONFLICT_FREE_OFFSET_REDUCE( bi );

		temp[ai + bankOffsetA] = d_idata[mem_ai].y;
		
		if( OPERATOR == REDUCE_SUM || OPERATOR == REDUCE_AVERAGE )
		{
			if( isFull )
			{
				d_temp[mem_bi] = d_idata[mem_bi].x;
				temp[bi + bankOffsetB] = d_idata[mem_bi].y;
			}
			else
			{
				temp[bi + bankOffsetB] = (bi < numElementsPerBlock) ? (d_idata[mem_bi].y) : (0);
			}
		}
		else if( OPERATOR == REDUCE_MAX )
		{
			if( isFull )
			{
				d_temp[mem_bi] = d_idata[mem_bi].x;
				temp[bi + bankOffsetB] = d_idata[mem_bi].y;
			}
			else
			{
				temp[bi + bankOffsetB] = (bi < numElementsPerBlock) ? (d_idata[mem_bi].y) : ( TEST_MIN );
			}
		}
		else if( OPERATOR == REDUCE_MIN )
		{
			if( isFull )
			{
				d_temp[mem_bi] = d_idata[mem_bi].x;
				temp[bi + bankOffsetB] = d_idata[mem_bi].y;
			}
			else
			{
				temp[bi + bankOffsetB] = (bi < numElementsPerBlock) ? (d_idata[mem_bi].y) : ( TEST_MAX );
			}
		}


		__syncthreads();

		//build sum in place up the tree
		if( OPERATOR == REDUCE_SUM || OPERATOR == REDUCE_AVERAGE )
		{
			for( int d = (blockDim.x*2)>>1; d > 0; d >>= 1 )
			{
				__syncthreads();

				if( thid < d )
				{
					int ai = offset*( 2*thid + 1 ) - 1;
					int bi = offset*( 2*thid + 2 ) - 1;
					ai += CONFLICT_FREE_OFFSET_REDUCE( ai );
					bi += CONFLICT_FREE_OFFSET_REDUCE( bi );

					temp[bi] += temp[ai];
				}

				offset *= 2;
			}
		}
		else if( OPERATOR == REDUCE_MAX )
		{
			for( int d = (blockDim.x*2)>>1; d > 0; d >>= 1 )
			{
				__syncthreads();

				if( thid < d )
				{
					int ai = offset*( 2*thid + 1 ) - 1;
					int bi = offset*( 2*thid + 2 ) - 1;
					ai += CONFLICT_FREE_OFFSET_REDUCE( ai );
					bi += CONFLICT_FREE_OFFSET_REDUCE( bi );

					temp[bi] = (temp[bi] > temp[ai])?(temp[bi]):(temp[ai]);				
				}

				offset *= 2;
			}
		}
		else if( OPERATOR == REDUCE_MIN )
		{
			for( int d = (blockDim.x*2)>>1; d > 0; d >>= 1 )
			{
				__syncthreads();

				if( thid < d )
				{
					int ai = offset*( 2*thid + 1 ) - 1;
					int bi = offset*( 2*thid + 2 ) - 1;
					ai += CONFLICT_FREE_OFFSET_REDUCE( ai );
					bi += CONFLICT_FREE_OFFSET_REDUCE( bi );

					temp[bi] = (temp[bi] > temp[ai])?(temp[ai]):(temp[bi]);				
				}

				offset *= 2;
			}
		}


		__syncthreads();

		//write out the reduced block sums to d_odata
		if( thid == (blockDim.x - 1)  )
		{
			d_odata[blockIdx.x + d_odataOffset] = temp[bi+bankOffsetB];
		}
	}


	//with shared memory, without coalesced
	__global__
	void perscan_kernel( int* d_odata, int* d_idata, int numElementsPerBlock, bool isFull, int base, int d_odataOffset, int OPERATOR )
	{
		extern __shared__ int temp[];
		
		int thid = threadIdx.x;
		int offset = 1;
		int baseIdx = blockIdx.x*(blockDim.x*2) + base; 

		int mem_ai = baseIdx + 2*thid;
		int mem_bi = mem_ai + 1;

		int ai = 2*thid;
		int bi = ai + 1;

		int bankOffsetA = CONFLICT_FREE_OFFSET_REDUCE( ai );
		int bankOffsetB = CONFLICT_FREE_OFFSET_REDUCE( bi );

		temp[ai + bankOffsetA] = d_idata[mem_ai];

		if( OPERATOR == REDUCE_SUM || OPERATOR == REDUCE_AVERAGE )
		{
			if( isFull )
			{
				temp[bi + bankOffsetB] = d_idata[mem_bi];
			}
			else
			{
				temp[bi + bankOffsetB] = (bi < numElementsPerBlock) ? (d_idata[mem_bi]) : (0);
			}
		}
		else if( OPERATOR == REDUCE_MAX )
		{
			if( isFull )
			{
				temp[bi + bankOffsetB] = d_idata[mem_bi];
			}
			else
			{
				temp[bi + bankOffsetB] = (bi < numElementsPerBlock) ? (d_idata[mem_bi]) : (TEST_MIN);
			}
		}
		else if( OPERATOR == REDUCE_MIN )
		{
			if( isFull )
			{
				temp[bi + bankOffsetB] = d_idata[mem_bi];
			}
			else
			{
				temp[bi + bankOffsetB] = (bi < numElementsPerBlock) ? (d_idata[mem_bi]) : (TEST_MAX);
			}
		}


		__syncthreads();

		//build sum in place up the tree
		if( OPERATOR == REDUCE_SUM || OPERATOR == REDUCE_AVERAGE )
		{
			for( int d = (blockDim.x*2)>>1; d > 0; d >>= 1 )
			{
				__syncthreads();

				if( thid < d )
				{
					int ai = offset*( 2*thid + 1 ) - 1;
					int bi = offset*( 2*thid + 2 ) - 1;
					ai += CONFLICT_FREE_OFFSET_REDUCE( ai );
					bi += CONFLICT_FREE_OFFSET_REDUCE( bi );

					temp[bi] += temp[ai];
				}

				offset *= 2;
			}
		}
		else if( OPERATOR == REDUCE_MAX )
		{
			for( int d = (blockDim.x*2)>>1; d > 0; d >>= 1 )
			{
				__syncthreads();

				if( thid < d )
				{
					int ai = offset*( 2*thid + 1 ) - 1;
					int bi = offset*( 2*thid + 2 ) - 1;
					ai += CONFLICT_FREE_OFFSET_REDUCE( ai );
					bi += CONFLICT_FREE_OFFSET_REDUCE( bi );

					temp[bi] = (temp[bi] > temp[ai])?(temp[bi]):(temp[ai]);
				}

				offset *= 2;
			}
		}
		else if( OPERATOR == REDUCE_MIN )
		{
			for( int d = (blockDim.x*2)>>1; d > 0; d >>= 1 )
			{
				__syncthreads();

				if( thid < d )
				{
					int ai = offset*( 2*thid + 1 ) - 1;
					int bi = offset*( 2*thid + 2 ) - 1;
					ai += CONFLICT_FREE_OFFSET_REDUCE( ai );
					bi += CONFLICT_FREE_OFFSET_REDUCE( bi );

					temp[bi] = (temp[bi] > temp[ai])?(temp[ai]):(temp[bi]);
				}

				offset *= 2;
			}
		}

		__syncthreads();

		//write out the reduced block sums to d_odata
		if( thid == (blockDim.x - 1)  )
		{
			d_odata[blockIdx.x + d_odataOffset] = temp[bi+bankOffsetB];
		}	
	}
#endif


//without shared memory, with coalesced
__global__
void perscan_kernel( int* d_data, int* d_odata, int* d_idata, 
					int numElementsPerBlock, bool isFull, int base, int d_odataOffset, int OPERATOR, unsigned int sharedMemSize )
{
	//extern __shared__ int temp[];
	int bx = blockIdx.x;
	int* temp = d_data + (sharedMemSize/sizeof(int))*bx;
	
	int thid = threadIdx.x;
	int offset = 1;
	int baseIdx = blockIdx.x*(blockDim.x*2) + base; 

	int mem_ai = baseIdx + thid;
	int mem_bi = mem_ai + blockDim.x;

	int ai = thid;
	int bi = thid + blockDim.x;

	int bankOffsetA = CONFLICT_FREE_OFFSET_REDUCE( ai );
	int bankOffsetB = CONFLICT_FREE_OFFSET_REDUCE( bi );

	temp[ai + bankOffsetA] = d_idata[mem_ai];

	if( OPERATOR == REDUCE_SUM || OPERATOR == REDUCE_AVERAGE )
	{
		if( isFull )
		{
			temp[bi + bankOffsetB] = d_idata[mem_bi];
		}
		else
		{
			temp[bi + bankOffsetB] = (bi < numElementsPerBlock) ? (d_idata[mem_bi]) : (0);
		}
	}
	else if( OPERATOR == REDUCE_MAX )
	{
		if( isFull )
		{
			temp[bi + bankOffsetB] = d_idata[mem_bi];
		}
		else
		{
			temp[bi + bankOffsetB] = (bi < numElementsPerBlock) ? (d_idata[mem_bi]) : (TEST_MIN);
		}
	}
	else if( OPERATOR == REDUCE_MIN )
	{
		if( isFull )
		{
			temp[bi + bankOffsetB] = d_idata[mem_bi];
		}
		else
		{
			temp[bi + bankOffsetB] = (bi < numElementsPerBlock) ? (d_idata[mem_bi]) : (TEST_MAX);
		}
	}


	__syncthreads();

	//build sum in place up the tree
	if( OPERATOR == REDUCE_SUM || OPERATOR == REDUCE_AVERAGE )
	{
		for( int d = (blockDim.x*2)>>1; d > 0; d >>= 1 )
		{
			__syncthreads();

			if( thid < d )
			{
				int ai = offset*( 2*thid + 1 ) - 1;
				int bi = offset*( 2*thid + 2 ) - 1;
				ai += CONFLICT_FREE_OFFSET_REDUCE( ai );
				bi += CONFLICT_FREE_OFFSET_REDUCE( bi );

				temp[bi] += temp[ai];
			}

			offset *= 2;
		}
	}
	else if( OPERATOR == REDUCE_MAX )
	{
		for( int d = (blockDim.x*2)>>1; d > 0; d >>= 1 )
		{
			__syncthreads();

			if( thid < d )
			{
				int ai = offset*( 2*thid + 1 ) - 1;
				int bi = offset*( 2*thid + 2 ) - 1;
				ai += CONFLICT_FREE_OFFSET_REDUCE( ai );
				bi += CONFLICT_FREE_OFFSET_REDUCE( bi );

				temp[bi] = (temp[bi] > temp[ai])?(temp[bi]):(temp[ai]);
			}

			offset *= 2;
		}
	}
	else if( OPERATOR == REDUCE_MIN )
	{
		for( int d = (blockDim.x*2)>>1; d > 0; d >>= 1 )
		{
			__syncthreads();

			if( thid < d )
			{
				int ai = offset*( 2*thid + 1 ) - 1;
				int bi = offset*( 2*thid + 2 ) - 1;
				ai += CONFLICT_FREE_OFFSET_REDUCE( ai );
				bi += CONFLICT_FREE_OFFSET_REDUCE( bi );

				temp[bi] = (temp[bi] > temp[ai])?(temp[ai]):(temp[bi]);
			}

			offset *= 2;
		}
	}

	__syncthreads();

	//write out the reduced block sums to d_odata
	if( thid == (blockDim.x - 1)  )
	{
		d_odata[blockIdx.x + d_odataOffset] = temp[bi+bankOffsetB];
	}	
}

__global__
void copyLastElement( int* d_odata, Record* d_Rin, int base, int offset)
{
	d_odata[offset] = d_Rin[base].y;
}



//without shared memory, with coalesced
__global__
void perscanFirstPass_kernel( int* d_data, int* d_temp, int* d_odata, Record* d_idata, int numElementsPerBlock, 
							 bool isFull, int base, int d_odataOffset, int OPERATOR, unsigned int sharedMemSize )
{
	//extern __shared__ int temp[];
	int bx = blockIdx.x;
	int* temp = d_data + (sharedMemSize/sizeof(int))*bx;
	
	int thid = threadIdx.x;
	int offset = 1;
	int baseIdx = blockIdx.x*(blockDim.x*2) + base; 

	int mem_ai = baseIdx + thid;
	int mem_bi = mem_ai + blockDim.x;

	int ai = thid;
	int bi = thid + blockDim.x;

	int bankOffsetA = CONFLICT_FREE_OFFSET_REDUCE( ai );
	int bankOffsetB = CONFLICT_FREE_OFFSET_REDUCE( bi );

	d_temp[mem_ai] = d_idata[mem_ai].x;
	temp[ai + bankOffsetA] = d_idata[mem_ai].y;
	
	if( OPERATOR == REDUCE_SUM || OPERATOR == REDUCE_AVERAGE )
	{
		if( isFull )
		{
			d_temp[mem_bi] = d_idata[mem_bi].x;
			temp[bi + bankOffsetB] = d_idata[mem_bi].y;
		}
		else
		{
			temp[bi + bankOffsetB] = (bi < numElementsPerBlock) ? (d_idata[mem_bi].y) : (0);
		}
	}
	else if( OPERATOR == REDUCE_MAX )
	{
		if( isFull )
		{
			d_temp[mem_bi] = d_idata[mem_bi].x;
			temp[bi + bankOffsetB] = d_idata[mem_bi].y;
		}
		else
		{
			temp[bi + bankOffsetB] = (bi < numElementsPerBlock) ? (d_idata[mem_bi].y) : ( TEST_MIN );
		}
	}
	else if( OPERATOR == REDUCE_MIN )
	{
		if( isFull )
		{
			d_temp[mem_bi] = d_idata[mem_bi].x;
			temp[bi + bankOffsetB] = d_idata[mem_bi].y;
		}
		else
		{
			temp[bi + bankOffsetB] = (bi < numElementsPerBlock) ? (d_idata[mem_bi].y) : ( TEST_MAX );
		}
	}


	__syncthreads();

	//build sum in place up the tree
	if( OPERATOR == REDUCE_SUM || OPERATOR == REDUCE_AVERAGE )
	{
		for( int d = (blockDim.x*2)>>1; d > 0; d >>= 1 )
		{
			__syncthreads();

			if( thid < d )
			{
				int ai = offset*( 2*thid + 1 ) - 1;
				int bi = offset*( 2*thid + 2 ) - 1;
				ai += CONFLICT_FREE_OFFSET_REDUCE( ai );
				bi += CONFLICT_FREE_OFFSET_REDUCE( bi );

				temp[bi] += temp[ai];
			}

			offset *= 2;
		}
	}
	else if( OPERATOR == REDUCE_MAX )
	{
		for( int d = (blockDim.x*2)>>1; d > 0; d >>= 1 )
		{
			__syncthreads();

			if( thid < d )
			{
				int ai = offset*( 2*thid + 1 ) - 1;
				int bi = offset*( 2*thid + 2 ) - 1;
				ai += CONFLICT_FREE_OFFSET_REDUCE( ai );
				bi += CONFLICT_FREE_OFFSET_REDUCE( bi );

				temp[bi] = (temp[bi] > temp[ai])?(temp[bi]):(temp[ai]);				
			}

			offset *= 2;
		}
	}
	else if( OPERATOR == REDUCE_MIN )
	{
		for( int d = (blockDim.x*2)>>1; d > 0; d >>= 1 )
		{
			__syncthreads();

			if( thid < d )
			{
				int ai = offset*( 2*thid + 1 ) - 1;
				int bi = offset*( 2*thid + 2 ) - 1;
				ai += CONFLICT_FREE_OFFSET_REDUCE( ai );
				bi += CONFLICT_FREE_OFFSET_REDUCE( bi );

				temp[bi] = (temp[bi] > temp[ai])?(temp[ai]):(temp[bi]);				
			}

			offset *= 2;
		}
	}


	__syncthreads();

	//write out the reduced block sums to d_odata
	if( thid == (blockDim.x - 1)  )
	{
		d_odata[blockIdx.x + d_odataOffset] = temp[bi+bankOffsetB];
	}
}

void preallocBlockSums( int rLen, int numThread )
{
	unsigned int blockSize = numThread; // max size of the thread blocks
    unsigned int numElts = rLen;

    int level = 0;

    do
    {       
        unsigned int numBlocks = 
            max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1)
        {
            level++;
        }
        numElts = numBlocks;
    } while (numElts > 1);

	//at least one level, the last level has the sum
	d_numLevelsAllocated = level + 1;
	levelSize = (int*)malloc( sizeof(int)*d_numLevelsAllocated ); 

	//allocate the d_scanSumsBlock
	d_scanBlockSums = (int**)malloc( sizeof(int*)*d_numLevelsAllocated );
/*#ifdef DEBUG	
	h_scanBlockSums = (int**)malloc( sizeof(int*)*d_numLevelsAllocated );
#endif*/
	numElts = rLen;
    level = 0;
    
    do
    {       
        unsigned int numBlocks = 
            max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) 
        {
			levelSize[level] = numBlocks;
            GPUMALLOC((void**) &d_scanBlockSums[level],  
                                      numBlocks * sizeof(int));

/*#ifdef DEBUG
			CPUMALLOC((void**) &h_scanBlockSums[level],  
                                    numBlocks * sizeof(int));
#endif*/
			level++;
        }
        numElts = numBlocks;
    } while (numElts > 1);

	//the last level
 	levelSize[d_numLevelsAllocated - 1] = 1;
	GPUMALLOC( (void**)&d_scanBlockSums[d_numLevelsAllocated - 1], sizeof(int) ) ;
}

//return bool: if is multiple of maxNumThread
//if yes, info[0]: number of blocks, info[1] = maxNumThread
//if no, info[0]: number of blocks except of the last block, info[1]: number of thread in the last block
bool howPartition( int rLen, int maxNumThread, int* info )
{
	int maxNumElePerBlock = maxNumThread*2;
	int numBlock = (int)ceil((float)rLen/(maxNumElePerBlock));
	bool isMul = (rLen%maxNumElePerBlock == 0)?true:false;

	if( isMul )
	{
		info[0] = numBlock;
		info[1] = maxNumThread;
	}
	else
	{
		info[0] = numBlock - 1;

		int remainer = rLen - (numBlock - 1)*maxNumElePerBlock;
		if( isPowerOfTwo(remainer) )
		{
			info[1] = remainer/2;
		}
		else
		{
			info[1] = floorPow2( remainer );
		}
	}

	return isMul;
}

void reduceFirstPass( Record* d_Rin, int rLen, int numThread, int numMaxBlock, int OPERATOR )
{
	int* info = (int*)malloc( sizeof(int)*2 );
	//get the information of partition	
	//return bool: if is multiple of maxNumThread
	//if yes, info[0]: number of blocks, info[1] = maxNumThread
	//if no, info[0]: number of blocks except of the last block, info[1]: number of thread in the last block
	bool isMul = howPartition( rLen, numThread, info );

	//scan the isP2 blocks	
	unsigned int numBlock = info[0];
	unsigned int numElementsPerBlock = 0;
	unsigned int extraSpace = 0;
	unsigned int sharedMemSize = 0;

	int* d_temp; //for coalsed 
	GPUMALLOC( (void**)&d_temp, sizeof(int)*rLen );

	unsigned int base = 0;
	unsigned int offset = 0;
	int* d_data;

	if( numBlock > 0 )
	{
		int numChunk = ceil( (float)numBlock/numMaxBlock );

		for( int chunkIdx = 0; chunkIdx < numChunk; chunkIdx++ )
		{
			base = chunkIdx*numElementsPerBlock*numMaxBlock;
			offset = chunkIdx*numMaxBlock;
			int subNumBlock = (chunkIdx == (numChunk - 1))?( numBlock - chunkIdx*numMaxBlock ):(numMaxBlock);

			numElementsPerBlock = numThread*2;
			extraSpace = numElementsPerBlock/NUM_BANKS;
			sharedMemSize = sizeof(int)*( numElementsPerBlock + extraSpace );
#ifdef SHARED_MEM
			//printf( "YES, SHARED MEMORY\n" );
			perscanFirstPass_kernel<<<subNumBlock, numThread, sharedMemSize>>>
				( d_temp, d_scanBlockSums[0], d_Rin, numElementsPerBlock, true, base, offset, OPERATOR );
#else
			GPUMALLOC( (void**)&d_data, sharedMemSize*subNumBlock );
			perscanFirstPass_kernel<<<subNumBlock, numThread>>>
				( d_data, d_temp, d_scanBlockSums[0], d_Rin, numElementsPerBlock, true, base, offset, OPERATOR, sharedMemSize );
			GPUFREE( d_data );
#endif
			cudaThreadSynchronize();
		}
	
	}

	//scan the single not isP2 block
	if( (!isMul) || (numBlock == 0) )
	{
		base = numElementsPerBlock*info[0];
		offset = info[0];
		unsigned int remainer = rLen - numElementsPerBlock*info[0];

		numThread = info[1];//update the numThread
		
		//if only one elements
		if( numThread == 0 )
		{
			copyLastElement<<<1, 1>>>(d_scanBlockSums[0], d_Rin, base, offset);
			cudaThreadSynchronize();	
		}
		else
		{
			numBlock = 1;
			numElementsPerBlock = numThread*2;
			extraSpace = numElementsPerBlock/NUM_BANKS;
			sharedMemSize = sizeof(int)*( numElementsPerBlock + extraSpace );
			
			if( isPowerOfTwo( remainer ) )
			{

#ifdef SHARED_MEM
				perscanFirstPass_kernel<<<numBlock, numThread, sharedMemSize>>>
					( d_temp, d_scanBlockSums[0], d_Rin, remainer, true, base, offset, OPERATOR );
#else
				GPUMALLOC( (void**)&d_data, sharedMemSize*numBlock );
				perscanFirstPass_kernel<<<numBlock, numThread>>>
					( d_data, d_temp, d_scanBlockSums[0], d_Rin, remainer, true, base, offset, OPERATOR, sharedMemSize );
				GPUFREE( d_data );
#endif
				cudaThreadSynchronize();	
			}
			else
			{
#ifdef SHARED_MEM
				perscanFirstPass_kernel<<<numBlock, numThread, sharedMemSize>>>
					( d_temp, d_scanBlockSums[0], d_Rin, remainer, false, base, offset, OPERATOR );
#else
				GPUMALLOC( (void**)&d_data, sharedMemSize*numBlock);
				perscanFirstPass_kernel<<<numBlock, numThread>>>
					( d_data, d_temp, d_scanBlockSums[0], d_Rin, remainer, false, base, offset, OPERATOR, sharedMemSize );
				GPUFREE( d_data );
#endif
				cudaThreadSynchronize();	
			}	
		}
	
	}

	GPUFREE( d_temp );
	//test

/*#ifdef DEBUG
	Record* h_Rin = (Record*)malloc( sizeof(Record)*rLen );
	CUDA_SAFE_CALL( cudaMemcpy( h_Rin, d_Rin, sizeof(Record)*rLen, cudaMemcpyDeviceToHost ) );
	
	CUDA_SAFE_CALL( cudaMemcpy( h_scanBlockSums[0], d_scanBlockSums[0], sizeof(int)*levelSize[0], 
		cudaMemcpyDeviceToHost ) );

	int gpuSum = 0; 
	for( int i = 0; i < levelSize[0]; i++ )
	{
		gpuSum += h_scanBlockSums[0][i];
	}
	printf( "\nGPU sum: %d\n", gpuSum );

	int cpuSum = 0; 
	for( int i = 0; i < rLen; i++ )
	{
		cpuSum += h_Rin[i].y;
	}
	printf( "\nCPU sum:  %d\n", cpuSum );
#endif*/
}

__global__
void getResult_kernel( int* d_Result, Record* d_Rout, int rLen, int OPERATOR )
{
	d_Rout[0].x = 0;
	d_Rout[0].y = d_Result[0];

	if( OPERATOR == REDUCE_AVERAGE )
	{
		d_Rout[0].y = d_Result[0]/rLen;
	}
}


int reduceBlockSums( Record* d_Rout, int maxNumThread, int OPERATOR, int rLen )
{
	int* info = (int*)malloc( sizeof(int)*2 );

	int* d_data;

	//get the information of partition	
	//return bool: if is multiple of maxNumThread
	//if yes, info[0]: number of blocks, info[1] = maxNumThread
	//if no, info[0]: number of blocks except of the last block, info[1]: number of thread in the last block
	for( int level = 0; level < ( d_numLevelsAllocated - 1 ); level++ )
	{
		bool isMul = howPartition( levelSize[level], maxNumThread, info );

		unsigned int numBlock = info[0];
		unsigned int numElementsPerBlock = 0;
		unsigned int extraSpace = 0;
		unsigned int sharedMemSize = 0;
		
		//scan the isP2 blocks
		if( numBlock > 0 )
		{
			numElementsPerBlock = maxNumThread*2;
			extraSpace = numElementsPerBlock/NUM_BANKS;
			sharedMemSize = sizeof(int)*( numElementsPerBlock + extraSpace );			

#ifdef SHARED_MEM
			printf( " YES, SHARED MEMORY \n" );
			perscan_kernel<<<numBlock, maxNumThread, sharedMemSize>>>
				( d_scanBlockSums[level + 1], d_scanBlockSums[level], numElementsPerBlock, true, 0, 0, OPERATOR );
#else
			printf( " NO SHARED MEMORY\n" );
			GPUMALLOC( (void**)&d_data, sharedMemSize*numBlock);
			perscan_kernel<<<numBlock, maxNumThread>>>
				( d_data, d_scanBlockSums[level + 1], d_scanBlockSums[level], numElementsPerBlock, true, 0, 0, OPERATOR, sharedMemSize );
			GPUFREE( d_data );
#endif
			cudaThreadSynchronize();	
		}

		//scan the single not isP2 block
		if( (!isMul) || (numBlock == 0) )
		{
			unsigned int base = numElementsPerBlock*info[0];
			unsigned int offset = info[0];
			unsigned int remainer = levelSize[level] - numElementsPerBlock*info[0];

			int numThread = info[1];//update the numThread
			
			//only one number in the last block
			if( numThread == 0 )
			{				
				CUDA_SAFE_CALL( cudaMemcpy( d_scanBlockSums[level+1]+offset, d_scanBlockSums[level]+base, sizeof(int), cudaMemcpyDeviceToDevice ) );
			}
			else
			{
				numBlock = 1;
				numElementsPerBlock = numThread*2;
				extraSpace = numElementsPerBlock/NUM_BANKS;
				sharedMemSize = sizeof(int)*( numElementsPerBlock + extraSpace );
				
				if( isPowerOfTwo( remainer ) )
				{
#ifdef SHARED_MEM
					perscan_kernel<<<numBlock, numThread, sharedMemSize>>>
						( d_scanBlockSums[level + 1], d_scanBlockSums[level], remainer, true, base, offset, OPERATOR );
#else
					GPUMALLOC( (void**)&d_data, sharedMemSize*numBlock);
					perscan_kernel<<<numBlock, numThread>>>
						( d_data, d_scanBlockSums[level + 1], d_scanBlockSums[level], remainer, true, base, offset, OPERATOR, sharedMemSize );
					GPUFREE( d_data );
#endif
					cudaThreadSynchronize();	
				}
				else
				{
#ifdef SHARED_MEM
					perscan_kernel<<<numBlock, numThread, sharedMemSize>>>
						( d_scanBlockSums[level + 1], d_scanBlockSums[level], remainer, false, base, offset, OPERATOR );
#else
					GPUMALLOC( (void**)&d_data, sharedMemSize*numBlock);
					perscan_kernel<<<numBlock, numThread>>>
						( d_data, d_scanBlockSums[level + 1], d_scanBlockSums[level], remainer, false, base, offset, OPERATOR, sharedMemSize );
					GPUFREE( d_data );
#endif
					cudaThreadSynchronize();	
				}
			}
			
		}
	}
	
	//get the last sum	
	getResult_kernel<<<1, 1>>>( d_scanBlockSums[d_numLevelsAllocated - 1], d_Rout, rLen, OPERATOR );
	cudaThreadSynchronize();	

	//return h_sum[0];

	return 1;
}

void reduce_deallocBlockSums()
{
    for (int i = 0; i < g_numLevelsAllocated; i++)
    {
        cudaFree(g_scanBlockSums[i]);
    }

    free((void**)g_scanBlockSums);

    g_scanBlockSums = 0;
    g_numEltsAllocated = 0;
    g_numLevelsAllocated = 0;
}

int reduceImpl( Record* d_Rin, int rLen, Record* d_Rout, int OPERATOR, int numThread, int numMaxBlock )
{
#ifdef COALESCED
	printf( " YES, COALESCED\n" );
#else
	printf( " NO COALESCED\n " );
#endif
	preallocBlockSums( rLen, numThread );
	reduceFirstPass( d_Rin, rLen, numThread, numMaxBlock, OPERATOR );
	///gpuResult = reduceBlockSums( d_Rout, numThread, OPERATOR );
	
	int result = reduceBlockSums( d_Rout, numThread, OPERATOR, rLen );

	reduce_deallocBlockSums();

	return result;
}

#endif

