

#ifndef _STRING_BITONICSORT_KERNEL_
#define _STRING_BITONICSORT_KERNEL_

#include <stdio.h>
#include <common.cu>
#include <StringCmp.cu>

#define NUM_BLOCKS_CHUNK (512)
#define	STRING_NUM_THREADS_CHUNK (256)
#define CHUNK_SIZE (NUM_BLOCKS_CHUNK*STRING_NUM_THREADS_CHUNK)
#define NUM_CHUNKS_R (NUM_RECORDS_R/CHUNK_SIZE)






__global__ void
string_partBitonicSortKernel( void* d_rawData, int totalLenInBytes,cmp_type_t* d_R, unsigned int numRecords, int chunkIdx, int unitSize)
{
	__shared__ cmp_type_t shared[STRING_NUM_THREADS_CHUNK];

	int tx = threadIdx.x;
	int bx = blockIdx.x;

	//load the data
	int dataIdx = chunkIdx*CHUNK_SIZE+bx*blockDim.x+tx;
	int unitIdx = ((NUM_BLOCKS_CHUNK*chunkIdx + bx)/unitSize)&1;
	shared[tx] = d_R[dataIdx];
	__syncthreads();
	int ixj=0;
	int a=0;
	cmp_type_t temp1;
	cmp_type_t temp2;
	int k = STRING_NUM_THREADS_CHUNK;

	if(unitIdx == 0)
	{
		for (int j = (k>>1); j>0; j =(j>>1))
		{
			ixj = tx ^ j;
			//a = (shared[tx].y - shared[ixj].y);				
			temp1=shared[tx];
			temp2= shared[ixj];
			if (ixj > tx) {
				//a=temp1.y-temp2.y;
				//a=compareString((void*)(((char4*)d_rawData)+temp1.x),(void*)(((char4*)d_rawData)+temp2.x)); 
				a=getCompareValue(d_rawData, temp1.x, temp2.x);
				if ((tx & k) == 0) {
					if ( (a>0)) {
						shared[tx]=temp2;
						shared[ixj]=temp1;
					}
				}
				else {
					if ( (a<0)) {
						shared[tx]=temp2;
						shared[ixj]=temp1;
					}
				}
			}
				
			__syncthreads();
		}
	}
	else
	{
		for (int j = (k>>1); j>0; j =(j>>1))
		{
			ixj = tx ^ j;
			temp1=shared[tx];
			temp2= shared[ixj];
			
			if (ixj > tx) {					
				//a=temp1.y-temp2.y;					
				//a=compareString((void*)(((char4*)d_rawData)+temp1.x),(void*)(((char4*)d_rawData)+temp2.x));
				a=getCompareValue(d_rawData, temp1.x, temp2.x);
				if ((tx & k) == 0) {
					if( (a<0))
					{
						shared[tx]=temp2;
						shared[ixj]=temp1;
					}
				}
				else {
					if( (a>0))
					{
						shared[tx]=temp2;
						shared[ixj]=temp1;
					}
				}
			}
			
			__syncthreads();
		}
	}

	d_R[dataIdx] = shared[tx];
}

__global__ void
string_unitBitonicSortKernel(void* d_rawData, int totalLenInBytes, cmp_type_t* d_R, unsigned int numRecords, int chunkIdx )
{
	__shared__ cmp_type_t shared[STRING_NUM_THREADS_CHUNK];

	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int unitIdx = (NUM_BLOCKS_CHUNK*chunkIdx + bx)&1;

	//load the data
	int dataIdx = chunkIdx*CHUNK_SIZE+bx*blockDim.x+tx;
	shared[tx] = d_R[dataIdx];
	__syncthreads();

	cmp_type_t temp1;
	cmp_type_t temp2;
	int ixj=0;
	int a=0;
	if(unitIdx == 0)
	{
		for (int k = 2; k <= STRING_NUM_THREADS_CHUNK; (k =k<<1))
		{
			// bitonic merge:
			for (int j = (k>>1); j>0; (j=j>>1))
			{
				ixj = tx ^ j;	
				temp1=shared[tx];
				temp2= shared[ixj];
				if (ixj > tx) {					
					//a=temp1.y-temp2.y;
					//a=compareString((void*)(((char4*)d_rawData)+temp1.x),(void*)(((char4*)d_rawData)+temp2.x));
					a=getCompareValue(d_rawData, temp1.x, temp2.x);
					if ((tx & k) == 0) {
						if ( (a>0)) {
							shared[tx]=temp2;
							shared[ixj]=temp1;
						}
					}
					else {
						if ( (a<0)) {
							shared[tx]=temp2;
							shared[ixj]=temp1;
						}
					}
				}
				
				__syncthreads();
			}
		}
	}
	else
	{
		for (int k = 2; k <= STRING_NUM_THREADS_CHUNK; (k =k<<1))
		{
			// bitonic merge:
			for (int j = (k>>1); j>0; (j=j>>1))
			{
				ixj = tx ^ j;
				temp1=shared[tx];
				temp2= shared[ixj];
				if (ixj > tx) {					
					//a=temp1.y-temp2.y;
					//a=compareString((void*)(((char4*)d_rawData)+temp1.x),(void*)(((char4*)d_rawData)+temp2.x));
					a=getCompareValue(d_rawData, temp1.x, temp2.x);
					if ((tx & k) == 0) {
						if( (a<0))
						{
							shared[tx]=temp2;
							shared[ixj]=temp1;
						}
					}
					else {
						if( (a>0))
						{
							shared[tx]=temp2;
							shared[ixj]=temp1;
						}
					}
				}
				
				__syncthreads();
			}
		}

	}

	d_R[dataIdx] = shared[tx];
}

__global__ void
string_bitonicKernel( void* d_rawData, int totalLenInBytes, cmp_type_t* d_R, unsigned int numRecords, int k, int j)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tid = threadIdx.x;
	int dataIdx = by*gridDim.x*blockDim.x + bx*blockDim.x + tid;

	int ixj = dataIdx^j;

	if( ixj > dataIdx )
	{
		cmp_type_t tmpR = d_R[dataIdx];
		cmp_type_t tmpIxj = d_R[ixj];
		if( (dataIdx&k) == 0 )
		{
			//if( tmpR.y > tmpIxj.y )
			//if(compareString((void*)(((char4*)d_rawData)+tmpR.x),(void*)(((char4*)d_rawData)+tmpIxj.x))==1) 
			if(getCompareValue(d_rawData, tmpR.x, tmpIxj.x)==1)
			{
				d_R[dataIdx] = tmpIxj;
				d_R[ixj] = tmpR;
			}
		}
		else
		{
			//if( tmpR.y < tmpIxj.y )
			//if(compareString((void*)(((char4*)d_rawData)+tmpR.x),(void*)(((char4*)d_rawData)+tmpIxj.x))==-1) 
			if(getCompareValue(d_rawData, tmpR.x, tmpIxj.x)==-1)
			{
				d_R[dataIdx] = tmpIxj;
				d_R[ixj] = tmpR;
			}
		}
	}
}


#endif // #ifndef _TEMPLATE_KERNEL_H_
