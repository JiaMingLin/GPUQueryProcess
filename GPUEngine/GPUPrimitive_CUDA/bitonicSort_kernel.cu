

#ifndef _BITONICSORT_KERNEL_
#define _BITONICSORT_KERNEL_

#include <stdio.h>


#define NUM_BLOCKS_CHUNK (512)
#define	NUM_THREADS_CHUNK (512)
//#define	NUM_THREADS_CHUNK (256)
//#define CHUNK_SIZE (NUM_BLOCKS_CHUNK*NUM_THREADS_CHUNK)




#ifdef SHARED_MEM
__global__ void
partBitonicSortKernel( Record* d_R, unsigned int numRecords, int chunkIdx, 
					  int unitSize, int numThreadPerBlock, int chunkSize, int numBlockChunk)
{
	extern __shared__ Record shared[];

	int tx = threadIdx.x;
	int bx = blockIdx.x;

	//load the data
	int dataIdx = chunkIdx*chunkSize+bx*blockDim.x+tx;
	int unitIdx = ((numBlockChunk*chunkIdx + bx)/unitSize)&1;
	shared[tx] = d_R[dataIdx];
	__syncthreads();
	int ixj=0;
	int a=0;
	Record temp1;
	Record temp2;
	int k = numThreadPerBlock;

	if(unitIdx == 0)
	{
		for (int j = (k>>1); j>0; j =(j>>1))
		{
			ixj = tx ^ j;
			//a = (shared[tx].y - shared[ixj].y);				
			temp1=shared[tx];
			temp2= shared[ixj];
			if (ixj > tx) {
				a=temp1.y-temp2.y;
				//a=getCompareValue(d_rawData, temp1.x, temp2.x);
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
				a=temp1.y-temp2.y;					
				//a=getCompareValue(d_rawData, temp1.x, temp2.x);
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
unitBitonicSortKernel( Record* d_R, unsigned int numRecords, int chunkIdx, int numThreadPerBlock,
					 int chunkSize, int numBlockChunk)
{
	extern __shared__ Record shared[];

	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int unitIdx = (numBlockChunk*chunkIdx + bx)&1;

	//load the data
	int dataIdx = chunkIdx*chunkSize+bx*blockDim.x+tx;
	shared[tx] = d_R[dataIdx];
	__syncthreads();

	Record temp1;
	Record temp2;
	int ixj=0;
	int a=0;
	if(unitIdx == 0)
	{
		for (int k = 2; k <= numThreadPerBlock; (k =k<<1))
		{
			// bitonic merge:
			for (int j = (k>>1); j>0; (j=j>>1))
			{
				ixj = tx ^ j;	
				temp1=shared[tx];
				temp2= shared[ixj];
				if (ixj > tx) {					
					a=temp1.y-temp2.y;
					//a=getCompareValue(d_rawData, temp1.x, temp2.x);
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
		for (int k = 2; k <= numThreadPerBlock; (k =k<<1))
		{
			// bitonic merge:
			for (int j = (k>>1); j>0; (j=j>>1))
			{
				ixj = tx ^ j;
				temp1=shared[tx];
				temp2= shared[ixj];
				if (ixj > tx) {					
					a=temp1.y-temp2.y;
					//a=getCompareValue(d_rawData, temp1.x, temp2.x);
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
#else//no shared memory

__global__ void
partBitonicSortKernel( Record* d_R, unsigned int numRecords, int chunkIdx, 
					  int unitSize, int numThreadPerBlock, int chunkSize, int numBlockChunk)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;

	//load the data
	//int dataIdx = chunkIdx*chunkSize+bx*blockDim.x+tx;
	int base=chunkIdx*chunkSize+bx*blockDim.x;
	int unitIdx = ((numBlockChunk*chunkIdx + bx)/unitSize)&1;
	int ixj=0;
	int a=0;
	Record temp1;
	Record temp2;
	int k = numThreadPerBlock;

	if(unitIdx == 0)
	{
		for (int j = (k>>1); j>0; j =(j>>1))
		{
			ixj = tx ^ j;
			//a = (shared[tx].y - shared[ixj].y);				
			temp1=d_R[base+tx];
			temp2= d_R[base+ixj];
			if (ixj > tx) {
				a=temp1.y-temp2.y;
				//a=getCompareValue(d_rawData, temp1.x, temp2.x);
				if ((tx & k) == 0) {
					if ( (a>0)) {
						d_R[base+tx]=temp2;
						d_R[base+ixj]=temp1;
					}
				}
				else {
					if ( (a<0)) {
						d_R[base+tx]=temp2;
						d_R[base+ixj]=temp1;
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
			temp1=d_R[base+tx];
			temp2= d_R[base+ixj];
			
			if (ixj > tx) {					
				a=temp1.y-temp2.y;					
				//a=getCompareValue(d_rawData, temp1.x, temp2.x);
				if ((tx & k) == 0) {
					if( (a<0))
					{
						d_R[base+tx]=temp2;
						d_R[base+ixj]=temp1;
					}
				}
				else {
					if( (a>0))
					{
						d_R[base+tx]=temp2;
						d_R[base+ixj]=temp1;
					}
				}
			}
			
			__syncthreads();
		}
	}

	//d_R[dataIdx] = shared[tx];
}

__global__ void
unitBitonicSortKernel( Record* d_R, unsigned int numRecords, int chunkIdx, int numThreadPerBlock,
					 int chunkSize, int numBlockChunk)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int unitIdx = (numBlockChunk*chunkIdx + bx)&1;

	//load the data
	//int dataIdx = chunkIdx*chunkSize+bx*blockDim.x+tx;
	int base=chunkIdx*chunkSize+bx*blockDim.x;
	__syncthreads();

	Record temp1;
	Record temp2;
	int ixj=0;
	int a=0;
	if(unitIdx == 0)
	{
		for (int k = 2; k <= numThreadPerBlock; (k =k<<1))
		{
			// bitonic merge:
			for (int j = (k>>1); j>0; (j=j>>1))
			{
				ixj = tx ^ j;	
				temp1=d_R[base+tx];
				temp2= d_R[base+ixj];
				if (ixj > tx) {					
					a=temp1.y-temp2.y;
					//a=getCompareValue(d_rawData, temp1.x, temp2.x);
					if ((tx & k) == 0) {
						if ( (a>0)) {
							d_R[base+tx]=temp2;
							d_R[base+ixj]=temp1;
						}
					}
					else {
						if ( (a<0)) {
							d_R[base+tx]=temp2;
							d_R[base+ixj]=temp1;
						}
					}
				}
				
				__syncthreads();
			}
		}
	}
	else
	{
		for (int k = 2; k <= numThreadPerBlock; (k =k<<1))
		{
			// bitonic merge:
			for (int j = (k>>1); j>0; (j=j>>1))
			{
				ixj = tx ^ j;
				temp1=d_R[base+tx];
				temp2= d_R[base+ixj];
				if (ixj > tx) {					
					a=temp1.y-temp2.y;
					//a=getCompareValue(d_rawData, temp1.x, temp2.x);
					if ((tx & k) == 0) {
						if( (a<0))
						{
							d_R[base+tx]=temp2;
							d_R[base+ixj]=temp1;
						}
					}
					else {
						if( (a>0))
						{
							d_R[base+tx]=temp2;
							d_R[base+ixj]=temp1;
						}
					}
				}
				
				__syncthreads();
			}
		}

	}

	//d_R[dataIdx] = shared[tx];
}
#endif

__global__ void
bitonicKernel(  Record* d_R, unsigned int numRecords, int k, int j)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tid = threadIdx.x;
	int dataIdx = by*gridDim.x*blockDim.x + bx*blockDim.x + tid;

	int ixj = dataIdx^j;

	if( ixj > dataIdx )
	{
		Record tmpR = d_R[dataIdx];
		Record tmpIxj = d_R[ixj];
		if( (dataIdx&k) == 0 )
		{
			if( tmpR.y > tmpIxj.y )
			//if(getCompareValue(d_rawData, tmpR.x, tmpIxj.x)==1)
			{
				d_R[dataIdx] = tmpIxj;
				d_R[ixj] = tmpR;
			}
		}
		else
		{
			if( tmpR.y < tmpIxj.y )
			{
				d_R[dataIdx] = tmpIxj;
				d_R[ixj] = tmpR;
			}
		}
	}
}


#endif // #ifndef _TEMPLATE_KERNEL_H_
