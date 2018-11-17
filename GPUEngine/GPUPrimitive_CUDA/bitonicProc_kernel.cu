/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 *  saven modified NV's code
 */

#ifndef _BITONIC_SAVEN_KERNEL_H_
#define _BITONIC_SAVEN_KERNEL_H_

#include "QP_Utility.cu"

#define SHARED_MEM_INT2 512
__device__ inline void swap(Record & a, Record & b)
{
	// Alternative swap doesn't use a temporary register:
	// a ^= b;
	// b ^= a;
	// a ^= b;
	
    Record tmp = a;
    a = b;
    b = tmp;
}


__device__ int d_log2(int value)
{
	int result=0;
	while(value>1)
	{
		value=value>>1;
		result++;
	}
	return result;
}

__device__ int d_log2Ceil(int value)
{
	int result=d_log2(value);
	if(value>(1<<result))
		result++;
	return result;
}

__global__ void bitonicSortMultipleBlocks_kernel( Record * d_values, Record* d_bound, int startBlock, int numBlock, Record *d_output)
{
	__shared__ int bs_pStart;
	//__shared__ int bs_pEnd;
	__shared__ int bs_numElement;
    __shared__ Record bs_shared[SHARED_MEM_INT2];
	

    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	//const int numThread=blockDim.x;
	//const int resultID=(bx)*numThread+tid;
	if(bid>=numBlock) return;

	if(tid==0)
	{
		Record value=d_bound[(bid+startBlock)];
		bs_pStart=value.x;
		//bs_pEnd=value.y;
		bs_numElement=value.y-value.x;
		//if(bid==82&& bs_pStart==6339)
		//	printf("%d, %d, %d\n", bs_pStart, bs_pEnd, bs_numElement);
		
	}
	__syncthreads();
	if(bs_numElement<=1)
		return;
    // Copy input to shared mem.
	if(tid<bs_numElement)
	{
		bs_shared[tid] = d_values[tid+bs_pStart];
		//if(bid==82 && bs_pStart==6339)
		//	printf("tid %d, pos, %d, %d, %d, %d\n", tid,tid+bs_pStart, bs_pStart,bs_pEnd, d_values[tid+bs_pStart].x);
		//if(6342==tid+bs_pStart)
		//	printf(")))tid %d, pos, %d, %d, %d, %d\n", tid,tid+bs_pStart, bs_pStart,bs_pEnd, d_values[tid+bs_pStart].x);
	}
	else
	{
		bs_shared[tid].x =-1;
		bs_shared[tid].y =-1;
	}

    __syncthreads();

    // Parallel bitonic sort.
	//int compareValue=0;
	int bound=(1<<d_log2Ceil(bs_numElement));
	for (int k = 2; k <= bound; k *= 2)
    //for (int k = 2; k <= SHARED_MEM_INT2; k *= 2)
    {
        // Bitonic merge:
        for (int j = k / 2; j>0; j /= 2)
        {
            int ixj = tid ^ j;
            
            if (ixj > tid)
            {
                if ((tid & k) == 0)
                {
					//compareValue=getCompareValue(d_rawData, bs_shared[tid].x, bs_shared[ixj].x);
					if (bs_shared[tid].y > bs_shared[ixj].y)
					{
                        swap(bs_shared[tid], bs_shared[ixj]);
                    }
                }
                else
                {
					//compareValue=getCompareValue(d_rawData, bs_shared[tid].x, bs_shared[ixj].x);
                    if (bs_shared[tid].y < bs_shared[ixj].y)
					{
                        swap(bs_shared[tid], bs_shared[ixj]);
                    }
                }
            }
            
            __syncthreads();
        }
    }

    // Write result.
	if(tid<bs_numElement)
	{
		//d_output[tid+bs_pStart] = bs_shared[tid+SHARED_MEM_INT2-bs_numElement];
		d_output[tid+bs_pStart] = bs_shared[tid+bound-bs_numElement];
		//if(bid==82&& bs_pStart==6339)
		//	printf("tid %d, %d, %d, %d, %d\n", tid, tid+bs_pStart, bs_pStart,bs_pEnd, d_output[tid+bs_pStart].x);
		//if(6342==tid+bs_pStart)
		//	printf(">>bid %d, tid %d, %d, %d, %d, %d\n", bid, tid, tid+bs_pStart, bs_pStart,bs_pEnd, d_output[tid+bs_pStart].x);


	}
}

__global__ void
bitonicMultipleLargeBlocks_kernel(  Record* d_R, int largeBlockSize, int numChunk, int k, int j)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	if(by>=numChunk)
		return;
	int dataIdx = bx*blockDim.x + tid;
	int base=by*largeBlockSize;

	int ixj = dataIdx^j;

	if( ixj > dataIdx )
	{
		Record tmpR = d_R[dataIdx+base];
		Record tmpIxj = d_R[ixj+base];
		if( (dataIdx&k) == 0 )
		{
			if( tmpR.y > tmpIxj.y )
			//if(getCompareValue(d_rawData, tmpR.x, tmpIxj.x)==1)
			{
				d_R[dataIdx+base] = tmpIxj;
				d_R[ixj+base] = tmpR;
			}
		}
		else
		{
			if( tmpR.y < tmpIxj.y )
			{
				d_R[dataIdx+base] = tmpIxj;
				d_R[ixj+base] = tmpR;
			}
		}
	}
}



#endif // _BITONIC_SAVEN_KERNEL_H_
