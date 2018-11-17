/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 *  saven modified NV's code
 */

#ifndef _STRING_BITONIC_SAVEN_KERNEL_H_
#define _STRING_BITONIC_SAVEN_KERNEL_H_

#include "StringCmp.cu"

#define SHARED_MEM_INT2 512//256
__device__ inline void swap(cmp_type_t & a, cmp_type_t & b)
{
	// Alternative swap doesn't use a temporary register:
	// a ^= b;
	// b ^= a;
	// a ^= b;
	
    cmp_type_t tmp = a;
    a = b;
    b = tmp;
}

/*__global__ void bitonicSortSingleBlock_kernel(void* d_rawData, int totalLenInBytes, Record * d_values, int* d_bound, int startBlock, int numBlock, Record *d_output)
{
	__shared__ Record bs_shared[SHARED_MEM_INT2];
	

    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bx)*numThread+tid;
	
	if(tid<bs_numElement)
	{
		bs_shared[tid] = d_values[tid+bs_pStart];
	}
	else
	{
		bs_shared[tid].x =-1;
		bs_shared[tid].y =-1;
	}

    __syncthreads();

    // Parallel bitonic sort.
	int compareValue=0;
    for (int k = 2; k <= SHARED_MEM_INT2; k *= 2)
    {
        // Bitonic merge:
        for (int j = k / 2; j>0; j /= 2)
        {
            int ixj = tid ^ j;
            
            if (ixj > tid)
            {
                if ((tid & k) == 0)
                {
					compareValue=getCompareValue(d_rawData, bs_shared[tid].x, bs_shared[ixj].x);
					//if (shared[tid] > shared[ixj])
					if(compareValue>0)
                    {
                        swap(bs_shared[tid], bs_shared[ixj]);
                    }
                }
                else
                {
					compareValue=getCompareValue(d_rawData, bs_shared[tid].x, bs_shared[ixj].x);
                    //if (shared[tid] < shared[ixj])
					if(compareValue<0)
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
		d_output[tid+bs_pStart] = bs_shared[tid+SHARED_MEM_INT2-bs_numElement];
	}
}*/

__global__ void string_bitonicSortMultipleBlocks_kernel(void* d_rawData, int totalLenInBytes, cmp_type_t * d_values, int* d_bound, int startBlock, int numBlock, cmp_type_t *d_output)
{
	__shared__ int bs_pStart;
	__shared__ int bs_pEnd;
	__shared__ int bs_numElement;
    __shared__ cmp_type_t bs_shared[SHARED_MEM_INT2];
	

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
		bs_pStart=d_bound[(bid+startBlock)<<1];
		bs_pEnd=d_bound[((bid+startBlock)<<1)+1];
		bs_numElement=bs_pEnd-bs_pStart;
		//if(bid==82&& bs_pStart==6339)
		//	printf("%d, %d, %d\n", bs_pStart, bs_pEnd, bs_numElement);
		
	}
	__syncthreads();
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
	}

    __syncthreads();

    // Parallel bitonic sort.
	int compareValue=0;
    for (int k = 2; k <= SHARED_MEM_INT2; k *= 2)
    {
        // Bitonic merge:
        for (int j = k / 2; j>0; j /= 2)
        {
            int ixj = tid ^ j;
            
            if (ixj > tid)
            {
                if ((tid & k) == 0)
                {
					compareValue=getCompareValue(d_rawData, bs_shared[tid].x, bs_shared[ixj].x);
					//if (shared[tid] > shared[ixj])
					if(compareValue>0)
                    {
                        swap(bs_shared[tid], bs_shared[ixj]);
                    }
                }
                else
                {
					compareValue=getCompareValue(d_rawData, bs_shared[tid].x, bs_shared[ixj].x);
                    //if (shared[tid] < shared[ixj])
					if(compareValue<0)
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
		d_output[tid+bs_pStart] = bs_shared[tid+SHARED_MEM_INT2-bs_numElement];
		//if(bid==82&& bs_pStart==6339)
		//	printf("tid %d, %d, %d, %d, %d\n", tid, tid+bs_pStart, bs_pStart,bs_pEnd, d_output[tid+bs_pStart].x);
		//if(6342==tid+bs_pStart)
		//	printf(">>bid %d, tid %d, %d, %d, %d, %d\n", bid, tid, tid+bs_pStart, bs_pStart,bs_pEnd, d_output[tid+bs_pStart].x);


	}
}


__global__ void initialize_kernel(cmp_type_t* d_data, int startPos, int rLen, cmp_type_t value)
{
    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	d_data[pos]=value;
}

__global__ void int4toint2_kernel(int4* d_input, int startPos, int rLen, Record* d_output)
{
    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	{
		int4 value=d_input[pos];
		d_output[pos].x=value.x;
		d_output[pos].y=resultID;//this is for RID.
	}
}


__global__ void getIntYArray_kernel(Record* d_input, int startPos, int rLen, int* d_output)
{
    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	{
		Record value=d_input[pos];
		d_output[pos]=value.y;
	}
}


__global__ void getXYArray_kernel(cmp_type_t* d_input, int startPos, int rLen, Record* d_output)
{
    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	{
		cmp_type_t value=d_input[pos];
		d_output[pos].x=value.x;
		d_output[pos].y=value.y;
	}
}

__global__ void getZWArray_kernel(cmp_type_t* d_input, int startPos, int rLen, Record* d_output)
{
    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	{
		cmp_type_t value=d_input[pos];
		d_output[pos].x=value.z;
		d_output[pos].y=value.w;
	}
}


__global__ void setXYArray_kernel(cmp_type_t* d_input, int startPos, int rLen, Record* d_value)
{
    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	{
		cmp_type_t value=d_input[pos];
		value.x=d_value[pos].x;
		value.y=d_value[pos].y;
		d_input[pos]=value;
	}
}

__global__ void setZWArray_kernel(cmp_type_t* d_input, int startPos, int rLen, Record* d_value)
{
    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	{
		cmp_type_t value=d_input[pos];
		value.z=d_value[pos].x;
		value.w=d_value[pos].y;
		d_input[pos]=value;
	}
}



#endif // _BITONIC_SAVEN_KERNEL_H_
