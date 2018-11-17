/*
modified by saven. on March 07. 
complete working on shared memory.
*/

#ifndef _SCAN_BEST_KERNEL_H_
#define _SCAN_BEST_KERNEL_H_



#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

// Define this to more rigorously avoid bank conflicts, even at the lower (root) levels of the tree
//#define ZERO_BANK_CONFLICTS 

#ifdef CHECK_BANK_CONFLICTS
#define TEMP(index)   CUT_BANK_CHECKER(temp, index)
#else
#define TEMP(index)   temp[index]
#endif


__device__ void scanPS(int *g_odata, int *g_idata, int n)
{
	extern __shared__ int partial_sum[];
    // Dynamically allocated shared memory for scan kernels
    int tid = threadIdx.x;

    int ai = tid;
    int bi = tid + (n>>1);

    // compute spacing to avoid bank conflicts
    int bankOffset1 = (ai >> LOG_NUM_BANKS);
    int bankOffset2 = (bi >> LOG_NUM_BANKS);
	// Cache the computational window in shared memory
    partial_sum[ai + bankOffset1] = g_idata[ai]; 
    partial_sum[bi + bankOffset2] = g_idata[bi]; 

    int offset = 1;

    // build the sum in place up the tree
    for (int d = n>>1; d > 0; d >>= 1)
    {
        __syncthreads();

        if (tid < d)      
        {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            ai += (ai >> LOG_NUM_BANKS);
            bi += (bi >> LOG_NUM_BANKS);
            partial_sum[bi] += partial_sum[ai];
        }

        offset *= 2;
    }

    // scan back down the tree

    // clear the last element
    if (tid == 0)
    {
        int conflict_offset = (n-1) >> LOG_NUM_BANKS;
        partial_sum[n - 1 + conflict_offset] = 0;
    }   

    // traverse down the tree building the scan in place
    for (int d = 1; d < n; d *= 2)
    {
        offset >>= 1;
        __syncthreads();

        if (tid < d)
        {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            ai += (ai >> LOG_NUM_BANKS);
            bi += (bi >> LOG_NUM_BANKS);
            int t  = partial_sum[ai];
            partial_sum[ai] = partial_sum[bi];
            partial_sum[bi] += t;
        }
    }

    __syncthreads();

    // write results to global memory
    g_odata[ai] = partial_sum[ai + bankOffset1]; 
    g_odata[bi] = partial_sum[bi + bankOffset2]; 
}

__global__ void computePS(int *g_odata, int *g_idata, int blockSize)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bid=bx+by*gridDim.x;
	int start=blockSize*bid;
	scanPS((g_odata+start), (g_idata+start), blockSize);
}




#endif // #ifndef _SCAN_BEST_KERNEL_H_

