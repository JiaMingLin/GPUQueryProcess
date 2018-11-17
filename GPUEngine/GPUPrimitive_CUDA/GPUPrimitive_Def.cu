#ifndef GPUPRIMITIVE_DEF_CU
#define GPUPRIMITIVE_DEF_CU

#include "stdlib.h"

unsigned int gpuMemSize = 0;

#define GPUMALLOC(D_POINTER, SIZE) CUDA_SAFE_CALL( cudaMalloc( D_POINTER, SIZE) )
#define CPUMALLOC(H_POINTER, SIZE) CUDA_SAFE_CALL(cudaMallocHost (H_POINTER, SIZE))

#define CPUFREE(H_POINTER) if(H_POINTER!=NULL) CUDA_SAFE_CALL(cudaFreeHost (H_POINTER))
#define GPUFREE(D_POINTER) CUDA_SAFE_CALL( cudaFree( D_POINTER) )


#define TOGPU(D_POINTER,H_POINTER, SIZE)  CUDA_SAFE_CALL(cudaMemcpy(D_POINTER,H_POINTER, SIZE, cudaMemcpyHostToDevice))
#define FROMGPU(H_POINTER, D_POINTER, SIZE)  CUDA_SAFE_CALL(cudaMemcpy(H_POINTER, D_POINTER, SIZE, cudaMemcpyDeviceToHost))
#define GPUTOGPU(D_TO, D_FROM, SIZE)  CUDA_SAFE_CALL(cudaMemcpy(D_TO, D_FROM, SIZE, cudaMemcpyDeviceToDevice))

#define GPUTOGPU_CONSTANT(D_TO, D_FROM, SIZE, OFFSET) CUDA_SAFE_CALL(cudaMemcpyToSymbol(D_TO, D_FROM, SIZE, OFFSET,cudaMemcpyDeviceToDevice))


#define SHARED_MEMORY_PER_PROCESSOR (8*1024)

#define NLJ_NUM_PROCESSOR 16//for GTX
#define NLJ_SHARED_MEM_PER_PROCESSOR (SHARED_MEMORY_PER_PROCESSOR)
#define NLJ_SHARED_MEM (NLJ_SHARED_MEM_PER_PROCESSOR*NLJ_NUM_PROCESSOR)
#define NLJ_MAX_NUM_BLOCK_PER_DIM (32*1024)
#define NLJ_NUM_THREADS_PER_BLOCK 512
#define NLJ_NUM_TUPLE_PER_THREAD 2
#define NLJ_S_BLOCK_SIZE (NLJ_NUM_THREADS_PER_BLOCK*NLJ_NUM_TUPLE_PER_THREAD)
#define NLJ_R_BLOCK_SIZE NLJ_NUM_THREADS_PER_BLOCK


#define PRED_EQUAL2(DATA) (DATA[0]==DATA[1])
#define PRED_EQUAL(V1,V2) (V1==V2)

//ke's definitions.
/////////////////////////////////////////////////////////////////////////defines
#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

///////////////////////////////////////general define

#define _charHistOpt 0	//NOTE: if charOpt, then can't have too many (>255) duplicate pid!!



typedef int2 Rec;
struct/* __align__(16)*/ RecRS
{
	unsigned int val;
	int ridR;
	int ridS;
};
//end of Ke's definition.


#define TEST_MAX (1<<30)
#define TEST_MIN (0)

#define SHARED_MEM 1

#define COALESCED 1

//#define OUTPUT_INFO 1

#define BINARY_SEARCH 1

//#define BINARY_SEARCH_HASH 1

#define CONSTANT_BUFFER_SIZE (1024*64)



#endif
