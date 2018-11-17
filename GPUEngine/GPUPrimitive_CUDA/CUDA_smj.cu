#ifndef CUDA_SMJ_CU
#define CUDA_SMJ_CU

#include "sortImpl.cu"
#include "CSSTree.cu"
#include "getQuantile.cu"
#include "CUDA_ninlj.cu"
#include "joinMatchingBlocks.cu"
#include "GPU_Dll.h"

//d_Rin and d_Sin are sorted
int cuda_mj( Record* d_Rin, int rLen, Record* d_Sin, int sLen, Record** d_Joinout )
{
	int numResult = 0;

	//get the pivot
	unsigned int timer=0;
	startTimer(&timer);
	int interval=SMJ_NUM_THREADS_PER_BLOCK;
	int numQuanR=divRoundUp(rLen,interval);
	int* d_quanKeyR;
	GPUMALLOC((void**)&d_quanKeyR,numQuanR*sizeof(int)*2); 

	getQuantile(d_Rin, rLen, interval,d_quanKeyR,numQuanR); 
	endTimer("quantile R", &timer);
	//gpuPrint(d_quanKeyR, numQuanR*2, "d_quanR");
#ifdef BINARY_SEARCH
	printf("YES, BINARY_SERACH");
#else
	printf("NO, BINARY_SERACH");
#endif
	//find the matching boundary in S.
	startTimer(&timer);
	int* d_quanLocS;
	GPUMALLOC((void**)&d_quanLocS,numQuanR*sizeof(int)*2); 
	CUDA_CSSTree* tree;
	gpu_constructCSSTree(d_Sin, sLen, &tree);
	cuda_search_index_usingKeys(tree->data, tree->nDataNodes, tree->dir,
		tree->nDirNodes, d_quanKeyR, d_quanLocS, numQuanR*2);
	endTimer("quantile S", &timer);
	printf("numQuanInR, %d\n", numQuanR);

	//matching
	startTimer(&timer);
	numResult=joinMatchingBlocks(d_Rin, rLen,d_Sin,sLen, 
		d_quanLocS, numQuanR, d_Joinout);
	endTimer("joinMatchingBlocks", &timer);
	
	GPUFREE(d_quanKeyR);
	GPUFREE(d_quanLocS);

	return numResult;
}

//R is the smaller relation:)
int cuda_smj(Record *d_R, int rLen, Record *d_S, int sLen, Record** d_Joinout)
{
	int numResult=0;
	Record *d_Rout;
	GPUMALLOC((void**)&d_Rout, sizeof(Record)*rLen);
	//radixSortImpl(d_R,rLen,d_Rout);
	sortImpl(d_R,rLen,d_Rout);
	//GPUOnly_QuickSort( d_R, rLen, d_Rout );
	//gpuPrintInt2(d_Rout,rLen, "d_R");
	//gpuValidateSort(d_Rout,rLen);
	Record *d_Sout;
	GPUMALLOC((void**)&d_Sout, sizeof(Record)*sLen);
	//radixSortImpl(d_S,sLen,d_Sout);
	sortImpl(d_S,sLen,d_Sout);
	//GPUOnly_QuickSort(d_S,sLen,d_Sout);
	//gpuValidateSort(d_Sout,rLen);

	//just for the experiment to avoid out of memory
#ifndef SHARED_MEM
	GPUFREE( d_R );
	GPUFREE( d_S );
#endif

	numResult = cuda_mj( d_Rout, rLen, d_Sout, sLen, d_Joinout );

	GPUFREE( d_Rout );
	GPUFREE( d_Sout );

	return numResult;
}

extern "C"
int GPUOnly_mj( Record* d_Rin, int rLen, Record* d_Sin, int sLen, Record** d_Joinout )
{
	return cuda_mj( d_Rin, rLen, d_Sin, sLen, d_Joinout );
}

extern "C"
int GPUCopy_mj( Record* h_Rin, int rLen, Record* h_Sin, int sLen, Record** h_Joinout )
{
	Record* d_Rin;
	Record* d_Sin;
	Record* d_Joinout;

	GPUMALLOC( (void**)&d_Rin, sizeof(Record)*rLen );
	GPUMALLOC( (void**)&d_Sin, sizeof(Record)*sLen );

	TOGPU( d_Rin, h_Rin, sizeof(Record)*rLen );
	TOGPU( d_Sin, h_Sin, sizeof(Record)*sLen );

	int outSize = cuda_mj( d_Rin, rLen, d_Sin, sLen, &d_Joinout );

	//*h_Joinout = (Record*)malloc( sizeof(Record)*outSize );
	CPUMALLOC( (void**)h_Joinout, sizeof(Record)*outSize );

	FROMGPU( *h_Joinout, d_Joinout, sizeof(Record)*outSize );

	return outSize;
}


extern "C"
int GPUOnly_smj(Record *d_R, int rLen, Record *d_S, int sLen, Record** d_Joinout)
{
	return cuda_smj(d_R, rLen, d_S, sLen, d_Joinout);
}

extern "C"
int	GPUCopy_smj( Record* h_R, int rLen, Record* h_S, int sLen, Record** h_Joinout )
{
	Record* d_R;
	Record* d_S;
	Record* d_Joinout;

	GPUMALLOC( (void**)&d_R, sizeof(Record)*rLen );
	GPUMALLOC( (void**)&d_S, sizeof(Record)*sLen );
	TOGPU( d_R, h_R, sizeof(Record)*rLen );
	TOGPU( d_S, h_S, sizeof(Record)*sLen );

	int numResult = cuda_smj(d_R, rLen, d_S, sLen, &d_Joinout);

	//*h_Joinout = (Record*)malloc( sizeof(Record)*numResult );
	CPUMALLOC( (void**)h_Joinout, sizeof(Record)*numResult );

	FROMGPU( *h_Joinout, d_Joinout, sizeof(Record)*numResult );

	GPUFREE( d_R );
	GPUFREE( d_S );
	GPUFREE( d_Joinout );

	return numResult;
}


#endif
