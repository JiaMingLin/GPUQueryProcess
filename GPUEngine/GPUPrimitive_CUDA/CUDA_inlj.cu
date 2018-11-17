#ifndef CUDA_INLJ_CU
#define CUDA_INLJ_CU
#include "QP_Utility.cu"

//the indexed relation is R!!
int cuda_inlj(Record *R, int rLen, CUDA_CSSTree *d_tree, Record *d_S, int sLen, Record** d_Rout)
{
#ifdef BINARY_SEARCH
	printf("YES, BINARY_SERACH");
#else
	printf("NO, BINARY_SERACH");
#endif
	unsigned int timer=0;
	int*	d_locations;	// Location array on device
	GPUMALLOC((void**) &d_locations, sizeof(int) * sLen);	
	startTimer(&timer);
	cuda_search_index(d_tree->data, d_tree->nDataNodes, d_tree->dir, d_tree->nDirNodes, d_S, d_locations, sLen);
	endTimer("cuda_search_index", &timer);
	//Record* d_Result;
	startTimer(&timer);
	int result = cuda_join_after_search((Record*)d_tree->data, rLen, d_S, d_locations, sLen, d_Rout);
	endTimer("cuda_join_after_search", &timer);
	cudaThreadSynchronize();	
	
	/*startTimer(&timer);
	copyBackToHost(d_Result, (void**)Rout, sizeof(Record) * result, 1, 1);
	endTimer("copy back", &timer);*/
	
	GPUFREE(d_locations);
	return result;
}

extern "C"
int GPUOnly_inlj(Record* d_Rin, int rLen, CUDA_CSSTree* d_tree, Record* d_Sin, int sLen, Record** d_Rout )
{
	return cuda_inlj(d_Rin, rLen, d_tree, d_Sin, sLen, d_Rout);
}

extern "C"
int GPUCopy_inlj( Record* h_Rin, int rLen, CUDA_CSSTree* tree, Record* h_Sin, int sLen, Record** h_Rout )
{
	Record* d_Rin;
	Record* d_Sin;
	Record* d_Rout;

	GPUMALLOC( (void**)&d_Rin, sizeof(Record)*rLen );
	GPUMALLOC( (void**)&d_Sin, sizeof(Record)*sLen );

	TOGPU( d_Rin, h_Rin, sizeof(Record)*rLen );
	TOGPU( d_Sin, h_Sin, sizeof(Record)*sLen );

	int outSize = cuda_inlj(d_Rin, rLen, tree, d_Sin, sLen, &d_Rout);

	//*h_Rout = (Record*)malloc( sizeof(Record)*outSize );
	CPUMALLOC( (void**)h_Rout, sizeof(Record)*outSize );
	FROMGPU( *h_Rout, d_Rout, sizeof(Record)*outSize );

	GPUFREE( d_Rin );
	GPUFREE( d_Sin );
	GPUFREE( d_Rout );

	return outSize;
}

#endif
