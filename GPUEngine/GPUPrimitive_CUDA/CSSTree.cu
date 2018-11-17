#ifndef CSSTREE_H
#define CSSTREE_H

#include "assert.h"
#include "QP_Utility.cu"
#include "math.h"
#include "stdlib.h"

#include <cuCSSTree_api.cu>

#include "GPU_Dll.h"

void init_tree(Record *R, int rLen, int f, CUDA_CSSTree* h_me)
{
	IDataNode* d_data;
	IDirectoryNode* d_dir;
	unsigned int nDataNodes;
	unsigned int nDirNodes;

	nDataNodes = uintCeilingDiv(rLen, TREE_NODE_SIZE);
	unsigned int rSize = sizeof(Record) * rLen;
	unsigned int dSize = sizeof(IDataNode) * nDataNodes;

    GPUMALLOC((void**)&d_data, dSize);
    CUDA_SAFE_CALL(cudaMemset(d_data, 0x7f, dSize));
    TOGPU(d_data, R, rSize);

	cuda_create_index(d_data, nDataNodes, &d_dir, &nDirNodes);

	h_me->data = d_data;
	h_me->nDataNodes = nDataNodes;
	h_me->dir = d_dir;
	h_me->nDirNodes = nDirNodes;

}

void destroy_tree( CUDA_CSSTree* tree )
{
	CUDA_SAFE_CALL( cudaFree( tree->data ) );
	CUDA_SAFE_CALL( cudaFree( tree->dir ) );
}


void gpu_init_tree(Record *d_R, int rLen, int f, CUDA_CSSTree* h_me)
{
	IDataNode* d_data;
	IDirectoryNode* d_dir;
	unsigned int nDataNodes;
	unsigned int nDirNodes;

	nDataNodes = uintCeilingDiv(rLen, TREE_NODE_SIZE);
	unsigned int rSize = sizeof(Record) * rLen;
	unsigned int dSize = sizeof(IDataNode) * nDataNodes;
	if(rSize!=dSize)
	{
		GPUMALLOC((void**)&d_data, dSize);
		CUDA_SAFE_CALL(cudaMemset(d_data, 0x7f, dSize));
		GPUTOGPU(d_data, d_R, rSize);
		printf("!!! allocating % byte for new data array~", dSize);
		GPUFREE(d_R);
	}
	else
		d_data=(IDataNode *)d_R;

	cuda_create_index(d_data, nDataNodes, &d_dir, &nDirNodes);

	h_me->data = d_data;
	h_me->nDataNodes = nDataNodes;
	h_me->dir = d_dir;
	h_me->nDirNodes = nDirNodes;

}

int cuda_constructCSSTree(Record *Rin, int rLen, CUDA_CSSTree **tree)
{
	//__DEBUG__("cuda_constructCSSTree");

	*tree = (CUDA_CSSTree*)malloc(sizeof(CUDA_CSSTree));
	init_tree(Rin, rLen, 0, *tree);

	return 0;
}

//the data is on GPU
int gpu_constructCSSTree(Record *d_Rin, int rLen, CUDA_CSSTree **h_tree)
{
	//__DEBUG__("cuda_constructCSSTree");

	*h_tree = (CUDA_CSSTree*)malloc(sizeof(CUDA_CSSTree));
	gpu_init_tree(d_Rin, rLen, 0, *h_tree);

	return 0;
}

/*
@description: multiple equi-searches on the CSSTree.
@return: the number of matching records.
*/

int cuda_multi_equiTreeSearch(Record *Rin, int rLen, CUDA_CSSTree *tree, int* searchKeys, int nKey, Record** Rout)
{
	//__DEBUG__("cuda_multi_equiTreeSearch");

	int*	d_locations;	// Location array on device
	GPUMALLOC((void**) &d_locations, sizeof(int) * nKey);
	int*	d_keys;	// Location array on device
	GPUMALLOC((void**)&d_keys, sizeof(int) * nKey);
    CUDA_SAFE_CALL(cudaMemcpy(d_keys, searchKeys, sizeof(int)*nKey, cudaMemcpyHostToDevice) );
	cuda_search_index_k(tree->data, tree->nDataNodes, tree->dir, tree->nDirNodes, d_keys, d_locations, nKey);

	Record* d_Result;

	int result = cuda_join_after_search_k((Record*)tree->data, tree->nDataNodes, d_keys, d_locations, nKey, &d_Result);
	copyBackToHost(d_Result, (void**)Rout, sizeof(Record) * result, 1, 1);
	CUDA_SAFE_CALL(cudaFree(d_locations));
	CUDA_SAFE_CALL(cudaFree(d_keys));

	return result;
}

/*
@description: an equi-search on the CSSTree.
@return: the number of matching records.
*/
int cuda_equiTreeSearch(Record *Rin, int rLen, CUDA_CSSTree *tree, int keyForSearch, Record** Rout)
{
	//__DEBUG__("cuda_equiTreeSearch");
	return cuda_multi_equiTreeSearch(Rin, rLen, tree, &keyForSearch, 1, Rout);
}


extern "C"
int GPUCopy_BuildTreeIndex( Record* h_Rin, int rLen, CUDA_CSSTree **tree )
{
	int outSize = cuda_constructCSSTree(h_Rin, rLen, tree);

	return outSize;
}

extern "C"
int GPUOnly_BuildTreeIndex( Record* d_Rin, int rLen, CUDA_CSSTree** d_tree )
{
	return gpu_constructCSSTree(d_Rin, rLen, d_tree);
}



#endif

