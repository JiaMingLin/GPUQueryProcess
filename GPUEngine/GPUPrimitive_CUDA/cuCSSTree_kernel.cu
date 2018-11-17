#pragma once

#include "cuCSSTree.h"
#include "myMath.cu"
#include <math.h>

__device__ 
int getRightMostDescIdx(int tree_size, int nodeIdx)
{
	int tmp = nodeIdx * TREE_NODE_SIZE + TREE_FANOUT;
	int n = uintCeilingLog(TREE_FANOUT, uintCeilingDiv(TREE_NODE_SIZE * tree_size + TREE_FANOUT, tmp)) - 1;

	int result = (tmp * uintPower(TREE_FANOUT, n) - TREE_FANOUT) / TREE_NODE_SIZE;
    return result; 
}

__device__
int getDataArrayIdx(int dirSize, int tree_size, int bottom_start, int treeIdx)
{
	int idx;
	if(treeIdx < dirSize) {
		idx = tree_size - bottom_start - 1;
	}
	else if( treeIdx < bottom_start ) {
		idx = tree_size - bottom_start + treeIdx - dirSize;
	}
	else {
		idx = treeIdx - bottom_start;
	}
	return idx;
}
#ifdef BINARY_SEARCH
// Binary Search
__device__ 
int firstMatchingKeyInDirNode1(int keys[], int key)
{
	int min = 0;
	int max = TREE_NODE_SIZE;
	int mid;
	int cut;
	while(max - min > 1) {
		mid = (min + max) / 2;
		cut = keys[mid];

		if(key > cut)
			min = mid;
		else
			max = mid;
	}

	if(keys[min] >= key)
		return min;

	return max;

}

// Binary Search
__device__ 
int firstMatchingKeyInDataNode2(Record records[], IKeyType key)
{
	int min = 0;
	int max = TREE_NODE_SIZE;
	int mid;
	int cut;
	while(max - min > 1) {
		mid = (min + max) / 2;
		cut = records[mid].y;

		if(key > cut)
			min = mid;
		else
			max = mid;
	}

	if(records[min].y == key)
		return min;

	if(max < TREE_NODE_SIZE && records[max].y == key)
		return max;

	return -1;
}
#else//sequential search

__device__ 
int firstMatchingKeyInDirNode1(int keys[], int key)
{
	for(int i = 0; i < TREE_NODE_SIZE; i++) {
		if(keys[i] >= key)
			return i;
	}

	return TREE_NODE_SIZE;

}
//sequential search.
__device__ 
int firstMatchingKeyInDataNode2(Record records[], IKeyType key)
{
	for(int i = 0; i < TREE_NODE_SIZE; i++) {
		if(records[i].y == key)
			return i;
	}

	return -1;
}

#endif

//Scan Search
__device__ 
int firstMatchingKeyInDirNode(IKeyType keys[], IKeyType key)
{
	for(int i = 0; i < TREE_NODE_SIZE; i++) {
		if(keys[i] >= key)
			return i;
	}

	return TREE_NODE_SIZE;
}

// Binary Search
__device__ 
int firstMatchingKeyInDataNode(Record records[], IKeyType key)
{
	int min, max, mid, cut;

	if(records[TREE_NODE_SIZE - 1].y < key)
		min = TREE_NODE_SIZE;
	else if(records[TREE_NODE_SIZE - 2].y < key)
		min = TREE_NODE_SIZE - 1;
	else {
		min = 0;
		max = TREE_NODE_SIZE - 2;
		mid = TREE_NODE_SIZE/2 - 1;
		while(min != max) {
			cut = records[mid].y;
			if(key > cut)
			{
				min = mid+1;
			}
			else if(key > records[mid - 1].y)
			{
				min = mid;
				max = mid;
			}
			else
			{
				max = mid - 1;
			}
			mid = (min + max) / 2;
		}
	}

	return min;
}



__device__ 
int firstMatchingKeyInDataNode_saven(Record records[], IKeyType key)
{
	int min = 0;
	int max = TREE_NODE_SIZE;
	int mid;
	int cut;
	while(max - min > 1) {
		mid = (min + max) / 2;
		cut = records[mid].y;

		if(key > cut)
			min = mid;
		else
			max = mid;
	}

	if(records[min].y >= key)
		return min;

	return max;
}




/**
 Kernel function for creating the index
  @param data the data nodes
  @param dSize # of data nodes
  @param dir the directory node array to be populated
  @param dirSize the pre-computed dir size
  @param tree_size the pre-computed tree size
  @param bottom_start pre-computed parameter (for details, refer to the CSS tree concepts)
*/
__global__
void gCreateIndex(IDataNode data[], IDirectoryNode dir[], int dirSize, int tree_size, int bottom_start, int nNodesPerBlock)
{
		//int startIdx = blockIdx.x * nNodesPerBlock;
  //      int endIdx = startIdx + nNodesPerBlock;
  //      if(endIdx > dirSize)
  //              endIdx = dirSize;
  //      int keyIdx = threadIdx.x;
		//
		//dir[0].keys[0] = startIdx + keyIdx;

        int startIdx = blockIdx.x * nNodesPerBlock;
        int endIdx = startIdx + nNodesPerBlock;
        if(endIdx > dirSize)
                endIdx = dirSize;
        int keyIdx = threadIdx.x;

        // Proceed only when in internal nodes
        for(int nodeIdx = startIdx; nodeIdx < endIdx; nodeIdx++)
        {
                int childIdx = nodeIdx * TREE_FANOUT + keyIdx + 1;        // One step down to the left
                // Then look for the right most descendent
                int rightMostDesIdx;
                // Common cases
                if(childIdx < tree_size) {
                        rightMostDesIdx = getRightMostDescIdx(tree_size, childIdx);
                }
                // versus the unusual case when the tree is incomplete and the node does not have the full set of children
                else {
                        // pick the last node in the tree (largest element of the array)
                        rightMostDesIdx = tree_size - 1;
                }

                int dataArrayIdx = getDataArrayIdx(dirSize, tree_size, bottom_start, rightMostDesIdx);

                dir[nodeIdx].keys[keyIdx] = data[dataArrayIdx].records[TREE_NODE_SIZE - 1].y;
        }
}


/**
 Kernel function for searching the index
  @param data the data nodes
  @param nDataNodes # of data nodes
  @param dir the directory node array to be populated
  @param nDirNodes the pre-computed dir size
  @param lvlDir # of levels in the directory
  @param arr The array of records to be searched
  @param locations pointer to the [output] location (index in the data node array) array 
  @param nSearchKeys # of search keys
  @param nKeysPerThread # of search keys to be processed by each thread
  @param nSearchKeys # of search keys
  @param tree_size the pre-computed tree size
  @param bottom_start pre-computed parameter (for details, refer to the CSS tree concepts)
*/
#ifndef SHARED_MEM
	__global__
	void gSearchTree_noShared(IKeyType* d_RootNodeKeys,
	IDataNode* data, int nDataNodes, IDirectoryNode* dir,
	int nDirNodes, int lvlDir, Record* arr, int locations[], int nSearchKeys, int nKeysPerThread, int tree_size, int bottom_start)
	{
		// Bringing the root node (visited by every tuple) to the faster shared memory
		//__shared__ IKeyType RootNodeKeys[TREE_NODE_SIZE];
		IKeyType* RootNodeKeys = d_RootNodeKeys + blockIdx.x*TREE_NODE_SIZE;
		RootNodeKeys[threadIdx.x] = dir->keys[threadIdx.x];

		__syncthreads();

		int OverallThreadIdx = blockIdx.x * THRD_PER_BLCK_search + threadIdx.x;

		for(int keyIdx = OverallThreadIdx; keyIdx < nSearchKeys; keyIdx += THRD_PER_GRID_search)
		{
			IKeyType val = arr[keyIdx].y;
			int loc = firstMatchingKeyInDirNode1(RootNodeKeys, val) + 1;
			for(int i = 1; i < lvlDir && loc < nDirNodes; i++) {
				int kid = firstMatchingKeyInDirNode1(dir[loc].keys, val);
				loc = loc * TREE_FANOUT + kid + 1;
			}

			if(loc >= tree_size)
				loc = nDataNodes - 1;
			else
				loc = getDataArrayIdx(nDirNodes, tree_size, bottom_start, loc);

			int offset = firstMatchingKeyInDataNode2(data[loc].records, val);
			locations[keyIdx] = (offset <0)?-1:(loc * TREE_NODE_SIZE + offset);
		}
	}
#endif

#ifndef COALESCED
	__global__
	void gSearchTree_noCoalesced(IDataNode* data, int nDataNodes, IDirectoryNode* dir, 
	int nDirNodes, int lvlDir, Record* arr, int locations[], int nSearchKeys, int nKeysPerThread, int tree_size, int bottom_start)
	{
		// Bringing the root node (visited by every tuple) to the faster shared memory
		__shared__ IKeyType RootNodeKeys[TREE_NODE_SIZE];
		RootNodeKeys[threadIdx.x] = dir->keys[threadIdx.x];

		__syncthreads();

		int bx = blockIdx.x;
		int tx = threadIdx.x;	
		int numThread = blockDim.x;
		int numBlock = gridDim.x;
		int totalTx = bx*THRD_PER_BLCK_search + tx;
		int threadSize = nSearchKeys/( numThread*numBlock );
		int start = totalTx*threadSize;
		int end = start + threadSize;

		int OverallThreadIdx = blockIdx.x * THRD_PER_BLCK_search + threadIdx.x;

		//for(int keyIdx = OverallThreadIdx; keyIdx < nSearchKeys; keyIdx += THRD_PER_GRID_search)
		for( int keyIdx = start; keyIdx < end; keyIdx++ )
		{
			IKeyType val = arr[keyIdx].y;
			int loc = firstMatchingKeyInDirNode1(RootNodeKeys, val) + 1;
			for(int i = 1; i < lvlDir && loc < nDirNodes; i++) {
				int kid = firstMatchingKeyInDirNode1(dir[loc].keys, val);
				loc = loc * TREE_FANOUT + kid + 1;
			}

			if(loc >= tree_size)
				loc = nDataNodes - 1;
			else
				loc = getDataArrayIdx(nDirNodes, tree_size, bottom_start, loc);

			int offset = firstMatchingKeyInDataNode2(data[loc].records, val);
			locations[keyIdx] = (offset <0)?-1:(loc * TREE_NODE_SIZE + offset);
		}
	}
#endif

__global__
void gSearchTree(IDataNode* data, int nDataNodes, IDirectoryNode* dir, int nDirNodes, int lvlDir, Record* arr, int locations[], int nSearchKeys, int nKeysPerThread, int tree_size, int bottom_start)
{
	// Bringing the root node (visited by every tuple) to the faster shared memory
	__shared__ IKeyType RootNodeKeys[TREE_NODE_SIZE];
	RootNodeKeys[threadIdx.x] = dir->keys[threadIdx.x];

	__syncthreads();

	int OverallThreadIdx = blockIdx.x * THRD_PER_BLCK_search + threadIdx.x;

	for(int keyIdx = OverallThreadIdx; keyIdx < nSearchKeys; keyIdx += THRD_PER_GRID_search)
	{
		IKeyType val = arr[keyIdx].y;
		int loc = firstMatchingKeyInDirNode1(RootNodeKeys, val) + 1;
		for(int i = 1; i < lvlDir && loc < nDirNodes; i++) {
			int kid = firstMatchingKeyInDirNode1(dir[loc].keys, val);
			loc = loc * TREE_FANOUT + kid + 1;
		}

		if(loc >= tree_size)
			loc = nDataNodes - 1;
		else
			loc = getDataArrayIdx(nDirNodes, tree_size, bottom_start, loc);

		int offset = firstMatchingKeyInDataNode2(data[loc].records, val);
		locations[keyIdx] = (offset <0)?-1:(loc * TREE_NODE_SIZE + offset);
	}
}


__global__
void gSearchTree_usingKeys(IDataNode* data, int nDataNodes, IDirectoryNode* dir, int nDirNodes, int lvlDir, 
						   int* arr, int locations[], int nSearchKeys, int nKeysPerThread, 
						   int tree_size, int bottom_start)
{
	// Bringing the root node (visited by every tuple) to the faster shared memory
	__shared__ IKeyType RootNodeKeys[TREE_NODE_SIZE];
	RootNodeKeys[threadIdx.x] = dir->keys[threadIdx.x];

	__syncthreads();

	int OverallThreadIdx = blockIdx.x * THRD_PER_BLCK_search + threadIdx.x;

	for(int keyIdx = OverallThreadIdx; keyIdx < nSearchKeys; keyIdx += THRD_PER_GRID_search)
	{
		IKeyType val = arr[keyIdx];
		int loc = firstMatchingKeyInDirNode1(RootNodeKeys, val) + 1;
		for(int i = 1; i < lvlDir && loc < nDirNodes; i++) {
			int kid = firstMatchingKeyInDirNode1(dir[loc].keys, val);
			loc = loc * TREE_FANOUT + kid + 1;
		}

		if(loc >= tree_size)
			loc = nDataNodes - 1;
		else
			loc = getDataArrayIdx(nDirNodes, tree_size, bottom_start, loc);

		int offset = firstMatchingKeyInDataNode_saven(data[loc].records, val);
		locations[keyIdx] = loc * TREE_NODE_SIZE + offset;
	}
}


__global__
void gIndexJoin(Record R[], int rLen, Record S[], int g_locations[], int sLen, int g_ResNums[], int clusterSize)
{
	int cluster_id = blockIdx.x * THRD_PER_BLCK_join + threadIdx.x;

	if(cluster_id >= sLen) {
		g_ResNums[cluster_id] = 0;
		return;
	}

	int count = 0;

	int s_cur = cluster_id;
	int r_cur;
	int s_key;

	// Outputing to result
	while(s_cur < sLen) {
		r_cur = g_locations[s_cur];
		if(r_cur >= 0 && r_cur < rLen) {
			s_key = S[s_cur].y;
			while(s_key == R[r_cur].y) {
				count++;
				r_cur++;
			}
		}

		s_cur += THRD_PER_GRID_join;
	}

	g_ResNums[cluster_id] = count;
}

#ifndef COALESCED
	__global__
	void gIndexJoin_noCoalesced(Record R[], int rLen, Record S[], int g_locations[], int sLen, int g_ResNums[], int clusterSize)
	{
		int cluster_id = blockIdx.x * THRD_PER_BLCK_join + threadIdx.x;

		if(cluster_id >= sLen) {
			g_ResNums[cluster_id] = 0;
			return;
		}

		int count = 0;

		//int s_cur = cluster_id;
		int r_cur;
		int s_key;

		int bx = blockIdx.x;
		int tx = threadIdx.x;	
		int numThread = blockDim.x;
		int numBlock = gridDim.x;
		int totalTx = bx*numThread + tx;
		int threadSize = sLen/( numThread*numBlock );
		int start = totalTx*threadSize;
		int end = start + threadSize;
		if( (threadIdx.x == (blockDim.x - 1)) && (blockIdx.x == (gridDim.x - 1)) )
		{
			end = sLen;
		}

		// Outputing to result
		//for( int s_cur = cluster_id; s_cur < sLen; s_cur += THRD_PER_GRID_join )
		for( int s_cur = start; s_cur < end; s_cur++ )
		{
			r_cur = g_locations[s_cur];
			if(r_cur >= 0 && r_cur < rLen) {
				s_key = S[s_cur].y;
				while(s_key == R[r_cur].y) {
					count++;
					r_cur++;
				}
			}
		}

		g_ResNums[cluster_id] = count;
	}
#endif

//best, with shared memory, with coalesced access
__global__
void gJoinWithWrite(Record R[], int rLen, Record S[], int g_locations[], int sLen, int g_PrefixSums[], Record g_joinResultBuffers[], int clusterSize)
{
	int cluster_id = blockIdx.x * THRD_PER_BLCK_join + threadIdx.x;

	if(cluster_id >= sLen) {
		return;
	}

	int s_cur = cluster_id;
	int r_cur;
	int s_key;

	Record* pen = g_joinResultBuffers + (g_PrefixSums[cluster_id]);

	// Outputing to result
	while(s_cur < sLen) {
		r_cur = g_locations[s_cur];
		if(r_cur >= 0 && r_cur < rLen) {
			s_key = S[s_cur].y;
			while(s_key == R[r_cur].y) {
				pen ->x = R[r_cur].x;
				pen -> y = S[s_cur].x;
				pen++;
				r_cur++;
			}
		}

		s_cur += THRD_PER_GRID_join;
	}
}

#ifndef COALESCED
	__global__
	void gJoinWithWrite_noCoalesced(Record R[], int rLen, Record S[], int g_locations[],
		int sLen, int g_PrefixSums[], Record g_joinResultBuffers[], int clusterSize)
	{
		int cluster_id = blockIdx.x * THRD_PER_BLCK_join + threadIdx.x;

		if(cluster_id >= sLen) {
			return;
		}

		int s_cur = cluster_id;
		int r_cur;
		int s_key;

		Record* pen = g_joinResultBuffers + (g_PrefixSums[cluster_id]);

		int bx = blockIdx.x;
		int tx = threadIdx.x;	
		int numThread = blockDim.x;
		int numBlock = gridDim.x;
		int totalTx = bx*numThread + tx;
		int threadSize = sLen/( numThread*numBlock );
		int start = totalTx*threadSize;
		int end = start + threadSize;
		if( (threadIdx.x == (blockDim.x - 1)) && (blockIdx.x == (gridDim.x - 1)) )
		{
			end = sLen;
		}

		// Outputing to result
		for( int s_cur = start; s_cur < end; s_cur++ )
		{
			r_cur = g_locations[s_cur];
			if(r_cur >= 0 && r_cur < rLen) {
				s_key = S[s_cur].y;
				while(s_key == R[r_cur].y) {
					pen ->x = R[r_cur].x;
					pen -> y = S[s_cur].x;
					pen++;
					r_cur++;
				}
			}
		}
	}
#endif


__global__
void gTestResult(Record R[], int rLen, Record S[], int g_locations[], int sLen, int* g_fail, int nKeysPerThread)
{
	int OverallThreadIdx = blockIdx.x * THRD_PER_BLCK_search + threadIdx.x;

	int startIdx = OverallThreadIdx;

	for(int keyIdx = startIdx; keyIdx < sLen; keyIdx += THRD_PER_GRID_search)
	{
		// R
		int match = 1;
		int loc = g_locations[keyIdx];
		if(loc < 0) {
			match = 0;
			loc = ~loc;
		}
		int ridx = loc;

		if(match) {
			if(R[ridx].y != S[keyIdx].y)
				*g_fail = keyIdx + 1;
		}
		else {
			if(ridx >= rLen) {
				if(R[rLen-1].y >= S[keyIdx].y)
					*g_fail = keyIdx + 1;
			}
			else if(R[ridx].y <= S[keyIdx].y)
				*g_fail = keyIdx + 1;
			else if(ridx) {
				if(R[ridx-1].y >= S[keyIdx].y)
						*g_fail = keyIdx + 1;
			}
		}
	}
}

__global__
void gSearchTree_k(IDataNode* data, int nDataNodes, IDirectoryNode* dir, int nDirNodes, int lvlDir, int* arr, int locations[], int nSearchKeys, int nKeysPerThread, int tree_size, int bottom_start)
{
	// Bringing the root node (visited by every tuple) to the faster shared memory
	__shared__ IKeyType RootNodeKeys[TREE_NODE_SIZE];
	RootNodeKeys[threadIdx.x] = dir->keys[threadIdx.x];
	__syncthreads();

	int OverallThreadIdx = blockIdx.x * THRD_PER_BLCK_search + threadIdx.x;

	for(int keyIdx = OverallThreadIdx; keyIdx < nSearchKeys; keyIdx += THRD_PER_GRID_search)
	{
		IKeyType val = arr[keyIdx];
		int loc = firstMatchingKeyInDirNode1(RootNodeKeys, val) + 1;
		for(int i = 1; i < lvlDir && loc < nDirNodes; i++) {
			int kid = firstMatchingKeyInDirNode1(dir[loc].keys, val);
			loc = loc * TREE_FANOUT + kid + 1;
		}

		if(loc >= tree_size)
			loc = nDataNodes - 1;
		else
			loc = getDataArrayIdx(nDirNodes, tree_size, bottom_start, loc);

		int offset = firstMatchingKeyInDataNode2(data[loc].records, val);
		locations[keyIdx] = (offset <0)?-1:(loc * TREE_NODE_SIZE + offset);
	}
}


__global__
void gIndexJoin_k(Record R[], int rLen, int S[], int g_locations[], int sLen, int g_ResNums[], int clusterSize)
{
	int cluster_id = blockIdx.x * THRD_PER_BLCK_join + threadIdx.x;

	if(cluster_id >= sLen) {
		g_ResNums[cluster_id] = 0;
		return;
	}

	int count = 0;

	int s_cur = cluster_id;
	int r_cur;
	int s_key;
	while(s_cur < sLen) {
		r_cur = g_locations[s_cur];
		if(r_cur >= 0 && r_cur < rLen) {
			s_key = S[s_cur];
			while(s_key == R[r_cur].y) {
				count++;
				r_cur++;
			}
		}

		s_cur += THRD_PER_GRID_join;
	}

	g_ResNums[cluster_id] = count;

}

__global__
void gJoinWithWrite_k(Record R[], int rLen, int S[], int g_locations[], int sLen, int g_PrefixSums[], Record g_joinResultBuffers[], int clusterSize)
{
	int cluster_id = blockIdx.x * THRD_PER_BLCK_join + threadIdx.x;

	if(cluster_id >= sLen) {
		return;
	}

	int s_cur = cluster_id;
	int r_cur;
	int s_key;

	Record* pen = g_joinResultBuffers + (g_PrefixSums[cluster_id]);

	// Outputing to result
	while(s_cur < sLen) {
		r_cur = g_locations[s_cur];
		if(r_cur >= 0 && r_cur < rLen) {
			s_key = S[s_cur];
			while(s_key == R[r_cur].y) {
				pen ->x = R[r_cur].x;
				pen++;
				r_cur++;
			}
		}

		s_cur += THRD_PER_GRID_join;
	}
}



