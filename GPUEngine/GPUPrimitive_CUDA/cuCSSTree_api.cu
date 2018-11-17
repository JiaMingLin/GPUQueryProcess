#pragma once

// include header files
#include <cutil.h>
#include "cuCSSTree.h"

// include cu files
#include <scan.cu>
#include <cuCSSTree_kernel.cu>
#include <myMath.cu>
#include <myUtils.cu>

/*************************************************************************************************
	To allow more flexibility and efficiency (allowing saving transfers between host and device),
	Array params in the functions in this file are all on device unless otherwise specified.

	Users are expected to manage host/device transfers as appropriate for different purposes
**************************************************************************************************/



/**
 Indexing a sorted data array on device
  @param g_data the sorted data nodes on device global memory
  @param nDataNodes number of data nodes
  @param g_ptrDir pointer to starting pointer of the array which stores the Directory on the device
  @param ptrDirSize pointer to the variable that stores size of the directory, where the value will be returned
*/
void cuda_create_index(IDataNode g_data[], unsigned int nDataNodes, IDirectoryNode** g_ptrDir, unsigned int* ptrDirSize)
{
	//#region Calculate parameters on host
	unsigned int lvlDir = uintCeilingLog(TREE_FANOUT, nDataNodes);
	unsigned int nDirNodes = uintCeilingDiv(nDataNodes - 1, TREE_NODE_SIZE);
	unsigned int tree_size = nDirNodes + nDataNodes;
	unsigned int bottom_start = ( uintPower(TREE_FANOUT, lvlDir) - 1 ) / TREE_NODE_SIZE;
	//#endregion

	// Allocate space for directory on device
	GPUMALLOC( (void**) g_ptrDir, sizeof(IDirectoryNode) * nDirNodes ) ;

	unsigned int nNodesPerBlock = uintCeilingDiv(nDirNodes, BLCK_PER_GRID_create);

	//#region Execute on device
	dim3  Db(THRD_PER_BLCK_create, 1, 1);
	dim3  Dg(BLCK_PER_GRID_create, 1, 1);
	gCreateIndex <<<Dg, Db>>> (g_data, *g_ptrDir, nDirNodes, tree_size, bottom_start, nNodesPerBlock);
	//#endregion

	// Passback the size of the directory
	*ptrDirSize = nDirNodes;
}


/**
 Search a sorted array on device
  @param g_data the sorted data nodes
  @param nDataNodes number of data nodes in the tree
  @param g_dir the array of tree nodes which stores the Directory
  @param nDirNodes number of directory nodes in the tree
  @param g_keys array of searchKeys
  @param g_locations array where the returned node indices will be stored
  @param nSearchKeys number of search keys
*/
void cuda_search_index(IDataNode g_data[], unsigned int nDataNodes, IDirectoryNode g_dir[], unsigned int nDirNodes, Record g_keys[], int g_locations[], unsigned int nSearchKeys)
{
	unsigned int lvlDir = uintCeilingLog(TREE_FANOUT, nDataNodes);
	unsigned int tree_size = nDataNodes + nDirNodes;
	unsigned int bottom_start = ( uintPower(TREE_FANOUT, lvlDir) - 1 ) / TREE_NODE_SIZE;

	dim3 Db(THRD_PER_BLCK_search, 1, 1);
	dim3 Dg(BLCK_PER_GRID_search, 1, 1);

	unsigned int nKeysPerThread = uintCeilingDiv(nSearchKeys, THRD_PER_GRID_search);

#ifdef SHARED_MEM
	printf( "\nYES, SHARED MEMORY, gSearchTree\n" );
	#ifdef COALESCED
		printf( "\nYES, COALESCED, gSearchTree\n" );
		gSearchTree <<<Dg, Db>>> (g_data, nDataNodes, g_dir, nDirNodes, lvlDir,
			g_keys, g_locations, nSearchKeys, nKeysPerThread, tree_size, bottom_start);
	#else
		printf( "\NO COALESCED, gSearchTree\n" );
		gSearchTree_noCoalesced<<<Dg, Db>>> (g_data, nDataNodes, g_dir, nDirNodes, lvlDir,
			g_keys, g_locations, nSearchKeys, nKeysPerThread, tree_size, bottom_start);
	#endif
#else
	printf( "\nNO SHARED MEMORY, gSearchTree\n" );
	IKeyType* d_RootNodeKeys;
	GPUMALLOC( (void**)&d_RootNodeKeys, sizeof(IKeyType)*TREE_NODE_SIZE*Dg.x );
	gSearchTree_noShared<<<Dg, Db>>> (d_RootNodeKeys, 
		g_data, nDataNodes, g_dir, nDirNodes, lvlDir, g_keys, g_locations, nSearchKeys, nKeysPerThread, tree_size, bottom_start);
	GPUFREE( d_RootNodeKeys );
#endif
}




/**
 Search a sorted array on device
  @param g_data the sorted data nodes
  @param nDataNodes number of data nodes in the tree
  @param g_dir the array of tree nodes which stores the Directory
  @param nDirNodes number of directory nodes in the tree
  @param g_keys array of searchKeys
  @param g_locations array where the returned node indices will be stored
  @param nSearchKeys number of search keys
*/
void cuda_search_index_usingKeys(IDataNode g_data[], unsigned int nDataNodes, IDirectoryNode g_dir[], unsigned int nDirNodes, 
								 int g_keys[], int g_locations[], unsigned int nSearchKeys)
{
	unsigned int lvlDir = uintCeilingLog(TREE_FANOUT, nDataNodes);
	unsigned int tree_size = nDataNodes + nDirNodes;
	unsigned int bottom_start = ( uintPower(TREE_FANOUT, lvlDir) - 1 ) / TREE_NODE_SIZE;

	dim3 Db(THRD_PER_BLCK_search, 1, 1);
	dim3 Dg(BLCK_PER_GRID_search, 1, 1);

	unsigned int nKeysPerThread = uintCeilingDiv(nSearchKeys, THRD_PER_GRID_search);

	gSearchTree_usingKeys <<<Dg, Db>>> (g_data, nDataNodes, g_dir, nDirNodes, lvlDir, g_keys, g_locations, nSearchKeys, nKeysPerThread, tree_size, bottom_start);
}

/**
 Join the two arrays after having performed search on device
  @param g_R the R table (indexed), points to a location on device
  @param rLen number of tuples in the R table
  @param g_S the S table, points to a location on device
  @param g_locations array where the returned node indices are stored, points to a location on device
  @param sLen number of tuples in S (i.e. the same as # of elements in g_locations)

  @return # of join results
*/
int cuda_join_after_search(Record g_R[], int rLen, Record g_S[], int g_locations[], unsigned int sLen, 	Record** pG_Results)
{
	int clusterSize = uintCeilingDiv(sLen, THRD_PER_GRID_join);

	dim3 Dg;
	dim3 Db;

	Dg.x = BLCK_PER_GRID_join;
	Db.x = THRD_PER_BLCK_join;

	int* d_ResNums;

	GPUMALLOC((void**) &d_ResNums, sizeof(int) * THRD_PER_GRID_join);
#ifdef COALESCED
	printf( "YES, COALESCED, gIndexJoin\n" );
	gIndexJoin <<<Dg, Db>>> (g_R, rLen, g_S, g_locations, sLen, d_ResNums, clusterSize);
#else
	printf( "NO, COALESCED, gIndexJoin\n" );
	gIndexJoin_noCoalesced <<<Dg, Db>>> (g_R, rLen, g_S, g_locations, sLen, d_ResNums, clusterSize);

#endif

	int* d_sum;
	//saven_initialPrefixSum(THRD_PER_GRID_join);
	GPUMALLOC((void**) &d_sum, sizeof(int) * THRD_PER_GRID_join);
	//prescanArray( d_sum, d_ResNums, THRD_PER_GRID_join);
	scanImpl(d_ResNums, THRD_PER_GRID_join, d_sum);
	//deallocBlockSums();

	int sum = 0;
	int last;
    CUDA_SAFE_CALL(cudaMemcpy(&sum, d_sum + (THRD_PER_GRID_join - 1), sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&last, d_ResNums + (THRD_PER_GRID_join - 1), sizeof(int), cudaMemcpyDeviceToHost));
	sum += last;

	GPUMALLOC((void**) pG_Results, sizeof(Record) * sum);
#ifdef COALESCED
	printf( "YES, COALESCED, gJoinWithWrite\n" );
	gJoinWithWrite<<<Dg, Db>>> (g_R, rLen, g_S, g_locations, sLen, d_sum, *pG_Results, clusterSize);
#else
	printf( "NO, COALESCED, gJoinWithWrite\n" );
	gJoinWithWrite_noCoalesced<<<Dg, Db>>> (g_R, rLen, g_S, g_locations, sLen, d_sum, *pG_Results, clusterSize);
#endif
	return sum;
}


/**
 Perform Index Join on a pre-constructed index.
 This is equivalent to performing search directly followed by a post-search join


 @return # of join results
*/
int cuda_join(Record R[], unsigned int rLen, unsigned int nDataNodes, IDirectoryNode g_dir[], unsigned int nDirNodes, Record S[], unsigned int sLen)
{
	// Allocate Memory Space
	int*	d_locations;	// Location array on device
	GPUMALLOC((void**) &d_locations, sizeof(int) * sLen);

	// Search
	cuda_search_index((IDataNode*) R, nDataNodes, g_dir, nDirNodes, S, d_locations, sLen);

	// Post-Search Join
//	startMyUtilTimer(&tmr);
	Record * tmp;
	int result = cuda_join_after_search(R, rLen, S, d_locations, sLen, &tmp);
//	printf("Post Search Join completed in %f ms\n", endMyUtilTimer(tmr));

	// Free CUDA memory
	CUDA_SAFE_CALL(cudaFree(d_locations));
	CUDA_SAFE_CALL(cudaFree(tmp));

	return result;
}



/**
 Search a sorted array on device
  @param g_data the sorted data nodes
  @param nDataNodes number of data nodes in the tree
  @param g_dir the array of tree nodes which stores the Directory
  @param nDirNodes number of directory nodes in the tree
  @param g_keys array of searchKeys
  @param g_locations array where the returned node indices will be stored
  @param nSearchKeys number of search keys
*/
void cuda_search_index_k(IDataNode g_data[], unsigned int nDataNodes, IDirectoryNode g_dir[], unsigned int nDirNodes, int g_keys[], int g_locations[], unsigned int nSearchKeys)
{
	unsigned int lvlDir = uintCeilingLog(TREE_FANOUT, nDataNodes);
	unsigned int tree_size = nDataNodes + nDirNodes;
	unsigned int bottom_start = ( uintPower(TREE_FANOUT, lvlDir) - 1 ) / TREE_NODE_SIZE;

	dim3 Db(THRD_PER_BLCK_search, 1, 1);
	dim3 Dg(BLCK_PER_GRID_search, 1, 1);

	unsigned int nKeysPerThread = uintCeilingDiv(nSearchKeys, THRD_PER_GRID_search);

	gSearchTree_k <<<Dg, Db>>> (g_data, nDataNodes, g_dir, nDirNodes, lvlDir, g_keys, g_locations, nSearchKeys, nKeysPerThread, tree_size, bottom_start);
}

/**
 Join the two arrays after having performed search on device
  @param g_R the R table (indexed), points to a location on device
  @param rLen number of tuples in the R table
  @param g_S the S table, points to a location on device
  @param g_locations array where the returned node indices are stored, points to a location on device
  @param sLen number of tuples in S (i.e. the same as # of elements in g_locations)

  @return # of join results
*/
int cuda_join_after_search_k(Record g_R[], int rLen, int g_S[], int g_locations[], unsigned int sLen, Record** pG_result)
{
	int clusterSize = uintCeilingDiv(sLen, THRD_PER_GRID_join);

	dim3 Dg;
	dim3 Db;

	Dg.x = BLCK_PER_GRID_join;
	Db.x = THRD_PER_BLCK_join;

	int* d_ResNums;

	GPUMALLOC((void**) &d_ResNums, sizeof(int) * THRD_PER_GRID_join);
	gIndexJoin_k <<<Dg, Db>>> (g_R, rLen, g_S, g_locations, sLen, d_ResNums, clusterSize);

	int* d_sum;
	//saven_initialPrefixSum(THRD_PER_GRID_join);
	GPUMALLOC((void**) &d_sum, sizeof(int) * THRD_PER_GRID_join);
	//prescanArray( d_sum, d_ResNums, THRD_PER_GRID_join);
	scanImpl(d_ResNums, THRD_PER_GRID_join, d_sum);
	//deallocBlockSums();
	
	int sum = 0;
	int last;
    CUDA_SAFE_CALL(cudaMemcpy(&sum, d_sum + (THRD_PER_GRID_join - 1), sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&last, d_ResNums + (THRD_PER_GRID_join - 1), sizeof(int), cudaMemcpyDeviceToHost));
	sum += last;

	GPUMALLOC((void**) pG_result, sizeof(Record) * sum);
	gJoinWithWrite_k<<<Dg, Db>>> (g_R, rLen, g_S, g_locations, sLen, d_sum, *pG_result, clusterSize);

	return sum;
}


