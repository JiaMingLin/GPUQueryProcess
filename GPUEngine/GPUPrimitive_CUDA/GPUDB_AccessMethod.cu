#ifndef GPUDB_ACCESS_METHOD
#define GPUDB_ACCESS_METHOD

#include "hashtable.cu"
#include "GPU_Dll.h"

/*
This file implements the tree searches and the hash probes.
*/

int tree_search(Record *d_R, int rLen, CUDA_CSSTree *tree, Record *d_S, int sLen, Record** d_Rout)
{
	unsigned int timer=0;
	int*	d_locations;	// Location array on device
	GPUMALLOC((void**) &d_locations, sizeof(int) * sLen);	
	startTimer(&timer);
	cuda_search_index(tree->data, tree->nDataNodes, tree->dir, tree->nDirNodes, d_S, d_locations, sLen);
	endTimer("cuda_search_index_usingKeys", &timer);
	//Record* d_Result;
	startTimer(&timer);
	int result = cuda_join_after_search((Record*)tree->data, rLen, d_S, d_locations, sLen, d_Rout);
	endTimer("cuda_join_after_search", &timer);
	cudaThreadSynchronize();	
	
	
	
	/*startTimer(&timer);
	copyBackToHost(d_Result, (void**)Rout, sizeof(Record) * result, 1, 1);
	endTimer("copy back", &timer);*/
	GPUFREE(d_locations);
	return result;
}


extern "C"
int GPUOnly_TreeSearch( Record* d_Rin, int rLen, CUDA_CSSTree* tree, Record* d_Sin, int sLen, Record** d_Rout )
{
	return tree_search(d_Rin, rLen, tree, d_Sin, sLen, d_Rout);
}

extern "C"
int GPUCopy_TreeSearch( Record* h_Rin, int rLen, CUDA_CSSTree* tree, Record* h_Sin, int sLen, Record** h_Rout )
{
	Record* d_Rin;
	Record* d_Sin;
	Record* d_Rout;

	GPUMALLOC( (void**)&d_Rin, sizeof(Record)*rLen );
	GPUMALLOC( (void**)&d_Sin, sizeof(Record)*sLen );

	TOGPU( d_Rin, h_Rin, sizeof(Record)*rLen );
	TOGPU( d_Sin, h_Sin, sizeof(Record)*sLen );

	int outSize = tree_search(d_Rin, rLen, tree, d_Sin, sLen, &d_Rout);

	//*h_Rout = (Record*)malloc( sizeof(Record)*outSize );
	CPUMALLOC( (void**)&(*h_Rout), sizeof(Record)*outSize );
	FROMGPU( *h_Rout, d_Rout, sizeof(Record)*outSize );

	GPUFREE( d_Rin );
	GPUFREE( d_Sin );
	GPUFREE( d_Rout );

	return outSize;
}

int hashSearch(Record *d_R, int rLen, Bound* d_bound, int* d_keys, int sLen, 
			   Record** d_result, int numThreadsPerBlock = 512)
{
	unsigned int timer=0;
	int *d_oSize;
	GPUMALLOC( (void**) & d_oSize, sLen*sizeof(int) ); 
	Bound *d_oBound;
	GPUMALLOC( (void**) & d_oBound, sLen*sizeof(Record) ); 
	int *d_sum;
	GPUMALLOC( (void**) & d_sum, sLen*sizeof(int) ); 

	//int numThreadsPerBlock=512;//also the block size
	int blockSize=numThreadsPerBlock*2;
	int numBlock_x=sLen/blockSize;
	if(rLen%blockSize!=0)
		numBlock_x++;
	dim3  thread( numThreadsPerBlock, 1, 1);
	dim3  grid( numBlock_x, 1 , 1);
	int from, to;
	int numRun=16;//16 is the best.
	int runSize=rLen/numRun;
	printf("sLen, %d, numRun, %d\n", sLen, numRun);
	startTimer(&timer);
	for(int i=0;i<numRun;i++)
	{
		from=i*runSize;
		to=from + runSize;
		optProbe_kernel<<<grid, thread>>>(d_bound,rLen,d_keys, d_oSize, d_oBound, blockSize, sLen, from, to);
	}
	endTimer("probe",&timer);
	
	startTimer(&timer);
	//saven_initialPrefixSum(sLen);
	scanImpl(d_oSize,sLen,d_sum);
	int *h_last;
	CPUMALLOC((void**) &h_last, sizeof(int));
	FROMGPU(h_last, (d_oSize+sLen-1), sizeof(int));
	int *h_lastSum;
	CPUMALLOC((void**) &h_lastSum, sizeof(int));
	FROMGPU(h_lastSum, (d_sum+sLen-1), sizeof(int));

	int total=*h_last+*h_lastSum;
	int *d_loc;
	GPUMALLOC( (void**) & d_loc, total*sizeof(int) ); 

	GPUMALLOC( (void**) &(*d_result), total*sizeof(Record) ); 
	numRun=16;
	printf("total result, %d, numRun II, %d\n", total, numRun);
	thread.x=256;
	grid.x=256;
	grid.y=sLen/grid.x/thread.x;
	location_kernel<<<grid, thread>>>(d_loc, d_sum, d_oBound);
	endTimer("locations",&timer);
	
	startTimer(&timer);
	numThreadsPerBlock=512;//also the block size
	thread.x=numThreadsPerBlock;
	blockSize=numThreadsPerBlock*2;
	grid.x=128;
	grid.y=total/grid.x/numThreadsPerBlock;
	numRun=8;//16 is the best.
	runSize=rLen/numRun;
	printf("total II, %d, numRun, %d, grid.y, %d\n", total, numRun,grid.y);
	for(int i=0;i<numRun;i++)
	{
		from=i*runSize;
		to=from + runSize;
		optFetch_kernel<<<grid, thread>>>(d_R, *d_result,d_loc, blockSize, total, from, to);
	}
	endTimer("fetch",&timer);

	GPUFREE(d_loc);
	GPUFREE(d_oSize);
	GPUFREE(d_oBound);
	GPUFREE(d_sum);
	return total;
}

extern "C"
int GPUOnly_HashSearch( Record* d_R, int rLen, Bound* d_bound, int* d_keys, int sLen, Record** d_result, int numThread )
{
	return hashSearch(d_R, rLen, d_bound, d_keys, sLen, d_result, numThread);
}

extern "C"
int GPUCopy_HashSearch( Record* h_R, int rLen, Bound* h_bound, int inBits, Record* h_S, int sLen, Record** h_Rout, int numThread )
{
	int* h_keys = (int*)malloc( sizeof(int)*sLen );
	for( int i = 0; i < sLen; i++ )
	{
		h_keys[i] = h_S[i].y;
	}
	
	int boundLen = TwoPowerN(inBits);

	Record* d_R;
	Bound* d_bound;
	int* d_keys;
	Record* d_Rout;

	GPUMALLOC( (void**)&d_R, sizeof(Record)*rLen );
	GPUMALLOC( (void**)&d_bound, sizeof(Bound)*boundLen );
	GPUMALLOC( (void**)&d_keys, sizeof(int)*sLen );
	
	TOGPU( d_R, h_R, sizeof(Record)*rLen );
	TOGPU( d_bound, h_bound, sizeof(Bound)*boundLen );
	TOGPU( d_keys, h_keys, sizeof(int)*sLen );

	int outSize = hashSearch(d_R, rLen, d_bound, d_keys, sLen, &d_Rout, numThread);

	//*h_Rout = (Record*)malloc( sizeof(Record)*outSize );
	CPUMALLOC( (void**)&(*h_Rout), sizeof(Record)*outSize );
	FROMGPU( *h_Rout, d_Rout, sizeof(Record)*outSize );

	GPUFREE( d_R );
	GPUFREE( d_bound );
	GPUFREE( d_keys );
	GPUFREE( d_Rout );

	return outSize;
}

//write your testing code here.
void testTreeSearch(int rLen, int sLen)
{
	int result=0;
	int memSizeR=sizeof(Record)*rLen;
	int memSizeS=sizeof(Record)*sLen;
	Record *h_R;
	CPUMALLOC((void**)&h_R, memSizeR);
	generateSort(h_R, TEST_MAX,rLen,0);
	CUDA_CSSTree* tree;
	
	unsigned int timer=0;
	Record *h_S;
	CPUMALLOC((void**)&h_S, memSizeS);
	generateRand(h_S, TEST_MAX,sLen,1);
	
	Record *d_Rout;
	
	startTime();
	startTimer(&timer);
	Record *d_R;
	GPUMALLOC((void**) & d_R, memSizeR);
	TOGPU(d_R, h_R,	memSizeR);
	endTimer("copy R to GPU",&timer);
	
	startTimer(&timer);
	gpu_constructCSSTree(d_R, rLen, &tree);
	endTimer("tree construction", &timer);

	startTimer(&timer);
	Record *d_S;
	GPUMALLOC((void**) & d_S, sLen*sizeof(Record) );
	TOGPU(d_S, h_S,	memSizeS);
	endTimer("copy to GPU",&timer);

	
	//ninlj
	startTimer(&timer);
	result=tree_search(d_R,rLen,tree,d_S,sLen,&d_Rout);
	double processingTime=endTimer("tree search",&timer);

	startTimer(&timer);
	Record *h_result;
	CPUMALLOC((void**)&h_result, sizeof(Record)*result);
	FROMGPU(h_result, d_Rout, sizeof(Record)*result);
	endTimer("copy back", &timer);

	
	double sec=endTime("tree search");
	//gpuPrint(d_Rout, rLen, "d_Rout");
	printf("rLen, %d, sLen, %d, result, %d\n", rLen, sLen, result);	

	CPUFREE(d_Rout);
	CPUFREE(h_R);
	CPUFREE(h_S);
	GPUFREE(d_R);
	GPUFREE(d_S);
}

void testHashSearch(int rLen, int sLen)
{
	Record* h_R=NULL;
	CPUMALLOC((void**)&h_R, sizeof(Record)*rLen);
	generateSort(h_R, TEST_MAX, rLen, 0);
	Bound *h_bound=NULL;	
	double bits=log2((double)rLen);
	int intBits=(int)bits;
	if(bits-intBits>=0.0000001)
		intBits++;
	intBits=intBits-1;//each bucket 8 tuples.
	int listLen=(1<<intBits);
	CPUMALLOC((void**)&h_bound, sizeof(Record)*listLen);
	buildHashTable(h_R, rLen,intBits,h_bound);
	Record *h_S;
	CPUMALLOC((void**)&h_S, sizeof(Record)*sLen);
	generateRand(h_S, TEST_MAX,sLen,1);
	//extract the keys.
	int* h_SKeys;
	CPUMALLOC((void**)&h_SKeys, sLen*sizeof(int));	
	int i=0;
	for(i=0;i<sLen;i++)
	{
		h_SKeys[i]=h_S[i].y;
	}
	CPUFREE(h_S);
	//device memory
	unsigned int timer=0;
	startTime();
	startTimer(&timer);
	Record **h_Rout;
	CPUMALLOC((void**)&h_Rout,sizeof(Record*));
	int *d_keys;
	GPUMALLOC( (void**) & d_keys, sLen*sizeof(int) ); 
	TOGPU( d_keys, h_SKeys,	sLen*sizeof(int));
	Record *d_R;
	GPUMALLOC( (void**) & d_R, rLen*sizeof(Record) ); 
	TOGPU( d_R, h_R,	rLen*sizeof(Record));
	Bound *d_bound;
	GPUMALLOC( (void**) & d_bound, listLen*sizeof(Record) ); 
	TOGPU( d_bound, h_bound,	listLen*sizeof(Record));
	endTimer("copy to GPU",&timer);

	startTimer(&timer);
    hashSearch(d_R, rLen, d_bound, d_keys, sLen, h_Rout);
	endTimer("hash search",&timer);


	double sec=endTime("hash search");
	CPUFREE(h_Rout);
	GPUFREE(d_bound);
	GPUFREE(d_R);
	GPUFREE(d_keys);

	
}

void test_AccessMethods(int argc, char **argv)
{
	int i=0;
	for(i=0;i<argc;i++)
	{
		if(strcmp(argv[i],"-TreeSearch")==0)
		{
			int rLen=8*1024*1024;
			int sLen=8*1024*1024;
			if(argc==(i+3))
			{
				rLen=atoi(argv[i+1])*1024;
				sLen=atoi(argv[i+2])*1024;
			}
			testTreeSearch(rLen,sLen);
		}

		if(strcmp(argv[i],"-HashSearch")==0)
		{
			int rLen=8*1024*1024;
			int sLen=8*1024*1024;
			if(argc==(i+3))
			{
				rLen=atoi(argv[i+1])*1024;
				sLen=atoi(argv[i+2])*1024;
			}
			testHashSearch(rLen,sLen);
		}
	}
}



#endif

