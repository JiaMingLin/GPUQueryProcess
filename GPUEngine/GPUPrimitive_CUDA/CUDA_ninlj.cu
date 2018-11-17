
#ifndef _CUDA_NLJ_H_
#define _CUDA_NLJ_H_

#include "stdio.h"
#include "stdlib.h"
#include "scan.cu"
#include "CUDA_ninlj_kernel.cu"
#include "common.cu"
#include "GPU_Dll.h"

int gpu_ninlj(Record *d_R, int rLen, Record *d_S, int sLen, Record** d_Rout)
{
	int numThreadPerBlock=NLJ_NUM_THREADS_PER_BLOCK;
	int numBlock_X=4*NLJ_SHARED_MEM/(sizeof(Record)*NLJ_S_BLOCK_SIZE);
	/*if(numThreadPerBlock*numBlock_X>sLen)
	{
		numBlock_X=sLen/numThreadPerBlock;
		if(sLen%numThreadPerBlock!=0)
			numBlock_X++;
	}*/
	int numBlock_Y=1;
	if(numBlock_X>NLJ_MAX_NUM_BLOCK_PER_DIM)
	{
		numBlock_Y=numBlock_X/NLJ_MAX_NUM_BLOCK_PER_DIM;
		if(numBlock_X%NLJ_MAX_NUM_BLOCK_PER_DIM!=0)
			numBlock_Y++;
		numBlock_X=NLJ_MAX_NUM_BLOCK_PER_DIM;
	}
	int numBlock=numBlock_X*numBlock_Y;
	int gridSize=numBlock*NLJ_S_BLOCK_SIZE;
	int numGrid=sLen/gridSize;
	if(sLen%gridSize!=0)
		numGrid++;
	printf("numBlock, %d, gridSize, %d, numGrid, %d\n",numBlock, gridSize, numGrid);
	dim3  threads_NLJ( numThreadPerBlock, 1, 1);
	dim3  grid_NLJ( numBlock_X, numBlock_Y, 1);
	int numResults=0;
	//the number of results for threads
	
	int originalResultBufSize=grid_NLJ.x*grid_NLJ.y*numThreadPerBlock;
	int resultBuf=get2N(originalResultBufSize);
	int* d_n;
	GPUMALLOC((void**)&d_n, sizeof(int)*resultBuf );
	//the prefix sum for d_n
	int *d_sum;//the prefix sum for d_n[1,...,n]
	GPUMALLOC((void**)&d_sum, sizeof(int)*resultBuf );
	
	
	int* h_n ;
	CPUMALLOC((void**)&h_n, sizeof(int));
	int* h_sum ;
	CPUMALLOC((void**)&h_sum, sizeof(int));
	int outSize=rLen;
	printf("output size: %d\n",outSize);
	//one result buffer for one grid.
	Record ** h_outBuf;
	CPUMALLOC((void**)&(h_outBuf),sizeof(Record*)*numGrid); 
	int *h_numResultCurRun;
	CPUMALLOC((void**)&h_numResultCurRun,sizeof(int)*numGrid); 
	
	for(int sg=0;sg<numGrid;sg++)	
	{
		h_outBuf[sg]=NULL;
		h_numResultCurRun[sg]=0;
	}
	Record *d_outBuf;//the prefix sum for d_n[1,...,n]
	GPUMALLOC((void**) &d_outBuf, sizeof(Record)*outSize);
	CUDA_SAFE_CALL( cudaMemset( d_outBuf, 0, sizeof(Record)*outSize));

#ifdef SHARED_MEM
	printf( "YES, SHARED MEMORY, ninlj\n" );
#else
	printf( "NO SHARED MEMORY, ninlj\n" );
#endif

#ifdef COALESCED
	printf( "YES, COALESCED, ninlj\n" );
#else
	printf( "NO COALESCED, ninlj\n" );
#endif

	int sStart=0;
	saven_initialPrefixSum(resultBuf);	
	for(int sg=0;sg<numGrid;sg++)	
	{
		sStart=sg*gridSize;
		printf("Start=%d, ", sStart);
#ifdef SHARED_MEM
	#ifdef COALESCED
		int* d_temp;
		GPUMALLOC( (void**)&d_temp, sizeof(int)*rLen );
		gpuNLJ_kernel<<< grid_NLJ, threads_NLJ >>>(d_temp, d_R, d_S, sStart, rLen, sLen, d_n);
		GPUFREE( d_temp );
	#else
		gpuNLJ_noCoalesced_kernel<<< grid_NLJ, threads_NLJ >>>(d_R, d_S, sStart, rLen, sLen, d_n);
	#endif
		CUDA_SAFE_CALL(cudaThreadSynchronize());
#else
		Record* d_shared_s;
		int* d_temp;
		GPUMALLOC( (void**)&d_temp, sizeof(int)*rLen );
		GPUMALLOC( (void**)&d_shared_s, sizeof(Record)*NLJ_S_BLOCK_SIZE*grid_NLJ.x*grid_NLJ.y );
		gpuNLJ_kernel<<< grid_NLJ, threads_NLJ >>>(d_temp, d_shared_s, d_R, d_S, sStart, rLen, sLen, d_n);		
		CUDA_SAFE_CALL(cudaThreadSynchronize());
		GPUFREE( d_temp );
		GPUFREE( d_shared_s );
#endif
		CUDA_SAFE_CALL(cudaThreadSynchronize());
		//prescanArray(d_sum, d_n,  16);
		//prefix sum to check out the result position.
		//gpuPrint(d_n, resultBuf, "d_n");
		prescanArray( d_sum,d_n, resultBuf);
		FROMGPU(h_n, (d_n+resultBuf-1), sizeof(int));
		FROMGPU(h_sum, (d_sum+resultBuf-1), sizeof(int));
		h_numResultCurRun[sg]=*h_n+*h_sum;
		numResults+=h_numResultCurRun[sg];
		printf("h_numResultCurRun=%d, ", h_numResultCurRun[sg]);
		if(h_numResultCurRun[sg]>0)
		{	//enlarge the output size.
			outSize=h_numResultCurRun[sg];
			GPUMALLOC((void**) &d_outBuf, sizeof(Record)*outSize );
#ifdef SHARED_MEM
	#ifdef COALESCED
				write<<< grid_NLJ, threads_NLJ >>>(d_R, d_S, sStart, rLen, sLen,d_sum, d_outBuf );	
	#else
				write_noCoalesced<<< grid_NLJ, threads_NLJ >>>(d_R, d_S, sStart, rLen, sLen,d_sum, d_outBuf );
	#endif
				CUDA_SAFE_CALL(cudaThreadSynchronize());
#else
			Record* d_shared_s;
			GPUMALLOC( (void**)&d_shared_s, sizeof(Record)*NLJ_S_BLOCK_SIZE*grid_NLJ.x*grid_NLJ.y );
			write<<< grid_NLJ, threads_NLJ >>>(d_shared_s, d_R, d_S, sStart, rLen, sLen,d_sum, d_outBuf );
			CUDA_SAFE_CALL(cudaThreadSynchronize());
			GPUFREE( d_shared_s );
#endif
			//h_outBuf[sg]=(Record *)malloc(sizeof(Record)*h_numResultCurRun[sg]);
			startSumTime();
			//CPUMALLOC((void**)&h_outBuf[sg],sizeof(Record)*h_numResultCurRun[sg]);
			//FROMGPU(h_outBuf[sg], d_outBuf, sizeof(Record)*h_numResultCurRun[sg]);
			h_outBuf[sg]=d_outBuf;
			endSumTime();
		}
		printf( "\nThe number of results 2: %d\n", numResults );//numResults=0;*/
		/*if((sg+1)!=numGrid)
		{
			CUDA_SAFE_CALL( cudaMemset( d_n, 0, sizeof(int)*resultBuf));
			CUDA_SAFE_CALL( cudaMemset( d_outBuf, 0, sizeof(Record)*outSize));
		}*/
	}
	deallocBlockSums();
	printSumTime("copy back");

	//dump the final results to Rout;
	GPUMALLOC((void **)&(*d_Rout),sizeof(Record)*numResults);
	int rstart=0;
	for(int sg=0;sg<numGrid;sg++)
	{
		GPUTOGPU(*d_Rout+rstart,h_outBuf[sg],sizeof(Record)*h_numResultCurRun[sg]);
		rstart+=h_numResultCurRun[sg];
	}

    
	for(int sg=0;sg<numGrid;sg++)
	{
		if(h_outBuf[sg]!=NULL)
			GPUFREE(h_outBuf[sg]);
	}
	CPUFREE(h_outBuf);
	GPUFREE(d_n);
	GPUFREE(d_sum);
	CPUFREE(h_n);
	CPUFREE(h_sum);
	return numResults;
}

//with Constant buffer.


int gpu_ninlj_Constant(Record *d_R, int rLen, Record *d_S, int sLen, Record** d_Rout)
{
	int numThreadPerBlock=NLJ_NUM_THREADS_PER_BLOCK;
	int numBlock_X=4*NLJ_SHARED_MEM/(sizeof(Record)*NLJ_S_BLOCK_SIZE);
	/*if(numThreadPerBlock*numBlock_X>sLen)
	{
		numBlock_X=sLen/numThreadPerBlock;
		if(sLen%numThreadPerBlock!=0)
			numBlock_X++;
	}*/
	int numBlock_Y=1;
	if(numBlock_X>NLJ_MAX_NUM_BLOCK_PER_DIM)
	{
		numBlock_Y=numBlock_X/NLJ_MAX_NUM_BLOCK_PER_DIM;
		if(numBlock_X%NLJ_MAX_NUM_BLOCK_PER_DIM!=0)
			numBlock_Y++;
		numBlock_X=NLJ_MAX_NUM_BLOCK_PER_DIM;
	}
	int numBlock=numBlock_X*numBlock_Y;
	int gridSize=numBlock*NLJ_S_BLOCK_SIZE;
	int numGrid=sLen/gridSize;
	printf("numBlock, %d, gridSize, %d, numGrid, %d\n",numBlock, gridSize, numGrid);
	if(sLen%gridSize!=0)
		numGrid++;
	dim3  threads_NLJ( numThreadPerBlock, 1, 1);
	dim3  grid_NLJ( numBlock_X, numBlock_Y, 1);
	int numResults=0;
	//the number of results for threads
	
	int originalResultBufSize=grid_NLJ.x*grid_NLJ.y*numThreadPerBlock;
	int resultBuf=get2N(originalResultBufSize);
	int* d_n;
	GPUMALLOC((void**)&d_n, sizeof(int)*resultBuf );
	//the prefix sum for d_n
	int *d_sum;//the prefix sum for d_n[1,...,n]
	GPUMALLOC((void**)&d_sum, sizeof(int)*resultBuf );
	
	
	int* h_n ;
	CPUMALLOC((void**)&h_n, sizeof(int));
	int* h_sum ;
	CPUMALLOC((void**)&h_sum, sizeof(int));
	int outSize=rLen;
	printf("output size: %d\n",outSize);
	//one result buffer for one grid.
	Record ** h_outBuf;
	CPUMALLOC((void**)&(h_outBuf),sizeof(Record*)*numGrid); 
	int *h_numResultCurRun;
	CPUMALLOC((void**)&h_numResultCurRun,sizeof(int)*numGrid); 
	
	for(int sg=0;sg<numGrid;sg++)	
	{
		h_outBuf[sg]=NULL;
		h_numResultCurRun[sg]=0;
	}
	Record *d_outBuf;//the prefix sum for d_n[1,...,n]
	GPUMALLOC((void**) &d_outBuf, sizeof(Record)*outSize);
	CUDA_SAFE_CALL( cudaMemset( d_outBuf, 0, sizeof(Record)*outSize));

	int sStart=0;
	saven_initialPrefixSum(resultBuf);	
	for(int sg=0;sg<numGrid;sg++)	
	{
		sStart=sg*gridSize;
		printf("Start=%d, ", sStart);
		int* d_temp;
		GPUMALLOC( (void**)&d_temp, sizeof(int)*rLen );
		gpuNLJ_kernel<<< grid_NLJ, threads_NLJ >>>(d_temp, d_R, d_S, sStart, rLen, sLen, d_n);
		GPUFREE( d_temp );
		//prescanArray(d_sum, d_n,  16);
		//prefix sum to check out the result position.
		//gpuPrint(d_n, resultBuf, "d_n");
		prescanArray( d_sum,d_n, resultBuf);
		FROMGPU(h_n, (d_n+resultBuf-1), sizeof(int));
		FROMGPU(h_sum, (d_sum+resultBuf-1), sizeof(int));
		h_numResultCurRun[sg]=*h_n+*h_sum;
		numResults+=h_numResultCurRun[sg];
		printf("h_numResultCurRun=%d, ", h_numResultCurRun[sg]);
		if(h_numResultCurRun[sg]>0)
		{	//enlarge the output size.
			outSize=h_numResultCurRun[sg];
			GPUMALLOC((void**) &d_outBuf, sizeof(Record)*outSize );
			write<<< grid_NLJ, threads_NLJ >>>(d_R, d_S, sStart, rLen, sLen,d_sum, d_outBuf );				
			//h_outBuf[sg]=(Record *)malloc(sizeof(Record)*h_numResultCurRun[sg]);
			startSumTime();
			//CPUMALLOC((void**)&h_outBuf[sg],sizeof(Record)*h_numResultCurRun[sg]);
			//FROMGPU(h_outBuf[sg], d_outBuf, sizeof(Record)*h_numResultCurRun[sg]);
			h_outBuf[sg]=d_outBuf;
			endSumTime();
		}
		printf( "\nThe number of results 2: %d\n", numResults );//numResults=0;*/
		/*if((sg+1)!=numGrid)
		{
			CUDA_SAFE_CALL( cudaMemset( d_n, 0, sizeof(int)*resultBuf));
			CUDA_SAFE_CALL( cudaMemset( d_outBuf, 0, sizeof(Record)*outSize));
		}*/
	}
	printSumTime("copy back");
	deallocBlockSums();

	//dump the final results to Rout;
	GPUMALLOC((void **)&(*d_Rout),sizeof(Record)*numResults);
	int rstart=0;
	for(int sg=0;sg<numGrid;sg++)
	{
		GPUTOGPU(*d_Rout+rstart,h_outBuf[sg],sizeof(Record)*h_numResultCurRun[sg]);
		rstart+=h_numResultCurRun[sg];
	}

    
	for(int sg=0;sg<numGrid;sg++)
	{
		if(h_outBuf[sg]!=NULL)
			GPUFREE(h_outBuf[sg]);
	}
	CPUFREE(h_outBuf);
	GPUFREE(d_n);
	GPUFREE(d_sum);
	CPUFREE(h_n);
	CPUFREE(h_sum);
	return numResults;
}



int matchingBlocks(Record *d_R, int rLen, Record *d_S, int sLen, Record** d_match)
{
	int numThreadPerBlock=NLJ_NUM_THREADS_PER_BLOCK;
	int numBlock_X=2*NLJ_SHARED_MEM/(sizeof(Record)*NLJ_S_BLOCK_SIZE);
	/*if(numThreadPerBlock*numBlock_X>sLen)
	{
		numBlock_X=sLen/numThreadPerBlock;
		if(sLen%numThreadPerBlock!=0)
			numBlock_X++;
	}*/
	int numBlock_Y=1;
	if(numBlock_X>NLJ_MAX_NUM_BLOCK_PER_DIM)
	{
		numBlock_Y=numBlock_X/NLJ_MAX_NUM_BLOCK_PER_DIM;
		if(numBlock_X%NLJ_MAX_NUM_BLOCK_PER_DIM!=0)
			numBlock_Y++;
		numBlock_X=NLJ_MAX_NUM_BLOCK_PER_DIM;
	}
	int numBlock=numBlock_X*numBlock_Y;
	int gridSize=numBlock*NLJ_S_BLOCK_SIZE;
	int numGrid=sLen/gridSize;
	if(sLen%gridSize!=0)
		numGrid++;
	dim3  threads_NLJ( numThreadPerBlock, 1, 1);
	dim3  grid_NLJ( numBlock_X, numBlock_Y, 1);
	int numResults=0;
	//the number of results for threads
	
	int originalResultBufSize=grid_NLJ.x*grid_NLJ.y*numThreadPerBlock;
	int resultBuf=get2N(originalResultBufSize);
	int* d_n;
	GPUMALLOC((void**)&d_n, sizeof(int)*resultBuf );
	//the prefix sum for d_n
	int *d_sum;//the prefix sum for d_n[1,...,n]
	GPUMALLOC((void**)&d_sum, sizeof(int)*resultBuf );
	
	
	int* h_n ;
	CPUMALLOC((void**)&h_n, sizeof(int));
	int* h_sum ;
	CPUMALLOC((void**)&h_sum, sizeof(int));
	int outSize=rLen;
	printf("output size: %d\n",outSize);
	//one result buffer for one grid.
	Record ** h_outBuf;
	CPUMALLOC((void**)&(h_outBuf),sizeof(Record*)*numGrid); 
	int *h_numResultCurRun;
	CPUMALLOC((void**)&h_numResultCurRun,sizeof(int)*numGrid); 
	
	for(int sg=0;sg<numGrid;sg++)	
	{
		h_outBuf[sg]=NULL;
		h_numResultCurRun[sg]=0;
	}
	Record *d_outBuf;//the prefix sum for d_n[1,...,n]
	GPUMALLOC((void**) &d_outBuf, sizeof(Record)*outSize);
	CUDA_SAFE_CALL( cudaMemset( d_outBuf, 0, sizeof(Record)*outSize));

	int sStart=0;
	saven_initialPrefixSum(resultBuf);	
	for(int sg=0;sg<numGrid;sg++)	
	{
		sStart=sg*gridSize;
		printf("Start=%d, ", sStart);
		matchCount_kernel<<< grid_NLJ, threads_NLJ >>>(d_R, d_S, sStart, rLen, sLen, d_n);
		//prescanArray(d_sum, d_n,  16);
		//prefix sum to check out the result position.
		//gpuPrint(d_n, resultBuf, "d_n");
		prescanArray( d_sum,d_n, resultBuf);
		FROMGPU(h_n, (d_n+resultBuf-1), sizeof(int));
		FROMGPU(h_sum, (d_sum+resultBuf-1), sizeof(int));
		h_numResultCurRun[sg]=*h_n+*h_sum;
		numResults+=h_numResultCurRun[sg];
		printf("h_numResultCurRun=%d, ", h_numResultCurRun[sg]);
		if(h_numResultCurRun[sg]>0)
		{	//enlarge the output size.
			outSize=h_numResultCurRun[sg];
			GPUMALLOC((void**) &d_outBuf, sizeof(Record)*outSize );
			matchWrite_kernel<<< grid_NLJ, threads_NLJ >>>(d_R, d_S, sStart, rLen, sLen,d_sum, d_outBuf );				
			//h_outBuf[sg]=(Record *)malloc(sizeof(Record)*h_numResultCurRun[sg]);
			startSumTime();
			//CPUMALLOC((void**)&h_outBuf[sg],sizeof(Record)*h_numResultCurRun[sg]);
			//FROMGPU(h_outBuf[sg], d_outBuf, sizeof(Record)*h_numResultCurRun[sg]);
			h_outBuf[sg]=d_outBuf;
			endSumTime();
		}
		printf( "\nThe number of matchingBlocks 2: %d\n", numResults );//numResults=0;*/
		/*if((sg+1)!=numGrid)
		{
			CUDA_SAFE_CALL( cudaMemset( d_n, 0, sizeof(int)*resultBuf));
			CUDA_SAFE_CALL( cudaMemset( d_outBuf, 0, sizeof(Record)*outSize));
		}*/
	}
	printSumTime("copy back");
	deallocBlockSums();

	//dump the final results to Rout;
	GPUMALLOC((void **)&(*d_match),sizeof(Record)*numResults);
	int rstart=0;
	for(int sg=0;sg<numGrid;sg++)
	{
		GPUTOGPU(*d_match+rstart,h_outBuf[sg],sizeof(Record)*h_numResultCurRun[sg]);
		rstart+=h_numResultCurRun[sg];
	}

    
	for(int sg=0;sg<numGrid;sg++)
	{
		if(h_outBuf[sg]!=NULL)
			GPUFREE(h_outBuf[sg]);
	}
	CPUFREE(h_outBuf);
	GPUFREE(d_n);
	GPUFREE(d_sum);
	CPUFREE(h_n);
	CPUFREE(h_sum);
	return numResults;
}

extern "C"
int GPUOnly_ninlj( Record *d_R, int rLen, Record *d_S, int sLen, Record** d_Rout )
{
	return gpu_ninlj(d_R, rLen, d_S, sLen, d_Rout);
}

extern "C"
int GPUCopy_ninlj( Record* h_R, int rLen, Record* h_S, int sLen, Record** h_Rout )
{
	Record* d_R;
	Record* d_S;
	Record* d_Rout;
	int rMemSize = sizeof(Record)*rLen;
	int sMemSize = sizeof(Record)*sLen;

	GPUMALLOC( (void**)&d_R, rMemSize );
	GPUMALLOC( (void**)&d_S, sMemSize );
	TOGPU( d_R, h_R, rMemSize );
	TOGPU( d_S, h_S, sMemSize );

	int numResults = gpu_ninlj(d_R, rLen, d_S, sLen, &d_Rout);

	//*h_Rout = (Record*)malloc( sizeof(Record)*numResults );
	CPUMALLOC( (void**)h_Rout, sizeof(Record)*numResults );
	FROMGPU( (*h_Rout), d_Rout, sizeof(Record)*numResults );

	GPUFREE( d_R );
	GPUFREE( d_S );
	GPUFREE( d_Rout );

	return numResults;
}


#endif
