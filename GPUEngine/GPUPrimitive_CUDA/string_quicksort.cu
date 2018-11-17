#ifndef STRING_QUICKSORT_CU
#define STRING_QUICKSORT_CU

#include "scan.cu"
#include "common.cu"
#include "string_quicksort_kernel.cu"
#include "scan_best_kernel.cu"
#include "assert.h"
#include "string_bitonicProc.cu"
#include "string_bitonicSort.cu"

unsigned int MAX_LEVEL=10;
static int s_MAX_LEVEL_SUPPORTED=20;
const int SMALL_CHUNK_THRESHOLD=SHARED_MEM_INT2;
const int LEVEL_CHUNK_THRESHOLD=1;
const int NUM_PART_THRESHOLD=4;
static int s_numTuplePerPartition=0;
static int s_sampleLevel=0;
const int PREFIX_SUM_SWAP_LEVEL=10;
int compareInt2 (const void * a, const void * b)
{
  return (( (cmp_type_t*)a)->x - ( (cmp_type_t*)b)->x );
}

void checkResult(cmp_type_t *d_R, int rLen, int *d_iBound, int numPart)
{
	//int numPart=(1<<level);//level=0, only one partitions.
#ifdef DEBUG_SAVEN
	cmp_type_t *h_R=(cmp_type_t *)malloc(sizeof(cmp_type_t)*rLen);
	CUDA_SAFE_CALL( cudaMemcpy( h_R, d_R, rLen*sizeof(cmp_type_t) , cudaMemcpyDeviceToHost) );
	int *h_iBound=(int *)malloc(sizeof(int)*numPart*2);
	CUDA_SAFE_CALL( cudaMemcpy( h_iBound, d_iBound, numPart*2*sizeof(int) , cudaMemcpyDeviceToHost) );

	int start=0;
	int end=0;
	int num=0;
	int i=0;
	for(i=0;i<numPart;i++)
	{
		start=h_iBound[(i<<1)];
		end=h_iBound[(i<<1)+1];
		num=end-start;
		if((start==444758 || end==444758))
			start=start;
		qsort(h_R+start,num,sizeof(cmp_type_t),compareInt2);
		if(i<=1)
			printf("checking %d: start, %d, end, %d, small, %d, large, %d\n", i, start, end, h_R[start].x, h_R[end-1].x);		
	}
	double sum=0;
	int failTimes=0;
	for(i=0;i<(rLen-1);i++)
	{
		sum+=h_R[i].x/1000.0;
		//printf("%d, ",h_R[i]);
		//if(i%10==9) printf("\n");
		if(h_R[i].x>h_R[i+1].x)
		{
			printf("!!!check fails, %d, %d, %d, %d\n",i, h_R[i].x, i+1, h_R[i+1].x);
			failTimes++;
			if(failTimes>10)
			break;
		}
	}
	if(i==(rLen-1)&&failTimes==0)
	{
		sum+=h_R[i].x/1000.0;
		printf("succeed: %f\n", sum);
	}
#endif
}

void checkSortedResult(cmp_type_t *d_R, int rLen, int *d_iBound, int numPart)
{
	//int numPart=(1<<level);//level=0, only one partitions.
#ifdef DEBUG_SAVEN
	if(numPart==0)
		return;
	cmp_type_t *h_R=(cmp_type_t *)malloc(sizeof(cmp_type_t)*rLen);
	CUDA_SAFE_CALL( cudaMemcpy( h_R, d_R, rLen*sizeof(cmp_type_t) , cudaMemcpyDeviceToHost) );
	int *h_iBound=(int *)malloc(sizeof(int)*numPart*2);
	CUDA_SAFE_CALL( cudaMemcpy( h_iBound, d_iBound, numPart*2*sizeof(int) , cudaMemcpyDeviceToHost) );

	int start=0;
	int end=0;
	int num=0;
	int i=0;
	int failTimes=0;
	for(i=0;i<numPart;i++)
	{
		start=h_iBound[(i<<1)];
		end=h_iBound[(i<<1)+1];
		num=end-start;
		for(int j=start+1;j<end;j++)
		{
			if(h_R[j].x<h_R[j-1].x)
			{
				printf("////part: %d, %d; range:[%d, %d]: check fails, %d, %d, %d, %d\n",i, numPart, start, end, j-1, h_R[j-1].x, j, h_R[j].x);
				failTimes++;
				if(failTimes>10)
				break;
			}
		}	
	}
	if(failTimes==0)
		printf("succeeds in checkSortedResult\n");
#endif
}

void checkFinalResult(cmp_type_t *d_R, int rLen)
{
#ifdef DEBUG_SAVEN
	cmp_type_t *h_R=(cmp_type_t *)malloc(sizeof(cmp_type_t)*rLen);
	CUDA_SAFE_CALL( cudaMemcpy( h_R, d_R, rLen*sizeof(cmp_type_t) , cudaMemcpyDeviceToHost) );
	double sum=0;
	int i=0;
	for(i=0;i<(rLen-1);i++)
	{
		sum+=h_R[i]/1000.0;
		//printf("%d, ",h_R[i].y);
		//if(i%10==9) printf("\n");
		if(h_R[i]>h_R[i+1] || h_R[i]<0)
		{
			printf("check fails, %d, %d\n",i, h_R[i], i+1, h_R[i+1]);
			break;
		}
	}
	if(i==(rLen-1))
	{
		sum+=h_R[i]/1000.0;
		printf("succeed: %f\n", sum);
	}
#endif
}


void sortSmallChunks(void* d_rawData, int totalLenInBytes, cmp_type_t *d_R, int rLen, int *d_iBound, int numSmallChunks, cmp_type_t* d_output)
{
	string_bitonicSortMultipleBlocks(d_rawData, totalLenInBytes, d_R, d_iBound, numSmallChunks, d_output);
}

int quickPartUsingPresetPivot(void* rawData, int totalLenInBytes,int numPart, int *h_bound, int *small_bound, cmp_type_t *d_R, int rLen, int *d_iBound, int level, cmp_type_t *d_S, int *d_oBound, int *d_count, int* d_sum, cmp_type_t *d_pivot)
{
	array_startTime(1);
	int resultNumPart=0;
	//int s_numTuplePerPartition=rLen/numPart;
	int expectedThreads=1<<((int)logb((float)s_numTuplePerPartition));
	int numThreadsPerBlock=expectedThreads;
	if(expectedThreads>256)
		numThreadsPerBlock=256;
	if(expectedThreads<32)
		numThreadsPerBlock=32;
	/*if(level>=7)
		numThreadsPerBlock=32;*/
	int numTuplePerThead=8;
	int temp=s_numTuplePerPartition/numThreadsPerBlock/numTuplePerThead;
	if(s_numTuplePerPartition%numThreadsPerBlock!=0) temp++;
	int expectedTheadBlock=1<<((int)logb((float)temp));
	int numBlock_x=expectedTheadBlock;//qBlocks[level];
	dim3  thread( numThreadsPerBlock, 1, 1);
	dim3  grid( numBlock_x, numPart , 1);
	//the number of one's (or zeroes) for each partition
	int numResultsPerCount=thread.x*grid.x;//[zeros,ones]
	//total number of counts.
	int numResults=numResultsPerCount*2*grid.y;
	//the number of tuples processed by one thread block
	int blockSize=s_numTuplePerPartition/numBlock_x;
	//the number of counts per partitions.
	int sumRun=numResultsPerCount*2;
	printf("level, %d, threadx, %d, blockx, %d, blockSize, %d, numPerThread, %d, numResultsPerCount, %d\n", level, thread.x, grid.x, blockSize, blockSize/thread.x, numResultsPerCount);
	startTime();
	int pivotStart=(1<<level)-1;
	binaryPart_level<<<grid,thread>>>(rawData, totalLenInBytes, numPart, d_R, d_iBound, blockSize,d_count,numResultsPerCount, d_pivot, pivotStart);
	cudaThreadSynchronize();
	CUT_CHECK_ERROR("Kernel execution failed");
	endTime("couting");
	gpuPrint(d_count, numResults, "count");	
	int offset=0;
	if(level==0)
		saven_initialPrefixSum(sumRun);
	//printf("prefix sum on ");
	
	if(level<PREFIX_SUM_SWAP_LEVEL)
	//if(level<5)
	{
		startTime();
		for(int i=0;i<numPart;i++)
		{
			offset=numResultsPerCount*i*2;
			prescanArray((d_sum+offset),(d_count+offset),sumRun);
			//printf("ps %i, ", i);
			gpuPrint((d_sum+offset), sumRun, "prefix sum");
		}
		cudaThreadSynchronize();
		endTime("prefix sum");
	}
	else
	{
		startTime();
		unsigned int extra_space = sumRun / NUM_BANKS;
		const unsigned int shared_mem_size = sizeof(int) * (sumRun + extra_space);
		printf("shared_mem_size %d KB,\n ", shared_mem_size/1024);
		assert(shared_mem_size/1024<=32);
		dim3  ps_Thread( sumRun/2, 1, 1);
		dim3  ps_grid( numPart, 1 , 1);
		computePS<<< ps_grid, ps_Thread, shared_mem_size >>>(d_sum, d_count, sumRun);
		cudaThreadSynchronize();
		endTime("prefix sum II");
	}
	
	int nChunk=1;
	startTime();
	write_level<<<grid,thread>>>(rawData, totalLenInBytes, numPart, d_R, d_iBound,blockSize,d_sum,numResultsPerCount,nChunk, d_S,d_oBound, d_pivot, pivotStart);
	endTime("write_level");
	array_endTime("first step", 1);

	array_startTime(1);	
	//gpuPrintInt2(d_S, rLen, "new relation");
	gpuPrintInterval(d_oBound, numPart*4, "d_oBound");
	resultNumPart=numPart*2;
	s_numTuplePerPartition=rLen/resultNumPart/2;
	array_endTime("second step", 1);
	
	return resultNumPart;
}

int quickPart_SelfPivot(void* rawData, int totalLenInBytes,int numPart, int *h_bound, int *small_bound, cmp_type_t *d_R, int rLen, int *d_iBound, int level, cmp_type_t *d_S, int *d_oBound, int *d_count, int* d_sum, cmp_type_t *d_pivot)
{
	array_startTime(1);
	int resultNumPart=0;
	//int s_numTuplePerPartition=rLen/numPart;
	int expectedThreads=1<<((int)logb((float)s_numTuplePerPartition));
	int numThreadsPerBlock=expectedThreads;
	if(expectedThreads>256)
		numThreadsPerBlock=256;
	if(expectedThreads<32)
		numThreadsPerBlock=32;
	/*if(level>=7)
		numThreadsPerBlock=32;*/
	int numTuplePerThead=8;
	int temp=s_numTuplePerPartition/numThreadsPerBlock/numTuplePerThead;
	if(s_numTuplePerPartition%numThreadsPerBlock!=0) temp++;
	int expectedTheadBlock=1<<((int)logb((float)temp));
	int numBlock_x=expectedTheadBlock;//qBlocks[level];
	dim3  thread( numThreadsPerBlock, 1, 1);
	dim3  grid( numBlock_x, numPart , 1);
	//int numBlocks=numBlock_x*numPart;
	int numResultsPerCount=thread.x*grid.x;//[zeros,ones]
	int numResults=numResultsPerCount*2*grid.y;
	//assert(numResults<=rLen*2);	
	//int blockSize=rLen/numBlocks;
	int blockSize=s_numTuplePerPartition/numBlock_x;//we need to consider the block size here.
	printf("level, %d, threadx, %d, blockx, %d, blockSize, %d, numPerThread, %d, numResultsPerCount, %d\n", level, thread.x, grid.x, blockSize, blockSize/thread.x, numResultsPerCount);
	startTime();
	binaryPart_level_selfPivot<<<grid,thread>>>(rawData, totalLenInBytes, numPart, d_R, d_iBound, blockSize,d_count,numResultsPerCount, d_pivot);
	cudaThreadSynchronize();
	CUT_CHECK_ERROR("Kernel execution failed");
	endTime("couting");
	gpuPrint(d_count, numResults, "count");
	int sumRun=numResultsPerCount*2;
	int offset=0;
	if(level==0)
		saven_initialPrefixSum(sumRun);
	//printf("prefix sum on ");
	
	if(numPart<(1<<PREFIX_SUM_SWAP_LEVEL))
	{
		startTime();
		for(int i=0;i<numPart;i++)
		{
			offset=numResultsPerCount*i*2;
			prescanArray((d_sum+offset),(d_count+offset),sumRun);
			//printf("ps %i, ", i);
			gpuPrint((d_sum+offset), sumRun, "prefix sum");
		}
		cudaThreadSynchronize();
		endTime("prefix sum");
	}
	else
	{
		startTime();
		unsigned int extra_space = sumRun / NUM_BANKS;
		const unsigned int shared_mem_size = sizeof(int) * (sumRun + extra_space);
		//printf("shared_mem_size %d KB,\n ", shared_mem_size/1024);
		assert(shared_mem_size/1024<=16);
		dim3  ps_Thread( sumRun/2, 1, 1);
		dim3  ps_grid( numPart, 1 , 1);
		computePS<<< ps_grid, ps_Thread, shared_mem_size >>>(d_sum, d_count, sumRun);
		cudaThreadSynchronize();
		endTime("prefix sum II");
	}
	
	int nChunk=1;
	startTime();
	write_level_selfPivot<<<grid,thread>>>(rawData, totalLenInBytes, numPart, d_R, d_iBound,blockSize,d_sum,numResultsPerCount,nChunk, d_S,d_oBound, d_pivot);
	endTime("write_level");
	array_endTime("SelfPivot first step", 1);

	printf(">>>>check the results:::\n");
	checkResult(d_S, rLen, d_oBound, numPart*2);

	array_startTime(1);	
	//gpuPrintInt2(d_S, rLen, "new relation");
	gpuPrintInterval(d_oBound, numPart*4, "d_oBound");
	//pick out the small chunks.
	array_startTime(2);
	CUDA_SAFE_CALL( cudaMemcpy( h_bound, d_oBound, numPart*4*sizeof(int), cudaMemcpyDeviceToHost) );
	int i=0;
	resultNumPart=0;
	numPart=numPart*4;
	int numSmallChunks=0;
	int sortedChunks=0;
	s_numTuplePerPartition=0;
	//int boundStart=0;
	for(i=0;i<numPart;i=i+2)
	{
		if(h_bound[i]==6339)
			//boundStart=1;
		if( ((h_bound[i]+1) >= h_bound[i+1]) )
		{
			sortedChunks++;
		}
		else
		{
			if( ((h_bound[i]+SMALL_CHUNK_THRESHOLD) >= h_bound[i+1]) )
			{
				small_bound[(numSmallChunks<<1)]=h_bound[i];
				small_bound[(numSmallChunks<<1)+1]=h_bound[i+1];
				//printf("h_bound: %d, %d; ", h_bound[i], h_bound[i+1]);
				numSmallChunks++;
			}
			else//big chunks.
			{
				h_bound[(resultNumPart<<1)]=h_bound[i];
				h_bound[(resultNumPart<<1)+1]=h_bound[i+1];
				s_numTuplePerPartition+=h_bound[i+1]-h_bound[i];
				resultNumPart++;				
			}
		}
		//boundStart=h_bound[i];
	}
	//printf("&&&&&&&&&& small bounds &&&&&&&&&&&&\n");
	//for(i=0;i<numSmallChunks;i++)
	//	printf(" %d, %d, %d; ", i, small_bound[i<<1], small_bound[(i<<1)+1]);
	//printf("&&&&&&&&&& big bounds &&&&&&&&&&&&\n");
	//for(i=0;i<resultNumPart;i++)
	//	printf("%d, %d, %d;", i, h_bound[i<<1], h_bound[(i<<1)+1]);
	//printf("\n");
	int remainUnSorted=s_numTuplePerPartition;
	if(resultNumPart>=NUM_PART_THRESHOLD)
	{
		s_numTuplePerPartition=s_numTuplePerPartition/resultNumPart;
		printf("T: %d, numSmallChunks, %d, sortedChunks, %d, numLargeChunks,%d,s_numTuplePerPartition, %d, unsorted, %dK\n ", SMALL_CHUNK_THRESHOLD, numSmallChunks, sortedChunks, resultNumPart, s_numTuplePerPartition,remainUnSorted/1024);
		CUDA_SAFE_CALL( cudaMemcpy( d_oBound, h_bound, resultNumPart*2*sizeof(int), cudaMemcpyHostToDevice) );	
	}
	CUDA_SAFE_CALL( cudaMemcpy( d_iBound, small_bound, numSmallChunks*2*sizeof(int), cudaMemcpyHostToDevice) );	
	
	array_endTime("copy time", 2);
	array_startTime(3);
	if(numSmallChunks>0)
	{
		//printf("------check the results----\n");
		//checkSortedResult(d_S, rLen, d_iBound, numSmallChunks);
		sortSmallChunks(rawData, totalLenInBytes,d_S, rLen, d_iBound, numSmallChunks, d_S);
		printf("------check the results----\n");
		checkSortedResult(d_S, rLen, d_iBound, numSmallChunks);
		cudaThreadSynchronize();
	}
	array_endTime("smallchunks",3);
	array_endTime("SelfPivot second step", 1);
	
	return resultNumPart;
}
//return the number of sample levels.
void getPivot(void* rawData, int totalLenInBytes, cmp_type_t *Rin, int rLen, int sampleLevel, void* d_pivot)
{
	array_startTime(4);
	int numSample=(1<<sampleLevel);
	cmp_type_t *sampleRin=(cmp_type_t*)malloc(sizeof(cmp_type_t)*numSample);
	int segSize=rLen/numSample;
	int i=0;
	int tempIndex=0;
	for(i=0;i<rLen && tempIndex<numSample;i=i+segSize)
	{
		sampleRin[tempIndex]=Rin[i];
		tempIndex++;
	}
	assert(tempIndex==numSample);
	cmp_type_t **Rout=(cmp_type_t**)malloc(sizeof(cmp_type_t*));
	
	//bitonicSort( *data, totalLenInBytes, Rin, numString, (void**)Rout);
	string_bitonicSort( rawData, totalLenInBytes, sampleRin, numSample, (void**)Rout);	
	//copy to sampleInt
	int j=0;
	int startIndex=0, delta=0;
	numSample=numSample-1;
	tempIndex=0;
	for(i=0;i<sampleLevel;i++)
	{
		delta=1<<(sampleLevel-i);
		startIndex=numSample>>(i+1);		
		for(j=startIndex;j<numSample;j=j+delta)
		{
			sampleRin[tempIndex]=(*Rout)[j];
			//sampleRin[tempIndex].y=(*Rout)[j].y;
			tempIndex++;
		}
	}
	//printString(rawData, *Rout, numSample);
	//printf("---dump---\n");
	//printString(rawData, sampleRin, numSample);
	CUDA_SAFE_CALL( cudaMemcpy( d_pivot, sampleRin, numSample*sizeof(cmp_type_t), cudaMemcpyHostToDevice) );
	array_endTime("get pivot",4);
}

void quickSortGPU(void *d_rawData, int totalLenInBytes, cmp_type_t *d_Rin, int rLen, cmp_type_t* d_Rout)
{
}
/*
@rawData, the values storing in an array.
@totalLenInBytes, the size of rawData.
@Rin, input, the start position and the size of each record in the array;
Rin[i].x is the start position of the ith record in bytes; 
Rin[i].y is the size of the ith record (bytes);
@Rout, output, the element definition is the same as Rin.
*/

void quickSort(void* rawData, int totalLenInBytes, cmp_type_t *Rin, int rLen, cmp_type_t** Rout)
{
	MAX_LEVEL=(int)logb((float)rLen/(LEVEL_CHUNK_THRESHOLD));
	s_sampleLevel=(int)logb((float)rLen/(SMALL_CHUNK_THRESHOLD))+2;
	s_MAX_LEVEL_SUPPORTED=s_sampleLevel*3/2+4;
	//s_MAX_LEVEL_SUPPORTED=s_sampleLevel+2;
	int numSample=(1<<s_sampleLevel)-1;
	printf("sampleLevel, %d, MAX_LEVEL_SUPPORTED, %d, numSample, %d, rLen, %d, PREFIX_SUM_SWAP_LEVEL, %d\n", s_sampleLevel,s_MAX_LEVEL_SUPPORTED, numSample, rLen, PREFIX_SUM_SWAP_LEVEL);
	cmp_type_t* d_pivot;
	GPUMALLOC( (void**) & d_pivot, numSample*sizeof(cmp_type_t) ); 
	

	//int numPart=(1<<(MAX_LEVEL+1));
	int numPart=(1<<(s_sampleLevel+2));
	printf("maxLevel, %d, maximum expected numPart, %d \n", MAX_LEVEL, numPart);
	
	int totalPASizeBytes=rLen*sizeof(cmp_type_t);
	cmp_type_t *d_S;
	GPUMALLOC( (void**) & d_S, totalPASizeBytes ); 
	cmp_type_t *d_R;
	GPUMALLOC( (void**) & d_R, totalPASizeBytes ); 
	CUDA_SAFE_CALL( cudaMemcpy( d_R, Rin,	totalPASizeBytes, cudaMemcpyHostToDevice) );
	int boundSize=numPart*sizeof(int)*2;
	int *d_oBound;
	GPUMALLOC( (void**) & d_oBound,boundSize  ); 
	CUDA_SAFE_CALL( cudaMemset( d_oBound, -1, boundSize));
	int* small_bound=(int*)malloc(boundSize);
	int* h_iBound=(int*)malloc(boundSize);
	h_iBound[0]=0;
	h_iBound[1]=rLen;
	s_numTuplePerPartition=rLen;
	int *d_iBound;
	GPUMALLOC( (void**) & d_iBound, boundSize ); 
	CUDA_SAFE_CALL( cudaMemset( d_iBound, -1, boundSize));
	CUDA_SAFE_CALL( cudaMemcpy( d_iBound, h_iBound,	sizeof(int)*2, cudaMemcpyHostToDevice) );
	void* d_rawData;
	GPUMALLOC( (void**) &d_rawData, totalLenInBytes) ;
	CUDA_SAFE_CALL( cudaMemcpy( d_rawData, rawData, totalLenInBytes, cudaMemcpyHostToDevice) );
	//gpuPrintInt2(d_R, rLen, "original");
	//we set the d_count and the d_sum to be large to avoid overflowing.
	int numResults=numPart*32;
	int* d_count;
	GPUMALLOC( (void**) & d_count, numResults*sizeof(int) ); 
	CUDA_SAFE_CALL( cudaMemset( d_count, 0, sizeof(int)*numResults));
	int *d_sum;
	GPUMALLOC( (void**) & d_sum, numResults*sizeof(int) ); 
	CUDA_SAFE_CALL( cudaMemset( d_sum, 0, sizeof(int)*numResults));

	int level=0;
	int curNumPart=1, resultNumPart=0;
	//get the pivots.
	getPivot(rawData, totalLenInBytes, Rin, rLen, s_sampleLevel, d_pivot);
	array_startTime(0);
	while(level<s_sampleLevel)
	{
		printf("****************level %d, numPart, %d***************\n", level, curNumPart);
		if(level%2==0)
		{
			CUDA_SAFE_CALL( cudaMemcpy( d_rawData, rawData, totalLenInBytes, cudaMemcpyHostToDevice) );
			resultNumPart=quickPartUsingPresetPivot(d_rawData, totalLenInBytes, curNumPart, h_iBound, small_bound, d_R, rLen, d_iBound, level, d_S, d_oBound,d_count, d_sum, d_pivot);
			checkResult(d_S, rLen, d_oBound, (1<<(level+1)));
			//checkFinalResult(d_oBound, (1<<(level+1)));
		}
		else
		{
			resultNumPart=quickPartUsingPresetPivot(d_rawData, totalLenInBytes, curNumPart, h_iBound, small_bound, d_S, rLen, d_oBound, level, d_R, d_iBound,d_count, d_sum, d_pivot);
			checkResult(d_R, rLen, d_iBound, (1<<(level+1)));
			//checkFinalResult(d_iBound, (1<<(level+1)));
		}
		curNumPart=resultNumPart;
		level++;
	}
	bool needToCopy=false;
	while(curNumPart>=NUM_PART_THRESHOLD && level < s_MAX_LEVEL_SUPPORTED)
	{
		if(level==12)
			level=level;
		printf("****************SelfPivot level %d, numPart, %d***************\n", level, curNumPart);
		if(level%2==0)
		{
			//copy d_S to d_R;
			if(needToCopy)
				CUDA_SAFE_CALL( cudaMemcpy( d_S, d_R,	totalPASizeBytes, cudaMemcpyDeviceToDevice) );			
			resultNumPart=quickPart_SelfPivot(d_rawData, totalLenInBytes, curNumPart, h_iBound, small_bound, d_R, rLen, d_iBound, level, d_S, d_oBound,d_count, d_sum, d_pivot);
			checkResult(d_S, rLen, d_oBound, resultNumPart);
			//checkFinalResult(d_oBound, (1<<(level+1)));
		}
		else
		{
			if(needToCopy)
				CUDA_SAFE_CALL( cudaMemcpy( d_R, d_S,	totalPASizeBytes, cudaMemcpyDeviceToDevice) );
			resultNumPart=quickPart_SelfPivot(d_rawData, totalLenInBytes, curNumPart, h_iBound, small_bound, d_S, rLen, d_oBound, level, d_R, d_iBound,d_count, d_sum, d_pivot);
			checkResult(d_R, rLen, d_iBound, resultNumPart);
			//checkFinalResult(d_iBound, (1<<(level+1)));
		}
		curNumPart=resultNumPart;
		level++;
		if(needToCopy==false)
			needToCopy=true;
		
	}
	array_endTime("totalPreviousSort",0);
	array_startTime(0);
	//the last level, sorting it, need to ping-pong.
	/*if(level%2==0)
	{
		CUDA_SAFE_CALL( cudaMemcpy( d_S, d_R,	totalPASizeBytes, cudaMemcpyDeviceToDevice) );			
		d_inputToFinalSort=d_R;
		d_outputToFinalSort=d_S;
	}
	else
	{
		CUDA_SAFE_CALL( cudaMemcpy( d_R, d_S,	totalPASizeBytes, cudaMemcpyDeviceToDevice) );
		d_inputToFinalSort=d_S;
		d_outputToFinalSort=d_R;
		//CUDA_SAFE_CALL( cudaMemcpy( Rin, d_S, sizeof(Record)*rLen, cudaMemcpyDeviceToHost) );
	}
	
	cudaThreadSynchronize();
	array_endTime("endTimeSort copy",0);
	array_startTime(0);
	for(int i=0;i<resultNumPart;i++)
	{
		bitonicSort_AllocatedData( d_rawData, totalLenInBytes, (d_inputToFinalSort)+h_iBound[(i<<1)], (h_iBound[(i<<1)+1]-h_iBound[(i<<1)]), (d_outputToFinalSort)+h_iBound[(i<<1)]);	
	}	
	
	*Rout=(Record*)malloc(sizeof(Record)*rLen);
	CUDA_SAFE_CALL( cudaMemcpy( *Rout, d_outputToFinalSort, sizeof(Record)*rLen, cudaMemcpyDeviceToHost) );*/
	*Rout=(cmp_type_t*)malloc(sizeof(cmp_type_t)*rLen);
	if(level%2==0)
	{
		CUDA_SAFE_CALL( cudaMemcpy( *Rout, d_R, sizeof(cmp_type_t)*rLen, cudaMemcpyDeviceToHost) );
	}
	else
	{
		CUDA_SAFE_CALL( cudaMemcpy( *Rout, d_S, sizeof(cmp_type_t)*rLen, cudaMemcpyDeviceToHost) );
	}
	
	cudaThreadSynchronize();
	array_endTime("endTimeSort copy",0);
	array_startTime(0);
	for(int i=0;i<resultNumPart;i++)
	{
		string_bitonicSort_AllocatedData( d_rawData, totalLenInBytes, (*Rout)+h_iBound[(i<<1)], (h_iBound[(i<<1)+1]-h_iBound[(i<<1)]), (*Rout)+h_iBound[(i<<1)]);	
	}
	//CUDA_SAFE_CALL( cudaMemcpy( *Rout, d_inputToFinalSort, sizeof(Record)*rLen, cudaMemcpyDeviceToHost) );
	

	array_endTime("endTimeSort",0);
	cudaFree(d_sum);
	cudaFree(d_count);
}


#endif
