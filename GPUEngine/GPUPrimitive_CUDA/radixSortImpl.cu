#ifndef RADIX_SORT_IMPL_CU
#define RADIX_SORT_IMPL_CU
#include "splitImpl.cu"
#include "pickBound.cu"
#include "bitonicProc.cu"

//kernels.
/*
for computing the partition ID according to the radix.
*/
__device__ int getRadixPartID(int key, int moveRightBits, int mask)
{
	return (key>>moveRightBits)&mask;
}

__global__ void 
mapRadix_kernel(Record *d_R, int delta, int rLen, int moveRightBits, int mask, int *d_extra, int *d_output)  
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	for(int pos=resultID;pos<rLen;pos+=delta)
	{
		d_extra[pos]=d_R[pos].x;
		d_output[pos]=getRadixPartID(d_R[pos].y, moveRightBits, mask);
	}	
}

void mapRadix(Record *d_R, int rLen, int moveRightBits, int mask,int *d_S, int *d_extra)
{
	//int *d_extra;
	//GPUMALLOC((void **)&d_extra,rLen*sizeof( int));
	int numThreadsPerBlock_x=256;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int numThread=numBlock_x*numThreadsPerBlock_x;
	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	mapRadix_kernel<<<grid,thread>>>(d_R, numThread, rLen, moveRightBits, mask, d_extra, d_S);
	//GPUFREE(d_extra);
}



int getMaxNumPS(int rLen, int* bitUsedPerPass, int totalNumPass)
{
	int result=0;
	int numPS;
	int i=0;
	int numPart;
	int numBitsUsed=0;
	int numBound;
	int expectedLength=0;
	for(i=0;i<totalNumPass;i++)
	{
		numPart=(1<<bitUsedPerPass[i]);
		numBound=(1<<numBitsUsed);
		expectedLength=rLen/numBound;
		numBitsUsed+=bitUsedPerPass[i];
		int numThreadsPerBlock_x=1<<(log2((int)(SHARED_MEMORY_PER_PROCESSOR/(numPart*sizeof(int)))));
		if(numThreadsPerBlock_x>256)
			numThreadsPerBlock_x=256;
		if(numThreadsPerBlock_x>rLen/numBound)
			numThreadsPerBlock_x=1<<(log2(rLen/numBound));
		numThreadsPerBlock_x=(numThreadsPerBlock_x>32)? numThreadsPerBlock_x: 32;
		//numThreadsPerBlock_x=(numThreadsPerBlock_x>8)? numThreadsPerBlock_x: 8;
		//assert(numThreadsPerBlock_x>=32);
		//int sharedMemSize=(numThreadsPerBlock_x*numPart+2)*sizeof(int);
		//int numThreadsPerBlock_y=1;
		int numTuplePerThread=128;
		int numBlock_x=ceil((double)expectedLength/(double)(numThreadsPerBlock_x*numTuplePerThread));
		int numBlock_y=numBound;
		int numThreadPerPartition=numBlock_x*numThreadsPerBlock_x;
		numPS=numPart*numThreadPerPartition*numBlock_y;
		if(result<numPS) result=numPS;
	}
	return result;
}


//procedures.
int radixPart(Record *d_R, int* d_pidArray, int *d_loc, int* d_Hist, int* d_psSum,
			  int rLen, int numPS, int moveRightBits, int bitPerPass, int totalBitsUsed,
			   Record* d_iBound, int numBound, int* d_extra, 
			   Record* d_oBound, Record *d_S)
{
	int mask=(1<<bitPerPass)-1;
	int numPart=mask+1;
	int expectedLength=rLen/numBound;
	
	mapRadix(d_R, rLen, moveRightBits, mask, d_pidArray, d_extra);	
	int resultPart=splitWithPIDArray(d_R, d_pidArray,d_loc,d_Hist,d_psSum, rLen,numPS, d_iBound, numBound, numPart, expectedLength, d_S, d_oBound);
	return resultPart;
}


void radixSort(Record *d_R, int rLen, int startBits, int totalBitsUsed, int bitPerPass, Record* d_S)
{
	int totalNumPass=totalBitsUsed/bitPerPass;
	//int remindingBits=startBits-totalBitsUsed;
	if(totalBitsUsed%bitPerPass!=0)
		totalNumPass++;
	int *bitUsedPerPass;
	CPUMALLOC((void**)&bitUsedPerPass, totalNumPass*sizeof(int));
	int curPass=0;
	for(curPass=0;curPass<totalNumPass-1;curPass++)
		bitUsedPerPass[curPass]=bitPerPass;
	//the last pass
	bitUsedPerPass[curPass]=totalBitsUsed-bitPerPass*(totalNumPass-1);
	//bitUsedPerPass[0]=6;bitUsedPerPass[1]=6;bitUsedPerPass[2]=5;
	for(curPass=0;curPass<totalNumPass;curPass++)
		printf("P%d, %d; ", curPass, bitUsedPerPass[curPass]);
	printf("\n");
	int moveRightBits=startBits;
	int numBound=1;
//	int totalSize=rLen*sizeof(Record);
	int totalBoundSize=sizeof(Record)*(1<<totalBitsUsed);
	int totalBoundFlagSize=sizeof(int)*(1<<totalBitsUsed)*2;
	Record* d_iBound;
	GPUMALLOC((void**)&d_iBound, totalBoundSize);
	Record* h_iBound;
	CPUMALLOC((void**)&h_iBound, totalBoundSize);
	h_iBound[0].x=0;
	h_iBound[0].y=rLen;
	TOGPU(d_iBound, h_iBound, sizeof(Record));
	Record* d_oBound;
	GPUMALLOC((void**)&d_oBound, totalBoundSize);
	Record* h_oBound;
	CPUMALLOC((void**)&h_oBound, totalBoundSize);
	int* d_boundFlag;
	GPUMALLOC((void**)&d_boundFlag, totalBoundFlagSize);
	int* d_boundFlagSum;
	GPUMALLOC((void**)&d_boundFlagSum, totalBoundFlagSize);

	int* d_extra;
	GPUMALLOC((void**)&d_extra, rLen*sizeof(int));
	int *d_pidArray;
	GPUMALLOC((void**)&d_pidArray, rLen*sizeof(int));
	//Record *d_pingpong;
	//GPUMALLOC((void**)&d_pingpong, totalSize);
	int numPS=getMaxNumPS(rLen,bitUsedPerPass, totalNumPass);//(1<<totalBitsUsed)*128;//*4096;
	int* d_Hist;
	GPUMALLOC((void**)&d_Hist, sizeof(int)*numPS);
	int* d_psSum;
	GPUMALLOC((void**)&d_psSum, sizeof(int)*numPS);
	//prefix sum initialization!!!
	int *d_loc;
	GPUMALLOC((void**)&d_loc, sizeof(int)*rLen);

	int resultPart=0;
	int threshold=512;
	for(curPass=0;curPass<totalNumPass;curPass++)
	{
		moveRightBits=moveRightBits-bitUsedPerPass[curPass];
		if((curPass&1)==0)
		{			
			resultPart=radixPart(d_R, d_pidArray, d_loc,d_Hist,d_psSum, rLen, numPS, moveRightBits, bitUsedPerPass[curPass], totalBitsUsed, 
				d_iBound, numBound, d_extra, d_oBound, d_S);
			//computeLargeBound(d_iBound, numBound, threshold, d_boundFlag, d_boundFlagSum, d_oBound);
		}
		else
		{
			resultPart=radixPart(d_S, d_pidArray, d_loc,d_Hist,d_psSum, rLen, numPS, moveRightBits, bitUsedPerPass[curPass], totalBitsUsed, 
				d_oBound, numBound, d_extra, d_iBound, d_R);
			//computeLargeBound(d_oBound, numBound, threshold, d_boundFlag, d_boundFlagSum,d_iBound);
		}
		numBound=resultPart;
	}
	cudaThreadSynchronize();
	int numLargeBound=0;//the ones larger than 512;
	if((curPass&1)==0)
	{
		numLargeBound=computeLargeBound(d_iBound, numBound, threshold, d_boundFlag, d_boundFlagSum,d_oBound);
		cudaThreadSynchronize();
		unsigned int timer=0;
		startTimer(&timer);
		bitonicSortMultipleBlocks(d_R, d_oBound+numLargeBound, numBound-numLargeBound, d_S);
		endTimer("bitonicSortMultipleBlocks1", &timer);
		if(numLargeBound>0)
		{
			startTimer(&timer);
			//sortLargeChunks(d_R, d_oBound, numLargeBound, d_S);
			bitonicSortMultipleLargeBlocks(d_R, d_oBound, numLargeBound, d_S);
			endTimer("sortLargeChunks 1", &timer);
		}
	}
	else
	{
		numLargeBound=computeLargeBound(d_oBound, numBound, threshold, d_boundFlag, d_boundFlagSum, d_iBound);
		cudaThreadSynchronize();
		unsigned int timer=0;
		startTimer(&timer);
		bitonicSortMultipleBlocks(d_S, d_iBound+numLargeBound, numBound-numLargeBound, d_S);
		endTimer("bitonicSortMultipleBlocks2", &timer);
		if(numLargeBound>0)
		{
			startTimer(&timer);
			//sortLargeChunks(d_S, d_iBound, numLargeBound, d_S);
			bitonicSortMultipleLargeBlocks(d_S, d_iBound, numLargeBound, d_S);
			endTimer("sortLargeChunks 2", &timer);
		}
	}
	//if((curPass&1)==0)
	//	GPUTOGPU(d_S, d_R, totalSize);
	//else
	//	GPUTOGPU(d_S, d_pingpong, totalSize);
	GPUFREE(d_R);
	//GPUFREE(d_pingpong);
	GPUFREE(d_iBound);
	GPUFREE(d_oBound);
	GPUFREE(d_extra);
	GPUFREE(d_pidArray);
	GPUFREE(d_Hist);
	GPUFREE(d_psSum);
	GPUFREE(d_loc);
	GPUFREE(d_boundFlag);
	GPUFREE(d_boundFlagSum);
}





#endif
