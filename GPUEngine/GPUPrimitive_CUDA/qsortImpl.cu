#ifndef QSORT_IMPL_CU
#define QSORT_IMPL_CU


#include "splitImpl.cu"
#include "pickBound.cu"
#include "bitonicProc.cu"
#include "CSSTree.cu"
#include "GPU_Dll.h"

//return the number of sample levels.
void getQSortPivot(Record* d_Rin, int rLen, int sampleLevel, Record* d_pivot, Record *d_sampleOut)
{
	int numSample=(1<<sampleLevel);
	Record *h_sampleRin;
	CPUMALLOC((void**)&h_sampleRin, sizeof(Record)*numSample);

	int segSize=rLen/numSample;
	/*int i=0;
	int tempIndex=0;
	for(i=0;i<rLen && tempIndex<numSample;i=i+segSize)
	{
		h_sampleRin[tempIndex]=h_Rin[i+(rand()%256)];
		tempIndex++;
	}
	assert(tempIndex==numSample);*/

	Record *d_sampleRin;
	GPUMALLOC((void**)&d_sampleRin, sizeof(Record)*numSample);

	int* h_loc1;
	int* d_loc1;
	CPUMALLOC( (void**)&h_loc1, sizeof(int)*numSample );
	GPUMALLOC( (void**)&d_loc1, sizeof(int)*numSample );
	int tempIndex=0;
	for(int i = 0; i < rLen && tempIndex < numSample; i = i + segSize )
	{
		h_loc1[tempIndex]=i+(rand()%256);
		tempIndex++;
	}
	assert(tempIndex==numSample);
	TOGPU( d_loc1, h_loc1, sizeof(int)*numSample );
	gatherImpl(d_Rin, rLen, d_loc1,d_sampleRin, numSample);
	GPUFREE( d_loc1 );
	CPUFREE( h_loc1 );

	//void gatherImpl(Record *d_R, int rLen, int *d_loc, Record *d_S, int sLen, int numThreadsPerBlock_x = 32, int numBlock_x = 64)


	//TOGPU( d_sampleRin, h_sampleRin, numSample*sizeof(Record));

	sortImpl(d_sampleRin, numSample, d_sampleOut);
	GPUFREE(d_sampleRin);
	//copy to sampleInt
	int j=0;
	int startIndex=0, delta=0;
	numSample=numSample-1;
	tempIndex=0;
	int* h_loc;
	CPUMALLOC((void**)&h_loc, sizeof(int)*numSample);
	int* d_loc;
	GPUMALLOC((void**)&d_loc, sizeof(int)*numSample);
	for(int i=0;i<sampleLevel;i++)
	{
		delta=1<<(sampleLevel-i);
		startIndex=numSample>>(i+1);		
		for(j=startIndex;j<numSample;j=j+delta)
		{
			//sampleRin[tempIndex]=(*Rout)[j];
			h_loc[j]=tempIndex;
			tempIndex++;
		}
	}
	//printString(rawData, sampleRin, numSample);	
	TOGPU( d_loc, h_loc, numSample*sizeof(int));
	scatterImpl(d_sampleOut, numSample, d_loc, d_pivot);
	//gpuPrintInt2(d_sampleOut, numSample, "d_sampleOut");
	//gpuPrintInt2(d_pivot, numSample, "d_pivot");
	//GPUFREE(d_sampleOut);
	GPUFREE(d_loc);
	CPUFREE(h_loc);

}

int getQSortMaxNumPS(int rLen, int* bitUsedPerPass, int totalNumPass)
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

//kernels.
/*
for computing the partition ID according to the radix.
*/
__device__ int getQSortPartID(Record rec, Record records[], int level)
{
	int i=0;
	int pid=0;
	int next=0;
	int base=(1<<level)-1;
	for(i=0;i<level;i++)
		if(rec.y<records[next].y)
		{
			next=(next<<1)+1;
		}
		else if(rec.y>records[next].y)
		{
			next=(next<<1)+2;
		}
		else//=
		{
			next=(next<<1)+(rec.x&1)+1;
		}
	pid=next-base;
	return pid;
}

__global__ void 
mapQSort_kernel(Record *d_R, int numThreadPerPartition, int rLen, Record* d_iBound, int numBound, 
				Record *d_pivot, int startLevel, int level, 
				int *d_extra, int *d_output)  
{
	extern __shared__ Record pivots[];// the last one is used as the boundary. (1<<level).
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
//	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	if(tid==0)
	{
		pivots[1<<level]=d_iBound[by];
	}
	//load the tree.
	int i=0;
	int rootbase=(1<<(startLevel))-1+by;
	int base=rootbase;
	for(i=0;i<level;i++)
	{
		if(tid<(1<<i))
		{
			pivots[(1<<i)-1+tid]=d_pivot[base+tid];
		}
		base=(base<<1)+1;
	}
	__syncthreads();
	int start=pivots[1<<level].x;
	int end=pivots[1<<level].y;
	const int resultID=(bx)*numThread+tid;
	Record tempValue;
	for(int pos=resultID+start;pos<end;pos+=numThreadPerPartition)
	{
		d_extra[pos]=d_R[pos].x;
		tempValue=d_R[pos];
		d_output[pos]=getQSortPartID(tempValue, pivots, level);
	}	
}

void mapQSort(Record *d_R, int rLen, Record* d_iBound, int numBound, int expectedLength,
			  Record *d_pivot, int startLevel, int level,
			  int *d_S, int *d_extra)
{
	//int *d_extra;
	//GPUMALLOC((void **)&d_extra,rLen*sizeof( int));
	int numThreadsPerBlock_x=128; //default 128
	int numThreadsPerBlock_y=1;
	int numBlock_x=log2Ceil(expectedLength/numThreadsPerBlock_x);
	int numBlock_y=numBound;
	int numThreadPerPartition=numBlock_x*numThreadsPerBlock_x;
	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	mapQSort_kernel<<<grid,thread, ((1<<level)+1)*sizeof(Record)>>>(d_R, numThreadPerPartition, rLen, d_iBound, numBound,
		d_pivot, startLevel, level, 
		d_extra, d_S);
	
}




//procedures.
int QSortPart(Record *d_R, int* d_pidArray, int *d_loc, int* d_Hist, int* d_psSum,
			  int rLen, int numPS, Record *d_pivot, int startLevel, int level,
			   Record* d_iBound, int numBound, int* d_extra, 
			   Record* d_oBound, Record *d_S)
{
	int numPart=(1<<level);
	int expectedLength=rLen/numBound;	
	mapQSort(d_R, rLen, d_iBound, numBound, expectedLength, d_pivot, startLevel, level, d_pidArray, d_extra);
	//gpuPrint(d_pidArray, rLen, "d_pidArray");
	int resultPart=splitWithPIDArray(d_R, d_pidArray,d_loc,d_Hist,d_psSum, rLen,numPS, d_iBound, numBound, numPart, expectedLength, d_S, d_oBound);
	return resultPart;
}


void QSort(Record *d_R, int rLen, Record* d_pivot, int totalLevel, int levelPerPass, Record* d_S)
{
	int totalNumPass=totalLevel/levelPerPass;
	if(totalLevel%levelPerPass!=0)
		totalNumPass++;
	int *levelUsedPerPass;
	CPUMALLOC((void**)&levelUsedPerPass, totalNumPass*sizeof(int));
	int curPass=0;
	for(curPass=0;curPass<totalNumPass-1;curPass++)
		levelUsedPerPass[curPass]=levelPerPass;
	//the last pass
	levelUsedPerPass[curPass]=totalLevel-levelPerPass*(totalNumPass-1);
	//bitUsedPerPass[0]=6;bitUsedPerPass[1]=6;bitUsedPerPass[2]=5;
	for(curPass=0;curPass<totalNumPass;curPass++)
		printf("P%d, %d; ", curPass, levelUsedPerPass[curPass]);
	printf("\n");
	
	int numBound=1;
	//int totalSize=rLen*sizeof(Record);
	int totalBoundSize=sizeof(Record)*(1<<totalLevel);
	int totalBoundFlagSize=sizeof(int)*(1<<totalLevel)*2;
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
	int numPS=getMaxNumPS(rLen,levelUsedPerPass, totalNumPass);//(1<<totalBitsUsed)*128;//*4096;
	int* d_Hist;
	GPUMALLOC((void**)&d_Hist, sizeof(int)*numPS);
	int* d_psSum;
	GPUMALLOC((void**)&d_psSum, sizeof(int)*numPS);
	//prefix sum initialization!!!
	int *d_loc;
	GPUMALLOC((void**)&d_loc, sizeof(int)*rLen);

	int resultPart=0;
	int threshold=512;
	int curLevel=0;
	for(curPass=0;curPass<totalNumPass;curPass++)
	{
		printf("curPass, %d, numPart, %d\n", curPass, numBound);
		if((curPass&1)==0)
		{			
			resultPart=QSortPart(d_R, d_pidArray, d_loc,d_Hist,d_psSum, rLen, numPS, 
				d_pivot, curLevel, levelUsedPerPass[curPass], 
				d_iBound, numBound, d_extra, d_oBound, d_S);
			//gpuPrintInt2(d_oBound, resultPart, "d_oBound2");
			//computeLargeBound(d_iBound, numBound, threshold, d_boundFlag, d_boundFlagSum, d_oBound);
		}
		else
		{
			resultPart=QSortPart(d_S, d_pidArray, d_loc,d_Hist,d_psSum, rLen, numPS, 
				d_pivot, curLevel, levelUsedPerPass[curPass],  
				d_oBound, numBound, d_extra, d_iBound, d_R);
			//gpuPrintInt2(d_iBound, resultPart, "d_iBound2");
			//computeLargeBound(d_oBound, numBound, threshold, d_boundFlag, d_boundFlagSum,d_iBound);
		}
		numBound=resultPart;
		curLevel+=levelUsedPerPass[curPass];
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

void gpuQSort(Record *d_Rin, int rLen, Record* d_Rout)
{
	//determine the number of levels to used.
	int totalBitsUsed=log2Ceil(rLen)-7;
	int levelPerPass=0;
	if(totalBitsUsed>15)//larger than 15
		levelPerPass=6;
	else
		if(totalBitsUsed>12)//(12,15]
			levelPerPass=5;
		else
			if(totalBitsUsed>=12)//[6,12]
				levelPerPass=6;
			else //if(totalBitsUsed>=8)
				levelPerPass=5;
	printf("QS sampleLevel, %d, bitPerPass, %d\n", totalBitsUsed, levelPerPass);
	
	//get the pivot.
	Record *d_pivot;
	int sampleLevel=totalBitsUsed;
	int numSample=(1<<sampleLevel);
	GPUMALLOC((void**)&d_pivot, numSample*sizeof(Record));
	Record *d_sampleOut;
	GPUMALLOC((void**)&d_sampleOut, sizeof(Record)*numSample);

	getQSortPivot(d_Rin, rLen, sampleLevel, d_pivot,d_sampleOut);

	//for debug
	Record* h_pivot = (Record*)malloc( sizeof(Record)*numSample );
	FROMGPU( h_pivot, d_pivot, sizeof(Record)*numSample );
	
    QSort(d_Rin,rLen,d_pivot, sampleLevel, levelPerPass,  d_Rout);
	
	GPUFREE(d_pivot);

}

extern "C"
void GPUCopy_QuickSort( Record* h_Rin, int rLen, Record* h_Rout )
{
	unsigned int memSize = sizeof(Record)*rLen;

	Record* d_Rin;
	Record* d_Rout;
	GPUMALLOC( (void**)&d_Rin, memSize );
	GPUMALLOC( (void**)&d_Rout, memSize );

	TOGPU( d_Rin, h_Rin, memSize );

	GPUOnly_QuickSort( d_Rin, rLen, d_Rout);

	FROMGPU( h_Rout, d_Rout, memSize );

	GPUFREE( d_Rin );
	GPUFREE( d_Rout );
}

extern "C"
void GPUOnly_QuickSort( Record* d_Rin, int rLen, Record* d_Rout )
{
	gpuQSort( d_Rin, rLen, d_Rout);
}

#endif
