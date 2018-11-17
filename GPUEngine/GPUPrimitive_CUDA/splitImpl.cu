#ifndef SPLIT_IMPL_CU
#define SPLIT_IMPL_CU

#include "splitImpl_kernel.cu"
#include "GPU_Dll.h"


void mapPart(Record *d_R, int rLen, int numPart, int *d_S, int SPLIT_TYPE )
{
	int *d_extra;
	GPUMALLOC((void **)&d_extra,rLen*sizeof(int));
	int numThreadsPerBlock_x=256;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int numThread=numBlock_x*numThreadsPerBlock_x;
	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	if( SPLIT_TYPE == SPLIT )
	{
		mapPart_kernel<<<grid,thread>>>(d_R, numThread, rLen, numPart, d_extra, d_S);
	}
	else if( SPLIT_TYPE == PARTITION )
	{
		partition_kernel<<<grid,thread>>>(d_R, numThread, rLen, numPart, d_extra, d_S);
	}
	else
	{
		printf( "split type error! \n" ); 
	}
	
	GPUFREE(d_extra);
}




void getBound(int *d_psSum, int interval, int rLen, int numPart, Record* d_bound)
{
	if(d_bound!=NULL)
	{
		int numThreadsPerBlock_x=32;
		int numThreadsPerBlock_y=1;
		int numBlock_x=ceil((double)numPart/(double)numThreadsPerBlock_x);
		int numBlock_y=1;
		dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
		dim3  grid( numBlock_x, numBlock_y , 1);
		getBound_kernel<<<grid,thread>>>(d_psSum, interval, rLen, numPart, d_bound);
	}
}


void getBound_MB(int *d_psSum, int interval, int rLen, int numPart, Record* d_bound)
{
	if(d_bound!=NULL)
	{
		int numThreadsPerBlock_x=256;
		int numThreadsPerBlock_y=1;
		int numBlock_x=ceil((double)numPart/(double)numThreadsPerBlock_x);
		int numBlock_y=1;
		dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
		dim3  grid( numBlock_x, numBlock_y , 1);
		getBound_MB_kernel<<<grid,thread>>>(d_psSum, interval, rLen, numPart, d_bound);
	}
}

void splitImpl(Record *d_R, int rLen, int numPart, Record *d_S, Record* d_bound,
			   int numThreadPB=-1, int numThreadBlock=-1, int SPLIT_TYPE = SPLIT )
{
#ifdef SHARED_MEM
	printf("YES, SHARED_MEM, SPLIT\n");
#else
	printf("NO, SHARED_MEM, SPLIT\n");
#endif

#ifdef COALESCED
	printf( "YES, COALSESCED, SPLIT" );
#else
	printf( "NO COALESCED, SPLIT\n" );
#endif
	//map->pid
	unsigned int timer=0;
	startTimer(&timer);
	int *d_pidArray;
	GPUMALLOC((void**)&d_pidArray, sizeof(int)*rLen);
	mapPart(d_R, rLen, numPart, d_pidArray, SPLIT_TYPE);
	endTimer("with split: mapPart",&timer);

	//pid->write loc
	int numThreadsPerBlock_x=0;
	if(numThreadPB==-1)
		numThreadsPerBlock_x=1<<(log2((int)(SHARED_MEMORY_PER_PROCESSOR/(numPart*sizeof(int)))));
	else
		numThreadsPerBlock_x=numThreadPB;
	if(numThreadsPerBlock_x>256)
		numThreadsPerBlock_x=256;
#ifdef SHARED_MEM
	int sharedMemSize=numThreadsPerBlock_x*numPart*sizeof(int);
#else
	int sharedMemSize=2*sizeof(int);
#endif
	assert(numThreadsPerBlock_x>=16);
	int numThreadsPerBlock_y=1;
	int numBlock_x;
	if(numThreadBlock==-1)
		numBlock_x=512;
	else
		numBlock_x=numThreadBlock;
	printf("numThreadsPerBlock_x, %d,sharedMemSize, %d,  numBlock_x, %d\n", numThreadsPerBlock_x, sharedMemSize,numBlock_x);
	int numBlock_y=1;
	int numThread=numBlock_x*numThreadsPerBlock_x;
	int numInPS=numThread*numPart;
	int* d_Hist;
	GPUMALLOC((void**)&d_Hist, sizeof(int)*numInPS);
	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	startTimer(&timer);
	countHist_kernel<<<grid, thread,sharedMemSize>>>(d_pidArray, numThread, rLen, numPart, d_Hist);
	endTimer("with split: countHist_kernel",&timer);
	//prefix sum
	int* d_psSum;
	GPUMALLOC((void**)&d_psSum, sizeof(int)*numInPS);
	startTimer(&timer);
	scanImpl(d_Hist, numInPS, d_psSum);
	endTimer("with split: scanImpl",&timer);

	GPUFREE(d_Hist);
	int *d_loc;
	GPUMALLOC((void**)&d_loc, sizeof(int)*rLen);

	startTimer(&timer);
	writeHist_kernel<<<grid, thread,sharedMemSize>>>(d_pidArray, numThread, rLen, numPart, d_psSum, d_loc);
	endTimer("with split: writeHist_kernel",&timer);

	getBound(d_psSum, numThread, rLen, numPart,d_bound);
	GPUFREE(d_pidArray);	
	GPUFREE(d_psSum);
	//scatter
	startTimer(&timer);
	scatterImpl_forPart(d_R, rLen, numPart, d_loc, d_S);
	endTimer("with split: scatterImpl_forPart",&timer);
}




/*
the input is partition ID array.
*/


int splitWithPIDArray(Record* d_R, int *d_pidArray, int *d_loc, int* d_Hist, int* d_psSum,
					  int rLen, int numPS, Record* d_iBound, int numBound,
					   int numPart, int expectedLength, 
					   Record *d_S, Record* d_oBound)
{
#ifdef SHARED_MEM
	printf("YES, SHARED_MEM in splitWithPIDArray\n");
#else
	printf("NO, SHARED_MEM in splitWithPIDArray\n");
#endif
	//pid->write loc
	int numThreadsPerBlock_x=1<<(log2((int)(SHARED_MEMORY_PER_PROCESSOR/(numPart*sizeof(int)))));
	if(numThreadsPerBlock_x>256)
		numThreadsPerBlock_x=256;
	if(numThreadsPerBlock_x>rLen/numBound)
		numThreadsPerBlock_x=1<<(log2(rLen/numBound));
	numThreadsPerBlock_x=(numThreadsPerBlock_x>32)? numThreadsPerBlock_x: 32;
	//numThreadsPerBlock_x=(numThreadsPerBlock_x>8)? numThreadsPerBlock_x: 8;
	//assert(numThreadsPerBlock_x>=32);
	int sharedMemSize=(numThreadsPerBlock_x*numPart+2)*sizeof(int);
	int numThreadsPerBlock_y=1;
	int numTuplePerThread=128;
	int numBlock_x=ceil((double)expectedLength/(double)(numThreadsPerBlock_x*numTuplePerThread));
	int numBlock_y=numBound;
	int numThreadPerPartition=numBlock_x*numThreadsPerBlock_x;
	int numInPS=numPart*numThreadPerPartition*numBlock_y;
	
	printf("expectedLength, %d, numThreadsPerBlock_x, %d, numBlock_x, %d, numBlock_y, %d, sharedMemSize, %d,  numInPS, %d, original numPS, %d\n",expectedLength, numThreadsPerBlock_x, numBlock_x, numBlock_y, sharedMemSize, numInPS, numPS);
	assert(numInPS<=numPS);
	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	unsigned int timer=0;
	startTimer(&timer);
	countHist_MB_kernel<<<grid, thread,sharedMemSize>>>(d_pidArray, d_iBound, numPart, d_Hist);
	endTimer("with split: countHist_kernel",&timer);
	//prefix sum
	//saven_initialPrefixSum(numPS);
	startTimer(&timer);
	scanImpl(d_Hist, numInPS, d_psSum);
	endTimer("with split: scanImpl",&timer);

	startTimer(&timer);
	writeHist_MB_kernel<<<grid, thread,sharedMemSize>>>(d_pidArray, d_iBound, numPart, d_psSum, d_loc);
	endTimer("with split: writeHist_kernel",&timer);
	
	int resultPart=numBound*numPart;
	getBound_MB(d_psSum, numThreadPerPartition, rLen, resultPart,d_oBound);
	
	//scatter
	startTimer(&timer);
	scatterImpl_forPart(d_R, rLen, numPart, d_loc, d_S);
	endTimer("with split: scatterImpl_forPart",&timer);
	return resultPart;
}

__global__
void boundToStartHist_kernel( Record* d_bound, int numPart, int* d_startHist )
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int gridSize = blockDim.x*gridDim.x;

	for( int idx = bx*blockDim.x + tx; idx < numPart; idx += gridSize )
	{
		d_startHist[idx] = d_bound[idx].x;
	}
}

extern "C"
void GPUOnly_Partition( Record* d_Rin, int rLen, int numPart, Record* d_Rout, int* d_startHist,
														  int numThreadPB, int numBlock)
{	
	Record* d_bound;
	GPUMALLOC( (void**)&d_bound, sizeof(Record)*numPart );

	splitImpl( d_Rin, rLen, numPart, d_Rout, d_bound,
			   numThreadPB, numBlock, PARTITION );

	boundToStartHist_kernel<<<16, 32>>>( d_bound, numPart, d_startHist );

	GPUFREE( d_bound );
}

extern "C"
void GPUCopy_Partition( Record* h_Rin, int rLen, int numPart, Record* h_Rout, int* h_startHist, 
														  int numThreadPB, int numBlock )
{
	Record* d_Rin;
	Record* d_Rout;
	int* d_startHist;

	GPUMALLOC( (void**)&d_Rin, sizeof(Record)*rLen );
	GPUMALLOC( (void**)&d_Rout, sizeof(Record)*rLen );
	GPUMALLOC( (void**)&d_startHist, sizeof(int)*numPart );

	TOGPU( d_Rin, h_Rin, sizeof(Record)*rLen );

	GPUOnly_Partition( d_Rin, rLen, numPart, d_Rout, d_startHist, numThreadPB, numBlock);

	FROMGPU( h_Rout, d_Rout, sizeof(Record)*rLen );
	FROMGPU( h_startHist, d_startHist, sizeof(int)*numPart );

	GPUFREE( d_Rin );
	GPUFREE( d_Rout );
	GPUFREE( d_startHist );
}

#endif
