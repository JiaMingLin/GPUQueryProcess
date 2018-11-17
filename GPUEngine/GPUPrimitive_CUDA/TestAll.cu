#include <cutil.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <GPUPrimitive_Def.cu>
#include <QP_Utility.cu>
#include <mapImpl.cu>
#include <scanImpl.cu>
#include <common.cu>
#include <scatterImpl.cu>
#include <sortImpl.cu>
#include <splitImpl.cu>
#include <radixSortImpl.cu>
#include <qsortImpl.cu>
#include <filterImpl.cu>
#include <reduceImpl.cu>
#include <groupByImpl.cu>
#include <aggAfterGroupBy.cu>
#include <projectionImpl.cu>

#include "StringGen.cu"
#include "string_quicksort.cu"
#include "GPUDB_Memory.cu"
#include "TableOpImpl.cu"

void testCopy(int rLen, int chunkSize)
{
	int result=0;
	int memSize=sizeof(Record)*rLen;
	Record *Rin;
	CPUMALLOC((void**)&Rin, memSize);
	generateRand(Rin,TEST_MAX,rLen,0);
	Record *d_Rin;
	GPUMALLOC((void**)&d_Rin, memSize);
	
	int numChunk=rLen/chunkSize;
	int i=0;
	startTime();
	//copy to
	unsigned int timer=0;
	startTimer(&timer);
	for(i=0;i<numChunk;i++)
		TOGPU(d_Rin+i*chunkSize, Rin+i*chunkSize, chunkSize*sizeof(Record));
	double processingTime=endTimer("copy to GPU",&timer);
	
	//copy back
	startTimer(&timer);
	for(i=0;i<numChunk;i++)
		FROMGPU(Rin+i*chunkSize, d_Rin+i*chunkSize, chunkSize*sizeof(Record));
	endTimer("copy back", &timer);
	double sec=endTime("copy");
	printf("rLen, %d, chunkSize, %d, KB result, %d\n", rLen, chunkSize*sizeof(Record)/1024, result);	
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	double kernel_bandwidth=dataSize/processingTime;
	printf("\n dataSize, %f MB, bandwidth, %f, kernel_bandwidth, %f\n", dataSize, bandwidth, kernel_bandwidth);
	CPUFREE(Rin);
	GPUFREE(d_Rin);
	
}

void testSelection( int rLen, int numThread = 64, int numBlock = 512 )
{
	Record* h_Rin;
	Record* h_Rout;
	int memSize = sizeof(Record)*rLen;
	CPUMALLOC( (void**)&h_Rin, memSize );
	generateRand( h_Rin, TEST_MAX, rLen, 0 );

	Record* d_Rin;
	GPUMALLOC( (void**)&d_Rin, memSize );
	Record* d_Rout;

	int rangeSmallKey = TEST_MAX/100;
	int rangeLargeKey = TEST_MAX/50;

	unsigned int timer = 0;
	//copy to
	startTimer( &timer );
	TOGPU( d_Rin, h_Rin, memSize );
	endTimer( "copy to gpu", &timer );

	startTimer( &timer );
	startTime();
	int numResult = GPUOnly_RangeSelection(d_Rin, rLen, rangeSmallKey, rangeLargeKey, &d_Rout, 
						   numThread, numBlock );
	double sec = endTime( "selection" );
	double processingTime = endTimer( "selection", &timer );

	//copy back
	startTimer( &timer );
	CPUMALLOC( (void**)&h_Rout, sizeof(Record)*numResult );
	FROMGPU( h_Rout, d_Rout, sizeof(Record)*numResult );
	endTimer( "copy back", &timer );

	validateFilter( h_Rin, 0, rLen, h_Rout, numResult, rangeSmallKey, rangeLargeKey );

	printf( "selectivity %f\n", (numResult)/(float)rLen );
	printf("rLen, %d, numResult, %d, numThread, %d, numBlock, %d \n", rLen, numResult, numThread,numBlock );	
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	double kernel_bandwidth=dataSize/processingTime;
	printf("\n dataSize, %f MB, bandwidth, %f, kernel_bandwidth, %f\n", dataSize, bandwidth, kernel_bandwidth);

	CPUFREE(h_Rin);
	CPUFREE(h_Rout);
	GPUFREE(d_Rin);
	GPUFREE(d_Rout);
}

void testProjection( int rLen, int pLen, int numThread = 32, int numBlock = 64 ) 
{
	printf( "TEST PROJECTION\n" );
	Record* h_Rin;
	Record* h_projTable;
	Record* originalProjTable; //for check
	CPUMALLOC( (void**)&h_Rin, sizeof(Record)*rLen );
	CPUMALLOC( (void**)&h_projTable, sizeof(Record)*pLen );
	CPUMALLOC( (void**)&originalProjTable, sizeof(Record)*pLen );

	generateRand( h_Rin, TEST_MAX, rLen, 0 );
	for( int i = 0; i < pLen; i++ )
	{
		h_projTable[i].x = rand()%rLen;
		originalProjTable[i].x = h_projTable[i].x;
	}

	unsigned int timer = 0;

	startTimer( &timer );
	Record* d_Rin;
	Record* d_projTable;
	GPUMALLOC( (void**)&d_Rin, sizeof(Record)*rLen );
	GPUMALLOC( (void**)&d_projTable, sizeof(Record)*pLen );
	TOGPU( d_Rin, h_Rin, sizeof(Record)*rLen );
	TOGPU( d_projTable, h_projTable, sizeof(Record)*pLen );
	endTimer( "copy to gpu", &timer );

	startTimer( &timer );
	GPUOnly_Projection( d_Rin, rLen, d_projTable, pLen, numThread, numBlock );
	endTimer( "projection", &timer );

	startTimer( &timer );
	FROMGPU( h_projTable, d_projTable, sizeof(Record)*pLen );
	endTimer( "copy back", &timer );

	validateProjection( h_Rin, rLen, originalProjTable, h_projTable, pLen );

	printf( "rLen: %d, pLen: %d, numThread = %d, numBlock = %d\n", rLen, pLen, numThread, numBlock );

	CPUFREE( h_Rin );
	CPUFREE( h_projTable );
	CPUFREE( originalProjTable );
	GPUFREE( d_Rin );
	GPUFREE( d_projTable );
}

void testAggAfterGroupByImpl( int rLen, int OPERATOR, int numThread = 64, int numBlock = 1024 )
{
	int memSize = sizeof(Record)*rLen;

	Record* h_Rin;
	Record* h_Rout;
	Record* h_Sin;
	Record* h_Sout;
	Record* d_Rin;
	Record* d_Rout;
	Record* d_Sin;
	Record* d_Sout;
	CPUMALLOC( (void**)&h_Rin, memSize );
	CPUMALLOC( (void**)&h_Rout, memSize );
	CPUMALLOC( (void**)&h_Sin, memSize );
	CPUMALLOC( (void**)&h_Sout, memSize );
	GPUMALLOC( (void**)&d_Rin, memSize );
	GPUMALLOC( (void**)&d_Rout, memSize );
	GPUMALLOC( (void**)&d_Sin, memSize );
	GPUMALLOC( (void**)&d_Sout, memSize );

	generateRand( h_Rin, 50, rLen, 0 );
	generateRand( h_Sin, TEST_MAX, rLen, 0 );

	unsigned int timer = 0;
	//copy to
	startTimer(&timer);
	TOGPU( d_Rin, h_Rin, memSize );
	TOGPU( d_Sin, h_Sin, memSize );
	endTimer( "copy to GPU", &timer );

	//group by
	startTimer(&timer);
	int numGroup = 0;
	int* d_startPos;
	numGroup = groupByImpl(d_Rin, rLen, d_Rout, &d_startPos, numThread, numBlock);
	endTimer( "group bgy", &timer );

	FROMGPU( h_Rout, d_Rout, memSize );
	int* h_startPos = (int*)malloc( sizeof(int)*numGroup );
	CPUMALLOC( (void**)&h_startPos, sizeof(int)*numGroup );
	FROMGPU( h_startPos, d_startPos, sizeof(int)*numGroup );
	validateGroupBy( h_Rin, rLen, h_Rout, h_startPos, numGroup );

	//aggregation after group by
	startTimer(&timer);
	int* d_aggResults;
	GPUMALLOC( (void**)&d_aggResults, sizeof(int)*numGroup );
	aggAfterGroupByImpl(d_Rout, rLen, d_startPos, numGroup, d_Sin, d_aggResults, OPERATOR, numThread);
	endTimer( "aggregration after group by", &timer );

	int* h_aggResults = (int*)malloc( sizeof(int)*numGroup );
	CUDA_SAFE_CALL( cudaMemcpy( h_aggResults, d_aggResults, sizeof(int)*numGroup, cudaMemcpyDeviceToHost ) );
	validateAggAfterGroupBy( h_Rin, rLen, h_startPos, numGroup, h_Sin, h_aggResults, OPERATOR );
}

void testGroupByImpl( int rLen, int numThread = 64, int numBlock = 1024 )
{
	int memSize = sizeof(Record)*rLen;

	Record* h_Rin;
	CPUMALLOC( (void**)&h_Rin, memSize );
	Record* h_Rout;
	CPUMALLOC( (void**)&h_Rout, memSize );
	generateRand( h_Rin, 64, rLen, 0 );

	Record* d_Rin;
	GPUMALLOC( (void**)&d_Rin, memSize );
	Record* d_Rout;
	GPUMALLOC( (void**)&d_Rout, memSize );

	unsigned int timer = 0;

	//copy to
	startTimer( &timer );
	TOGPU( d_Rin, h_Rin, memSize );
	endTimer( "copy to GPU", &timer );

	int numGroup = 0;
	int* d_startPos;

	//group by
	startTimer( &timer );
	startTime();
	numGroup = groupByImpl(d_Rin, rLen, d_Rout, &d_startPos, numThread, numBlock);
	double sec = endTime( "groupBy" );
	double processingTime = endTimer( "groupBy", &timer );

	//copy back
	startTimer( &timer );
	int* h_startPos = (int*)malloc( sizeof(int)*numGroup );
	FROMGPU( h_Rout, d_Rout, memSize );
	FROMGPU( h_startPos, d_startPos, sizeof(int)*numGroup );
	endTimer( "copy back", &timer );

	validateGroupBy( h_Rin, rLen, h_Rout, h_startPos, numGroup );

	printf("rLen, %d, numGroup, %d, numThreadPB, %d, numBlock, %d \n", rLen, numGroup, numThread, numBlock );	
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	double kernel_bandwidth=dataSize/processingTime;
	printf("\n dataSize, %f MB, bandwidth, %f, kernel_bandwidth, %f\n", dataSize, bandwidth, kernel_bandwidth);

	CPUFREE( h_Rin );
	CPUFREE( h_Rout );
	GPUFREE( d_Rin );
	GPUFREE( d_Rout );
	GPUFREE( d_startPos );
}

void testReduceImpl( int rLen, int OPERATOR = REDUCE_MAX, int numThreadPB = 256, int numMaxBlock = 1024 )
{
	int memSize = sizeof(Record)*rLen;

	Record* h_Rin;
	CPUMALLOC( (void**)&h_Rin, memSize );
	generateRand( h_Rin, TEST_MAX - 11111, rLen, 0 );

	Record* h_Rout;

	unsigned int numResult = 0;
	unsigned int timer = 0;


	Record* d_Rin;
	Record* d_Rout;

	GPUMALLOC( (void**)&d_Rin, memSize );

	//copy to gpu
	startTimer( &timer );
	TOGPU( d_Rin, h_Rin, memSize );
	endTimer( "copy to gpu", &timer );

	//reduce
	startTimer( &timer );
	startTime();
	numResult = GPUOnly_AggMax( d_Rin, rLen, &d_Rout, numThreadPB, numMaxBlock );
	double sec = endTime( "reduce" );
	double processingTime = endTimer( "reduce", &timer );

	//copy back
	startTimer( &timer );
	CPUMALLOC( (void**)&h_Rout, sizeof(Record)*numResult );
	FROMGPU( h_Rout, d_Rout, sizeof(Record)*numResult );
	endTimer( "copy back", &timer );

	validateReduce( h_Rin, rLen, h_Rout[0].y, OPERATOR );

	printf("rLen, %d, numThreadPB, %d, numMaxBlock, %d \n", rLen, numThreadPB, numMaxBlock );	
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	double kernel_bandwidth=dataSize/processingTime;
	printf("dataSize, %f MB, bandwidth, %f, kernel_bandwidth, %f\n", dataSize, bandwidth, kernel_bandwidth);
	printf( "numResult: %d\n", numResult );

	CPUFREE( h_Rin );
	CPUFREE( h_Rout );
}

void testFilterImpl( int rLen, int numThreadPB = 32, int numBlock = 256 )
{
	int beginPos = 0;
	int memSize = sizeof(Record)*rLen;
	
	Record* Rin;
	CPUMALLOC( (void**)&Rin, memSize );
	generateRand( Rin, 100, rLen, 0 );

	Record* Rout;

	Record* d_Rin;
	GPUMALLOC( (void**)&d_Rin, memSize );
	Record* d_Rout;

	int smallKey = rand()%100;
	int largeKey = smallKey;

	int* outSize = (int*)malloc( sizeof(int) );

	unsigned int timer = 0;

	//copy to
	startTimer( &timer );
	TOGPU( d_Rin, Rin, memSize );
	endTimer( "copy to GPU", &timer );

	//filter
	startTimer( &timer );
	startTime();
	filterImpl( d_Rin, beginPos, rLen, &d_Rout, outSize, 
				numThreadPB, numBlock, smallKey, largeKey );
	double sec = endTime( "filter" );
	double processingTime = endTimer( "filter", &timer );

	//copy back
	startTimer( &timer );
	CPUMALLOC( (void**)&Rout, sizeof(Record)*(*outSize) );
	FROMGPU( Rout, d_Rout, sizeof(Record)*(*outSize) );
	endTimer( "copy back", &timer );

	validateFilter( Rin, beginPos, rLen, Rout, *outSize, smallKey, largeKey );

	printf( "selectivity %f\n", (*outSize)/(float)rLen );
	printf("rLen, %d, numResult, %d, numThreadPB, %d, numBlock, %d \n", rLen, *outSize, numThreadPB,numBlock );	
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	double kernel_bandwidth=dataSize/processingTime;
	printf("\n dataSize, %f MB, bandwidth, %f, kernel_bandwidth, %f\n", dataSize, bandwidth, kernel_bandwidth);

	CPUFREE(Rin);
	CPUFREE(Rout);
	GPUFREE(d_Rin);
	GPUFREE(d_Rout);
}

void testMapImpl(int rLen, int numThreadPB = 64, int numBlock = 128)
{
	int result=0;
	int memSize=sizeof(Record)*rLen;
	int outSize=sizeof(int)*rLen;
	Record *Rin;
	CPUMALLOC((void**)&Rin, memSize);
	generateRand(Rin,TEST_MAX,rLen,0);
	int *Rout;
	CPUMALLOC((void**)&Rout, outSize);
	Record *d_Rin;
	GPUMALLOC((void**)&d_Rin, memSize);
	int *d_Rout;
	GPUMALLOC((void**)&d_Rout, outSize);
	int *d_Rout2;
	GPUMALLOC((void**)&d_Rout2, outSize);
	
	//copy to
	unsigned int timer=0;
	startTimer(&timer);
	TOGPU(d_Rin, Rin, memSize);
	endTimer("copy to GPU",&timer);
	
	//map
	startTimer(&timer);
	startTime();
	mapImpl_int(d_Rin,rLen,d_Rout, d_Rout2, numThreadPB, numBlock);
	double sec=endTime("map");
	double processingTime=endTimer("map",&timer);
	

	/*startTimer(&timer);
	Record value;
	value.x=value.y=-1;
	//mapInit(d_Rin,rLen/2, rLen,value);
	mapTest(d_Rin,rLen/2, rLen,value);
	processingTime=endTimer("mapInit",&timer);*/

	//copy back
	startTimer(&timer);
	FROMGPU(Rout, d_Rout, outSize);
	endTimer("copy back", &timer);
	
	//gpuPrint(d_Rout, rLen, "d_Rout");
	printf("rLen, %d, numThreadPB, %d, numBlock, %d, result, %d\n", rLen, numThreadPB,numBlock,result);	
	double dataSize=(double)(sizeof(Record)*rLen*2)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	double kernel_bandwidth=dataSize/processingTime;
	printf("\n dataSize, %f MB, bandwidth, %f,MB/sec, kernel_bandwidth, %f\n", dataSize, bandwidth, kernel_bandwidth);
	FROMGPU(Rin, d_Rin, memSize);
	CPUFREE(Rin);
	CPUFREE(Rout);
	GPUFREE(d_Rin);
	GPUFREE(d_Rout);
	GPUFREE(d_Rout2);
}

void testBitonicSort(int rLen, int numThreadPB = 128, int numBlock = 1024)
{
	int result=0;
	int memSize=sizeof(Record)*rLen;
	int outSize=sizeof(Record)*rLen;
	Record *Rin;
	CPUMALLOC((void**)&Rin, memSize);
	generateRand(Rin,TEST_MAX,rLen,0);
	Record *Rout;
	CPUMALLOC((void**)&Rout, outSize);
	Record *d_Rin;
	GPUMALLOC((void**)&d_Rin, memSize);
	Record *d_Rout;
	GPUMALLOC((void**)&d_Rout, outSize);

	
	//copy to
	unsigned int timer=0;
	startTimer(&timer);
	TOGPU(d_Rin, Rin, memSize);
	endTimer("copy to GPU",&timer);
	
	//sort
	startTimer(&timer);
	startTime();
	sortImpl(d_Rin,rLen,d_Rout,numThreadPB,numBlock);
	double sec=endTime("bitonicSort");
	double processingTime=endTimer("sort",&timer);

	//copy back
	startTimer(&timer);
	FROMGPU(Rout, d_Rout, outSize);
	endTimer("copy back", &timer);	
	validateSort(Rout, rLen);
	printf("rLen, %d, numThreadPB, %d, numBlock, %d, result, %d\n", rLen, numThreadPB,numBlock,result);	
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	double kernel_bandwidth=dataSize/processingTime;
	printf("\n dataSize, %f MB, bandwidth, %f, kernel_bandwidth, %f\n", dataSize, bandwidth, kernel_bandwidth);
	CPUFREE(Rin);
	CPUFREE(Rout);
	GPUFREE(d_Rin);
	GPUFREE(d_Rout);
	
}

void testPartition( int rLen, int numParts )
{
	Record* h_Rin;
	Record* h_Rout;
	int* h_startHist;


	CPUMALLOC( (void**)&h_Rin, sizeof(Record)*rLen );
	CPUMALLOC( (void**)&h_Rout, sizeof(Record)*rLen );
	CPUMALLOC( (void**)&h_startHist, sizeof(int)*numParts );

	generateRand( h_Rin, TEST_MAX, rLen, 0 );

	GPUCopy_Partition( h_Rin, rLen, numParts, h_Rout, h_startHist);

	validatePartition( h_Rout, rLen, numParts );
}

void testSplit(int rLen, int numPart, int numThreadPB = 32, int numThreadBlock = 1024)
{
	int result=0;
	int memSize=sizeof(Record)*rLen;
	int outSize=sizeof(Record)*rLen;
	Record *Rin;
	CPUMALLOC((void**)&Rin, memSize);
	Record *h_bound;
	CPUMALLOC((void**)&h_bound, numPart*sizeof(Record));
	generateRand(Rin,TEST_MAX,rLen,0);
	Record *Rout;
	CPUMALLOC((void**)&Rout, outSize);
	Record *d_Rin;
	GPUMALLOC((void**)&d_Rin, memSize);
	Record *d_Rout;
	GPUMALLOC((void**)&d_Rout, outSize);
	Record *d_bound;
	GPUMALLOC((void**)&d_bound, numPart*sizeof(Record));
	
	//copy to
	unsigned int timer=0;
	startTimer(&timer);
	TOGPU(d_Rin, Rin, memSize);
	endTimer("copy to GPU",&timer);
	
	//map
	startTimer(&timer);
	startTime();
	splitImpl(d_Rin,rLen,numPart, d_Rout, d_bound,numThreadPB,numThreadBlock);
	double sec=endTime("split");
	double processingTime=endTimer("split",&timer);

	//copy back
	startTimer(&timer);
	FROMGPU(Rout, d_Rout, outSize);
	FROMGPU(h_bound, d_bound, sizeof(Record)*numPart);
	endTimer("copy back", &timer);
	
	validateSplit(Rout, rLen, numPart);
	//gpuPrint(d_Rout, rLen, "d_Rout");
	printf("rLen, %d, numThread, %d, numBlock, %d, result, %d, numPart, %d\n", rLen, numThreadPB, numThreadBlock, result, numPart);	
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	double kernel_bandwidth=dataSize/processingTime;
	printf("\ndataSize, %f MB, bandwidth, %f, kernel_bandwidth, %f\n", dataSize, bandwidth, kernel_bandwidth);
	CPUFREE(Rin);
	CPUFREE(Rout);
	GPUFREE(d_Rin);
	GPUFREE(d_Rout);
}

void testScan(int rLen)
{
	int result=0;
	int memSize=sizeof(int)*rLen;
	int outSize=sizeof(int)*rLen;
	int *Rin;
	CPUMALLOC((void**)&Rin, memSize);
	generateRandInt(Rin, rLen,rLen,0);
	int *Rout;
	CPUMALLOC((void**)&Rout, outSize);
	int *d_Rin;
	GPUMALLOC((void**)&d_Rin, memSize);
	int *d_Rout;
	GPUMALLOC((void**)&d_Rout, outSize);
	
	
	//copy to
	unsigned int timer=0;
	startTimer(&timer);
	TOGPU(d_Rin, Rin, memSize);
	endTimer("copy to GPU",&timer);
	
	//map
	startTimer(&timer);
	startTime();
	scanImpl(d_Rin,rLen,d_Rout);
	double sec=endTime("prefix scan");
	double processingTime=endTimer("scan",&timer);

	//copy back
	startTimer(&timer);
	FROMGPU(Rout, d_Rout, outSize);
	endTimer("copy back", &timer);

	validateScan( Rin, rLen, Rout );
	
	//gpuPrint(d_Rout, rLen, "d_Rout");
	printf("rLen, %d, result, %d\n", rLen, result);	
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	double kernel_bandwidth=dataSize/processingTime;
	printf("\n dataSize, %f MB, bandwidth, %f, kernel_bandwidth, %f\n", dataSize, bandwidth, kernel_bandwidth);
	CPUFREE(Rin);
	CPUFREE(Rout);
	GPUFREE(d_Rin);
	GPUFREE(d_Rout);
}

void testScatter(int rLen, int numThreadPB=32, int numBlock=64)
{
	int result=0;
	int memSize=sizeof(Record)*rLen;
	int outSize=sizeof(Record)*rLen;
	Record *Rin;
	CPUMALLOC((void**)&Rin, memSize);
	generateRand(Rin,TEST_MAX,rLen,0);
	int *loc;
	CPUMALLOC((void**)&loc, sizeof(int)*rLen);
	generateRandInt(loc, rLen,rLen,0);
	Record *Rout;
	CPUMALLOC((void**)&Rout, outSize);
	Record *d_Rin;
	GPUMALLOC((void**)&d_Rin, memSize);
	Record *d_Rout;
	GPUMALLOC((void**)&d_Rout, outSize);
	int *d_loc;
	GPUMALLOC((void**)&d_loc, sizeof(int)*rLen);
	
	//copy to
	unsigned int timer=0;
	startTimer(&timer);
	TOGPU(d_Rin, Rin, memSize);
	TOGPU(d_loc, loc, sizeof(int)*rLen);
	endTimer("copy to GPU",&timer);
	
	//scatter
	startTimer(&timer);
	startTime();
	scatterImpl(d_Rin,rLen,d_loc,d_Rout,numThreadPB, numBlock);
	double sec=endTime("scatter");
	double processingTime=endTimer("scatter",&timer);

	//copy back
	startTimer(&timer);
	FROMGPU(Rout, d_Rout, outSize);
	endTimer("copy back", &timer);
	
	printf("rLen, %d, numThreadPB, %d, numBlock, %d, result, %d\n", rLen, numThreadPB,numBlock,result);	
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	double kernel_bandwidth=dataSize/processingTime;
	printf("\n dataSize, %f MB, bandwidth, %f, kernel_bandwidth, %f\n", dataSize, bandwidth, kernel_bandwidth);
	CPUFREE(Rin);
	CPUFREE(Rout);
	CPUFREE(loc);
	GPUFREE(d_Rin);
	GPUFREE(d_Rout);
	GPUFREE(d_loc);
}

void testGather(int rLen, int numThread = 32, int numBlock = 64)
{
	int result=0;
	int memSize=sizeof(Record)*rLen;
	int outSize=sizeof(Record)*rLen;
	Record *Rin;
	CPUMALLOC((void**)&Rin, memSize);
	generateRand(Rin,TEST_MAX,rLen,0);
	int *loc;
	CPUMALLOC((void**)&loc, sizeof(int)*rLen);
	generateRandInt(loc, rLen,rLen,0);
	Record *Rout;
	CPUMALLOC((void**)&Rout, outSize);
	Record *d_Rin;
	GPUMALLOC((void**)&d_Rin, memSize);
	Record *d_Rout;
	GPUMALLOC((void**)&d_Rout, outSize);
	int *d_loc;
	GPUMALLOC((void**)&d_loc, sizeof(int)*rLen);
	
	
	//copy to
	unsigned int timer=0;
	startTimer(&timer);
	TOGPU(d_Rin, Rin, memSize);
	TOGPU(d_loc, loc, sizeof(int)*rLen);
	endTimer("copy to GPU",&timer);
	
	//gather
	startTimer(&timer);
	startTime();
	gatherImpl(d_Rin,rLen,d_loc,d_Rout,rLen, numThread, numBlock);
	double sec=endTime("gather");
	double processingTime=endTimer("gather",&timer);

	//copy back
	startTimer(&timer);
	FROMGPU(Rout, d_Rout, outSize);
	endTimer("copy back", &timer);
	
	//gpuPrint(d_Rout, rLen, "d_Rout");
	printf("rLen, %d, numThread, %d, numBlock, %d, result, %d\n", rLen, numThread, numBlock, result);	
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	double kernel_bandwidth=dataSize/processingTime;
	printf("\n dataSize, %f MB, bandwidth, %f, kernel_bandwidth, %f\n", dataSize, bandwidth, kernel_bandwidth);
	CPUFREE(Rin);
	CPUFREE(Rout);
	CPUFREE(loc);
	GPUFREE(d_Rin);
	GPUFREE(d_Rout);
	GPUFREE(d_loc);
}

/*void testRadixSort(int rLen)
{
	int result=0;
	int totalBitsUsed=5;
	int bitPerPass=totalBitsUsed;
	int memSize=sizeof(Record)*rLen;
	int outSize=sizeof(Record)*rLen;
	Record *Rin;
	CPUMALLOC((void**)&Rin, memSize);
	generateRand(Rin, (1<<totalBitsUsed),rLen,0);
	Record *Rout;
	CPUMALLOC((void**)&Rout, outSize);
	Record *d_Rin;
	GPUMALLOC((void**)&d_Rin, memSize);
	Record *d_Rout;
	GPUMALLOC((void**)&d_Rout, outSize);

	startTime();
	//copy to
	unsigned int timer=0;
	startTimer(&timer);
	TOGPU(d_Rin, Rin, memSize);
	endTimer("copy to GPU",&timer);
	
	//sort
	//log2(rLen);
	int moveRightBits=totalBitsUsed-bitPerPass;
	int numBound=1;
	Record* d_iBound;
	GPUMALLOC((void**)&d_iBound, (1<<bitPerPass)*sizeof(Record));
	Record* h_iBound;
	CPUMALLOC((void**)&h_iBound, (1<<bitPerPass)*sizeof(Record));
	h_iBound[0].x=0;
	h_iBound[0].y=rLen;
	TOGPU(d_iBound, h_iBound, (1<<bitPerPass)*sizeof(Record));
	Record* d_oBound;
	GPUMALLOC((void**)&d_oBound, (1<<bitPerPass)*sizeof(Record)*numBound);
	Record* h_oBound;
	CPUMALLOC((void**)&h_oBound, (1<<bitPerPass)*sizeof(Record)*numBound);
	int* d_extra;
	GPUMALLOC((void**)&d_extra, rLen*sizeof(int));
	int* d_pidArray;
	GPUMALLOC((void**)&d_pidArray, rLen*sizeof(int));


	startTimer(&timer);
//	sortImpl(d_Rin,rLen,d_Rout);
	radixPart(d_Rin, d_pidArray, rLen, moveRightBits, bitPerPass, totalBitsUsed, d_iBound, numBound, d_extra, d_oBound, d_Rout);
	double processingTime=endTimer("radix",&timer);

	//copy back
	startTimer(&timer);
	FROMGPU(Rout, d_Rout, outSize);
	endTimer("copy back", &timer);
	double sec=endTime("total");
	FROMGPU(h_oBound, d_oBound, (1<<bitPerPass)*sizeof(Record)*numBound);
	//gpuPrint(d_Rout, rLen, "d_Rout");
	printf("rLen, %d, result, %d\n", rLen, result);	
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	double kernel_bandwidth=dataSize/processingTime;
	printf("\n dataSize, %f MB, bandwidth, %f, kernel_bandwidth, %f\n", dataSize, bandwidth, kernel_bandwidth);
	validateSort(Rout,rLen);
	CPUFREE(Rin);
	CPUFREE(Rout);
	CPUFREE(h_iBound);
	CPUFREE(h_oBound);
	GPUFREE(d_Rin);
	GPUFREE(d_Rout);
	GPUFREE(d_iBound);
	GPUFREE(d_oBound);
	GPUFREE(d_extra);
	GPUFREE(d_pidArray);
	
}*/


void testRadixSort(int rLen)
{
	int result=0;
	int memSize=sizeof(Record)*rLen;
	int outSize=sizeof(Record)*rLen;
	Record *Rin;
	CPUMALLOC((void**)&Rin, memSize);
	//generateRand(Rin,TEST_MAX,rLen,0);

	for( int i = 0; i < rLen; i++ )
	{
		Rin[i].x = i;
		Rin[i].y = rLen - i;
	}	
	
	Record *Rout;
	CPUMALLOC((void**)&Rout, outSize);
	Record *d_Rin;
	GPUMALLOC((void**)&d_Rin, memSize);
	Record *d_Rout;
	GPUMALLOC((void**)&d_Rout, outSize);

	unsigned int timer=0;
	//copy to	
	startTimer(&timer);
	TOGPU(d_Rin, Rin, memSize);
	endTimer("copy to GPU",&timer);
	
	startTimer(&timer);
	startTime();
	radixSortImpl(d_Rin,rLen,d_Rout);
	double sec=endTime("radixsort");
	double processingTime=endTimer("radix",&timer);

	//copy back
	startTimer(&timer);
	FROMGPU(Rout, d_Rout, outSize);
	endTimer("copy back", &timer);

	//gpuPrint(d_Rout, rLen, "d_Rout");
	printf("rLen, %d, result, %d\n", rLen, result);	
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	double kernel_bandwidth=dataSize/processingTime;
	printf("\n dataSize, %f MB, bandwidth, %f, kernel_bandwidth, %f\n", dataSize, bandwidth, kernel_bandwidth);
	validateSort(Rout,rLen);
	CPUFREE(Rin);
	CPUFREE(Rout);	
	//GPUFREE(d_Rin);
	//GPUFREE(d_Rout);
}

void testQSort(int rLen)
{
	int memSize=sizeof(Record)*rLen;

	Record *h_Rin;
	CPUMALLOC((void**)&h_Rin, memSize);
	generateRand(h_Rin, TEST_MAX,rLen,0);
	Record *h_Rout;
	CPUMALLOC((void**)&h_Rout, memSize);

	Record* d_Rin;
	Record* d_Rout;
	GPUMALLOC( (void**)&d_Rin, memSize ) ;
	GPUMALLOC( (void**)&d_Rout, memSize );

	unsigned int timer = 0;

	startTimer( &timer );
	TOGPU( d_Rin, h_Rin, memSize );
	endTimer( "copy to gpu", &timer );
	
	startTimer(&timer);
	GPUOnly_QuickSort( d_Rin, rLen, d_Rout);
	endTimer("qsort", &timer);

	startTimer( &timer );
	FROMGPU( h_Rout, d_Rout, memSize );
	endTimer( "copy back", &timer );

	validateSort(h_Rout, rLen);
	
	CPUFREE(h_Rin);
	CPUFREE(h_Rout);	
}

void testStringSort(int numString, int minLen, int maxLen)
{
	char **data;
	CPUMALLOC((void**)&data,sizeof(char*));
	int **len;
	CPUMALLOC((void**)&len,sizeof(int*));
	int **offset;
	CPUMALLOC((void**)&offset,sizeof(int*));
	int totalLenInBytes=generateStringGPU(numString,minLen,maxLen,data,len,offset); 
	cmp_type_t *Rin;
	CPUMALLOC((void**)&Rin,sizeof(cmp_type_t)*numString);
	cmp_type_t **Rout;
	CPUMALLOC((void**)&Rout,sizeof(cmp_type_t*));
	int i=0;
	for(i=0;i<numString;i++)
	{
		Rin[i].x=(*offset)[i];//rand();
		Rin[i].y=(*len)[i];
	}
	unsigned int timer=0;
	startTimer(&timer);
	quickSort( *data, totalLenInBytes, Rin, numString, Rout);
	endTimer("End-to-end GPU sorting", &timer);
	printString(*data, (cmp_type_t*)*Rout, numString);
}
void microBenchMark(int rLen)
{
	/*testMapImpl(rLen);
	testSplit(rLen, 64);
	testScan(rLen);
	testScatter(rLen);
	testGather(rLen);
	testReduceImpl(rLen);
	testFilterImpl(rLen);
	testQSort(rLen);
	testSort(rLen);*/
	testScatter(rLen);
	testGather(rLen);
}
int testAllPrimitive(int argc, char ** argv)
{
	int i=0;
	for(i=0;i<argc;i++)
		printf("%s ", argv[i]);
	printf("\n");
//	testMapImpl(16);
//	testGather(16);
//	testMapImpl(16);
//	testRadixSort(1024*1024*1);
//	testSort(1024*1024*2);
//	printf("%d", log2Ceil(7));
//	testRadixSort(1024*1024*16);
//	testQSort(1024*1024*8-1);
	for(i=0;i<argc;i++)
	{
		if(strcmp(argv[i], "-micro")==0)
		{
			int rLen=8*1024;
			//default block size.
			if(argc==(i+2))
			{
				rLen=atoi(argv[i+1])*1024;
				microBenchMark(rLen);
			}
			else
				microBenchMark(rLen);
			
		}
		if(strcmp(argv[i], "-map")==0)
		{
			int rLen=8*1024*1024;
			//default block size.
			if(argc==(i+2))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				testMapImpl(rLen);
			}
			//vary the block size
			if(argc==(i+4))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				int numThreadPB=atoi(argv[i+2]);
				int numBlock=atoi(argv[i+3]);
				testMapImpl(rLen,numThreadPB,numBlock);
			}
			
		}

		if(strcmp(argv[i], "-reduce")==0)
		{
			int rLen=8*1024*1024;
			//default block size, sum.
			if(argc==(i+2))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				testReduceImpl(rLen);
			}
			//vary the block size
			if(argc==(i+5))
			{				
				int op;
				if( strcmp(argv[i+1], "max") == 0 )
				{
					op = REDUCE_MAX;
				}
				else if( strcmp(argv[i+1], "min") == 0 )
				{
					op = REDUCE_MIN;
				}
				else if( strcmp(argv[i+1], "sum") == 0 )
				{
					op = REDUCE_SUM;
				}
				else if( strcmp(argv[i+1], "avg") == 0 )
				{
					op = REDUCE_AVERAGE;
				}

				rLen=atoi(argv[i+2])*1024*1024;

				//int op=atoi(argv[i+2]);
				int numThreadPB=atoi(argv[i+3]);
				int numMaxBlock = atoi(argv[i+4]);				

				//testReduceImpl( int rLen, int OPERATOR=REDUCE_MAX, int numThreadPB=512, int numMaxBlock = 512 )
				testReduceImpl(rLen, op, numThreadPB, numMaxBlock);
			}
			
		}
		//testFilterImpl
		if(strcmp(argv[i], "-filter")==0)
		{
			int rLen=8*1024*1024;
			//default block size, sum.
			if(argc==(i+2))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				testFilterImpl(rLen);
			}
			//vary the block size
			if(argc==(i+4))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				int numThreadPB=atoi(argv[i+2]);
				int numBlock=atoi(argv[i+3]);
				testFilterImpl(rLen,numThreadPB,numBlock);
			}
			
		}

		//test selection
		if(strcmp(argv[i], "-selection")==0)
		{
			int rLen=8*1024*1024;
			//default block size, sum.
			if(argc==(i+2))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				testSelection(rLen);
			}
			//vary the block size
			if(argc==(i+4))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				int numThreadPB=atoi(argv[i+2]);
				int numBlock=atoi(argv[i+3]);
				testSelection(rLen,numThreadPB,numBlock);
			}
			
		}
		
		if(strcmp(argv[i], "-group_by")==0)
		{
			int rLen=8*1024*1024;
			//default block size, sum.
			if(argc==(i+2))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				testGroupByImpl(rLen);
			}
			//vary the block size
			if(argc==(i+4))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				int numThreadPB=atoi(argv[i+2]);
				int numBlock=atoi(argv[i+3]);
				testGroupByImpl(rLen,numThreadPB,numBlock);
			}
			
		}

		if(strcmp(argv[i], "-split")==0)
		{
			int rLen=8*1024*1024;
			int numPart=64;
			if(argc==(i+3))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				numPart=atoi(argv[i+2]);
				testSplit(rLen, numPart);
			}
			if(argc==(i+5))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				numPart=atoi(argv[i+2]);
				int numThreadPB=atoi(argv[i+3]);				
				int numBlock=atoi(argv[i+4]);
				testSplit(rLen,numPart,numThreadPB,numBlock);
			}
			
		}
		
		if(strcmp(argv[i], "-scan")==0)
		{
			int rLen=8*1024*1024;
			if(argc==(i+2))
			{
				rLen=atoi(argv[i+1])*1024*1024;
			}
			testScan(rLen);
		}

		if(strcmp(argv[i], "-scatter")==0)
		{
			int rLen=8*1024*1024;
			if(argc==(i+2))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				testScatter(rLen);
			}
			
			//vary the block size
			if(argc==(i+4))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				int numThreadPB=atoi(argv[i+2]);
				int numBlock=atoi(argv[i+3]);
				testScatter(rLen,numThreadPB,numBlock);
			}
		}

		if(strcmp(argv[i], "-proj")==0)
		{
			if(argc==(i+4))
			{
				int rLen=atoi(argv[i+1])*1024*1024;
				int numThreadPB=atoi(argv[i+2]);
				int numBlock=atoi(argv[i+3]);
				int pLen = rLen*0.01;

				testProjection( rLen, pLen, numThreadPB, numBlock ); 
			}
		}


		if(strcmp(argv[i], "-bitonic_sort")==0)
		{
			int rLen=8*1024*1024;
			if(argc==(i+2))
			{
				rLen=atoi(argv[i+1])*1024;
				testBitonicSort(rLen);
			}
			if(argc==(i+4))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				int numThreadPB=atoi(argv[i+2]);
				int numBlock=atoi(argv[i+3]);
				testBitonicSort(rLen,numThreadPB,numBlock);
			}
			
		}

		if(strcmp(argv[i], "-gather")==0)
		{
			int rLen=8*1024*1024;
			int numThread;
			int numBlock;
			if(argc==(i+2))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				numThread = 512;
				numBlock = 256;
			}
			if(argc==(i+4))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				numThread = atoi(argv[i+2]);
				numBlock = atoi(argv[i+3]);
			}
			
			testGather(rLen, numThread, numBlock);
		}
		if(strcmp(argv[i], "-radix")==0)
		{
			int rLen=8*1024*1024;
			if(argc==(i+2))
			{
				rLen=atoi(argv[i+1])*1024*1024;
			}
			testRadixSort(rLen);
		}

		if(strcmp(argv[i], "-qsort")==0)
		{
			int rLen=8*1024*1024;
			if(argc==(i+2))
			{
				rLen=atoi(argv[i+1])*1024*1024;
			}
			testQSort(rLen);
		}

		if(strcmp(argv[i], "-copy")==0)
		{
			int rLen=8*1024*1024;
			int chunkSize=1024*1024;
			if(argc==(i+3))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				chunkSize=atoi(argv[i+2])*1024;
			}
			testCopy(rLen, chunkSize);
		}
		if(strcmp(argv[i], "-string")==0)
		{
			int rLen=4*1024*1024;
			int minLen=4;
			int maxLen=4;
			if(argc==(i+4))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				minLen=atoi(argv[i+2]);
				maxLen=atoi(argv[i+3]);
			}
			testStringSort(rLen, minLen, maxLen);
		}
		
	}
	return 0;
}


