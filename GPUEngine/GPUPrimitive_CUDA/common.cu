#ifndef COMMON_CU
#define COMMON_CU


//#define DEBUG_SAVEN 0

// includes, project
#include <cutil.h>
#include "stdio.h"
#include "stdlib.h"

#define NUM_RECORDS_R (512*512*4*16)

#define NUM_THREADS_SORT (512)
#define NUM_BLOCKS_X_SORT (NUM_RECORDS_R/NUM_THREADS_SORT)
#define NUM_BLOCKS_Y_SORT (1)

#define REDUCE_SUM (0)
#define REDUCE_MAX (1)
#define REDUCE_MIN (2)
#define REDUCE_AVERAGE (3)

#define SPLIT (4)
#define PARTITION (5)

//#define FILTER_CONDITION (d_Rin[idx].y < 100000000)

typedef int4 cmp_type_t;


//2^n
int TwoPowerN( int n )
{
	return (1<<n);
}

void gpuPrint(int *d_output, int numResults, char *notes)
{
#ifdef DEBUG_SAVEN
	printf("----------%s------------\n", notes);
	int result=0;
	int *h_output=(int *)malloc(sizeof(int)*numResults);
	CUDA_SAFE_CALL( cudaMemcpy( h_output, d_output, numResults*sizeof(int) , cudaMemcpyDeviceToHost) );
	for(int i=0;i<numResults;i++)
	{
		printf("%d, ", h_output[i]);
		result+=h_output[i];
		if(i%10==9) printf("\n");
	}
	printf("#results in GPU: %f K, length, %d\n",(double)result, numResults);
	free(h_output);
#endif
}

void gpuPrintInterval(int *d_output, int numResults, char *notes)
{
#ifdef DEBUG_SAVEN
	printf("----------%s------------\n", notes);
	int result=0;
	int *h_output=(int *)malloc(sizeof(int)*numResults);
	CUDA_SAFE_CALL( cudaMemcpy( h_output, d_output, numResults*sizeof(int) , cudaMemcpyDeviceToHost) );
	int unitSize=256;
	int hist[20];//each is 50.
	int k=0;
	for(k=0;k<20;k++) hist[k]=0;
	int interval=0;
	for(int i=0;i<numResults;i=i+2)
	{
		interval=h_output[i+1]-h_output[i];
		//printf("%d, ", interval);
		if(interval>1)
		result+=1;
		interval=interval/unitSize;
		if(interval>=20)
			interval=19;
		hist[interval]++;
		//if(i%10==8) printf("\n");
	}
	printf("#results in GPU: %f K, length, %d\n",(double)result, numResults);
	for(k=0;k<20;k++) 
		printf("%d, ", k*unitSize);
	printf("\n");
	for(k=0;k<20;k++) 
		printf("%d, ", hist[k]);
	printf("\n");
	free(h_output);
#endif
}

void gpuPrintInt2(Record *d_output, int numResults, char *notes)
{
//#ifdef DEBUG_SAVEN
	printf("----------%s------------\n", notes);
	int result=0;
	Record *h_output=(Record *)malloc(sizeof(Record)*numResults);
	CUDA_SAFE_CALL( cudaMemcpy( h_output, d_output, numResults*sizeof(Record) , cudaMemcpyDeviceToHost) );
	for(int i=0;i<numResults;i++)
	{
		printf("[%d,%d], ", h_output[i].x, h_output[i].y);
		result+=h_output[i].y;
		if(i%10==9) printf("\n");
	}
	printf("#results in GPU: %f K, length, %d\n",(double)result, numResults);
	free(h_output);
//#endif
}


void gpuPrintFloat(float *d_output, int numResults, char *notes)
{
#ifdef DEBUG_SAVEN
	printf("----------%s------------\n", notes);
	float result=0;
	float *h_output=(float *)malloc(sizeof(float)*numResults);
	CUDA_SAFE_CALL( cudaMemcpy( h_output, d_output, numResults*sizeof(float) , cudaMemcpyDeviceToHost) );
	for(int i=0;i<numResults;i++)
	{
		printf("%f, ", h_output[i]);
		result+=h_output[i];
		if(i%10==9) printf("\n");
	}
	printf("#results in GPU: %f K, length, %d\n",(double)result, numResults);
	free(h_output);
#endif
}

void validateScan( int* input, int rLen, int* output )
{
	int* checkOutput = (int*)malloc( sizeof(int)*rLen );

	checkOutput[0] = 0;

	for( int i = 1; i < rLen; i++ )
	{
		checkOutput[i] = checkOutput[i - 1] + input[i - 1];
	}

	bool pass = true;
	for( int i = 0; i < rLen; i++ )
	{
		if( checkOutput[i] != output[i] )
		{
			pass = false;
			printf( "!!!error\n" );
			break;
		}
	}

	if( pass )
	{
		printf( "scan test pass!\n" );
	}
	else
	{
		printf( "scan test failed!\n" );
	}
}

void validateProjection( Record* h_Rin, int rLen, Record* originalProjTable, Record* h_projTable, int pLen )
{
	Record* checkProjTable = (Record*)malloc( sizeof(Record)*pLen );
	bool pass = true;

	unsigned int timer = 0;

	startTimer( &timer );
	for( int i = 0; i < pLen; i++ )
	{
		checkProjTable[i].x = originalProjTable[i].x;
		checkProjTable[i].y = h_Rin[originalProjTable[i].x].y;
	}
	endTimer( "cpu projection", &timer );

	for( int i = 0; i < pLen; i++ )
	{
		if( (h_projTable[i].x != checkProjTable[i].x) || (h_projTable[i].y != h_projTable[i].y) )
		{
			pass = false;
			break;
		}
	}

	if( pass )
	{
		printf( "projection test pass!\n" );
	}
	else
	{
		printf( "!error, porjection test failed! \n" );
	}
}

void validateAggAfterGroupBy( Record *Rin, int rLen, int* startPos, int numGroups, Record* Ragg, int* aggResults, int OPERATOR )
{
	bool pass = true;

	int* checkResult = (int*)malloc( sizeof(int)*numGroups );
	int result;
	//int groupIdx = 0;

	Record* S = (Record*)malloc( sizeof(Record)*rLen );
	for( int i = 0; i < rLen; i++ )
	{
		S[i] = Ragg[Rin[i].x];
	}

	//aggregation
	unsigned int timer = 0;
	startTimer( &timer );
			
	int* endPos = (int*)malloc( sizeof(int)*numGroups );
	for( int i = 0; i < numGroups - 1; i++ )
	{
		endPos[i] = startPos[i + 1];
	}
	endPos[numGroups - 1] = rLen;

	for( int i = 0; i < numGroups; i++ )
	{
		if( OPERATOR == REDUCE_MAX )
		{
			result = 0;
			for( int j = startPos[i]; j < endPos[i]; j++ )
			{
				if( S[j].y > result )
				{
					result = S[j].y;
				}
			}
			checkResult[i] = result;
		}
		else if( OPERATOR == REDUCE_MIN )
		{
			result = TEST_MAX;
			for( int j = startPos[i]; j < endPos[i]; j++ )
			{
				if( S[j].y < result )
				{
					result = S[j].y;
				}
			}
			checkResult[i] = result;
		}
		else if( OPERATOR == REDUCE_SUM )
		{
			result = 0;
			for( int j = startPos[i]; j < endPos[i]; j++ )
			{
				result += S[j].y;
			}
			checkResult[i] = result;
		}
		else if( OPERATOR == REDUCE_AVERAGE )
		{
			result = 0;
			for( int j = startPos[i]; j < endPos[i]; j++ )
			{
				result += S[j].y;
			}
			checkResult[i] = result/(endPos[i] - startPos[i]);
		}
	}
	endTimer( "cpu aggregration after group by", &timer );

	for( int i = 0; i < numGroups; i++ )
	{
		if( checkResult[i] != aggResults[i] )
		{
			printf( "Aggregrate test failed!\n" );
			pass = false;
			break;
		}
	}

	if( pass == true )
	{
		printf( "Test Passed!\n" );
	}
	
	free( S );
}

void validateGroupBy( Record* h_Rin, int rLen, Record* h_Rout, int* h_startPos, int numGroup )
{
	bool pass = true;
	qsort(h_Rin,rLen,sizeof(Record),compare);

	//test sort
	for( int i = 0; i < rLen; i++ )
	{
		if( (h_Rin[i].y != h_Rout[i].y) )
		{
			pass = false;
			printf( "sort error!\n" );
		}
		break;
	}

	//test group
	int count = 1;
	for( int i = 0; i < rLen - 1; i++ )
	{
		if( h_Rin[i].y != h_Rin[i+1].y )
		{
			count++;
		}
	}
	if( count != numGroup )
	{
		pass = false;
		printf( "count error! GPU, %d, CPU, %d\n", numGroup, count );
	}
	int* startPos = (int*)malloc( sizeof(int)*count );
	int j = 1;
	for( int i = 0; i < rLen - 1; i++ )
	{
		if( h_Rin[i].y != h_Rin[i+1].y )
		{
			startPos[j] = i + 1;
			j++;
		}
	}
	startPos[0] = 0;
	for( int idx = 0; idx < count; idx++ )
	{
		if( h_startPos[idx] != startPos[idx] )
		{
			pass = false;
			printf( "start position error!, GPU position: %d, CPU position: %d\n", h_startPos[idx], startPos[idx] );
			break;
		}
	}

	if( pass == true )
	{
		printf( "GroupBy Test passed!\n" );
	}
	else
	{
		printf( "GroupBy Test failed!\n" );
	}
}

void validateFilter( Record* d_Rin, int beginPos, int rLen, 
					Record* Rout, int outSize, int smallKey, int largeKey)
{
	bool passed = true;
	
	unsigned int timer = 0;
	startTimer( &timer );
	int count = 0;
	for( int i = 0; i < rLen; i++ )
	{
		//the filter condition
		int idx = beginPos + i;
		if( ( d_Rin[idx].y >= smallKey ) && ( d_Rin[idx].y <= largeKey ) )
		{
			count++;
		}
	}

	if( count != outSize )
	{
		printf( "!!!filter error: the number error, GPU, %d, CPU, %d\n", outSize, count );
		passed = false;
		exit(0);
	}

	Record* v_Rout = (Record*)malloc( sizeof(Record)*outSize );
	int j = 0;
	for( int i = 0; i < rLen; i++ )
	{
		//the filter condition
		int idx = beginPos + i;
		if( ( d_Rin[idx].y >= smallKey ) && ( d_Rin[idx].y <= largeKey ) )
		{
			v_Rout[j] = d_Rin[beginPos+i];
			j++;
		}
	}
	endTimer( "cpu timer", &timer );

	for( int i = 0; i < outSize; i++ )
	{
		if( (v_Rout[i].x != Rout[i].x) || (v_Rout[i].y != Rout[i].y) )
		{
			printf( "!!! filter error\n" );
			passed = false;
			exit(0);
		}
	}

	if( passed )
	{
		printf( "filter passed\n" );
	}
}

void validateReduce( Record* R, int rLen, unsigned int gpuResult, int OPERATOR )
{
	unsigned int timer = 0;

	if( OPERATOR == REDUCE_SUM )
	{
		unsigned int cpuSum = 0;

		startTimer( &timer );
		for( int i = 0; i < rLen; i++ )
		{
			cpuSum += R[i].y;
		}
		endTimer( "cpu sum", &timer );

		if( gpuResult == cpuSum )
		{
			printf( "Test Passed: gpuSum = %d, cpuSum = %d\n", gpuResult, cpuSum );
		}
		else
		{
			printf( "!!!Test Failed: gpuSum = %d, cpuSum = %d\n", gpuResult, cpuSum );
		}
	}
	else if ( OPERATOR == REDUCE_AVERAGE )
	{
		unsigned int cpuAvg = 0;

		startTimer( &timer );
		for( int i = 0; i < rLen; i++ )
		{
			cpuAvg += R[i].y;
		}
		cpuAvg = cpuAvg/rLen;
		endTimer( "cpu sum", &timer );

		if( gpuResult == cpuAvg )
		{
			printf( "Test Passed: gpuAvg = %d, cpuAvg = %d\n", gpuResult, cpuAvg );
		}
		else
		{
			printf( "!!!Test Failed: gpuAvg = %d, cpuAvg = %d\n", gpuResult, cpuAvg );
		}
	}
	else if( OPERATOR == REDUCE_MAX )
	{
		int cpuMax = R[0].y;		

		startTimer( &timer );
		for( int i = 1; i < rLen; i++ )
		{
			if( R[i].y > cpuMax )
			{
				cpuMax = R[i].y;
			}
		}
		endTimer( "cpu max", &timer );

		if( gpuResult == cpuMax )
		{
			printf( "Test Passed: gpuMax = %d, cpuMax = %d\n", gpuResult, cpuMax );
		}
		else
		{
			printf( "!!!Test Failed: gpuMax = %d, cpuMax = %d\n", gpuResult, cpuMax );
		}
	}
	else if( OPERATOR == REDUCE_MIN )
	{
		int cpuMin = R[0].y;		

		startTimer( &timer );
		for( int i = 1; i < rLen; i++ )
		{
			if( R[i].y < cpuMin )
			{
				cpuMin = R[i].y;
			}
		}
		endTimer( "cpu min", &timer );

		if( gpuResult == cpuMin )
		{
			printf( "Test Passed: gpuMin = %d, cpuMin = %d\n", gpuResult, cpuMin );
		}
		else
		{
			printf( "!!!Test Failed: gpuMin = %d, cpuMin = %d\n", gpuResult, cpuMin );
		}
	}
}

void validateSort(Record *R, int rLen)
{
	int i=0;
	bool passed=true;
	for(i=1;i<rLen;i++)
	{
		if(R[i].y<R[i-1].y)
		{
			printf("!!!error in sorting: %d, %d, %d, %d\n", i-1, R[i-1].y, i,R[i].y);
			passed=false;
			return;
		}
	}
	if(passed)
		printf("sorting passed\n");
}


void gpuValidateSort(Record *d_R, int rLen)
{
	int i=0;
	Record *R;
	CPUMALLOC((void**)&R, rLen*sizeof(Record));
	FROMGPU(R, d_R, rLen*sizeof(Record));
	bool passed=true;
	for(i=1;i<rLen;i++)
	{
		if(R[i].y<R[i-1].y)
		{
			printf("!!!!!!!! error in sorting: %d, %d, %d, %d\n", i-1, R[i-1].y, i,R[i].y);
			passed=false;
			return;
		}
	}
	if(passed)
		printf("sorting passed\n");
	CPUFREE(R);
}
void validateSplit(Record *R, int rLen, int numPart)
{
	int i=0;
	bool passed=true;
	for(i=1;i<rLen;i++)
	{
		if((R[i].y%numPart)<(R[i-1].y%numPart))
		{
			printf("error in partition: %d, %d, %d, %d\n", i-1, R[i-1].y, i,R[i].y);
			passed=false;
			break;
		}
	}
	if(passed)
		printf("\npartition passed\n");
}

unsigned int cpu_RSHash(int value, int mask)
{

    unsigned int b=378551;
    unsigned int a=63689;
    unsigned int hash=0;
    int i=0;

    for(i=0;i<4;i++)
    {
        hash=hash*a+(value>>(24-(i<<3)));
        a*=b;
    }

    return (hash & mask);
}

void validatePartition( Record* R, int rLen, int numPart )
{
	bool pass = true;

	for( int i = 1; i < rLen; i++ )
	{
		if( cpu_RSHash( R[i].y, numPart - 1 ) < cpu_RSHash( R[i - 1].y, numPart - 1 ) )
		{
			printf("error in partition: %d, %d, %d, %d\n", i-1, R[i-1].y, i,R[i].y);
			pass = false;
			break;
		}
	}

	if( pass )
	{
		printf( "partition test pass! \n" );
	}
}


int get2N( int rLen )
{
	unsigned int numRecordsR = 0;

	unsigned int size = rLen;
	unsigned int level = 0;
	while( size != 1 )
	{
		size = size/2;
		level++;
	}

	if( (1<<level) < rLen )
	{
		level++;
	}

	numRecordsR = (1<<level);
	return numRecordsR;
}



bool is2n(unsigned int i)
{
    if(i==0) return false;    
    else return (i&(i-1))==0;
}

unsigned int g_timerArray[10];
unsigned long g_totalArray[10]={0};
void array_startTime(int i)
{
    CUT_SAFE_CALL( cutCreateTimer( &(g_timerArray[i])));
    CUT_SAFE_CALL( cutStartTimer( (g_timerArray[i])));
}


void array_endTime(char *info,int i)
{
	CUT_SAFE_CALL( cutStopTimer( g_timerArray[i]));
	g_totalArray[i]+=(long)cutGetTimerValue( g_timerArray[i]);
    printf( "%s (ms), %f, total, %d\n", info, cutGetTimerValue( g_timerArray[i]), g_totalArray[i]);
    CUT_SAFE_CALL( cutDeleteTimer( g_timerArray[i]));
}



#endif
