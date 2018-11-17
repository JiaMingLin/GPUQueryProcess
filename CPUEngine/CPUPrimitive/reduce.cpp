#include "stdafx.h"
#include "MyThreadPool.h"
#include "Primitive.h"
#include "omp.h"
#include "common.h"
#include <iostream>

using namespace std;

void validateReduce( Record* R, int rLen, unsigned int cpuResult, int OPERATOR )
{
	unsigned int timer = 0;

	if( OPERATOR == REDUCE_SUM )
	{
		unsigned int checkSum = 0;

		for( int i = 0; i < rLen; i++ )
		{
			checkSum += R[i].value;
		}

		if( cpuResult == checkSum )
		{
			printf( "Test Passed: cpuSum = %d, checkSum = %d\n", cpuResult, checkSum );
		}
		else
		{
			printf( "!!!Test Failed: cpuSum = %d, checkSum = %d\n", cpuResult, checkSum );
		}
	}
	else if( OPERATOR == REDUCE_MAX )
	{
		int checkMax = R[0].value;		

		for( int i = 1; i < rLen; i++ )
		{
			if( R[i].value > checkMax )
			{
				checkMax = R[i].value;
			}
		}

		if( cpuResult == checkMax )
		{
			printf( "Test Passed: cpuMax = %d, checkMax = %d\n", cpuResult, checkMax );
		}
		else
		{
			printf( "!!!Test Failed: cpuMax = %d, checkMax = %d\n", cpuResult, checkMax );
		}
	}
	else if( OPERATOR == REDUCE_MIN )
	{
		int checkMin = R[0].value;		

		for( int i = 1; i < rLen; i++ )
		{
			if( R[i].value < checkMin )
			{
				checkMin = R[i].value;
			}
		}

		if( cpuResult == checkMin )
		{
			printf( "Test Passed: cpuMin = %d, checkMin = %d\n", cpuResult, checkMin );
		}
		else
		{
			printf( "!!!Test Failed: cpuMin = %d, checkMin = %d\n", cpuResult, checkMin );
		}
	}
}

long int reduceFun( Record* Rin, int rLen, int OPERATOR )
{
	long int result;

	if( OPERATOR == REDUCE_SUM )
	{
		result = 0;
		for( int i = 0; i < rLen; i++ )
		{
			result += Rin[i].value;	
		}
	}
	else if( OPERATOR == REDUCE_MAX )
	{
		result = TEST_MIN;

		for( int i = 0; i < rLen; i++ )
		{
			result = (Rin[i].value > result) ? Rin[i].value : result;
		}		
	}
	else if( OPERATOR == REDUCE_MIN )
	{
		result = TEST_MAX;

		for( int i = 0; i < rLen; i++ )
		{
			result = (Rin[i].value < result) ? Rin[i].value : result;
		}	
	}

	return result;
}

long int reduce_openmp( Record* Rin, int rLen, int numThread, int OPERATOR )
{
	long int result;

	int* subResult = new int[numThread];
	int* start = new int[numThread];
	int* end = new int[numThread]; 
	int chunkSize = rLen/numThread;
	int remainer = rLen%numThread;

	for( int i = 0; i < numThread; i++ )
	{
		start[i] = i*chunkSize;
		end[i] = (i==(numThread - 1))?(rLen):((i + 1)*chunkSize);
	}

	omp_set_num_threads( numThread );

	if( OPERATOR == REDUCE_SUM )
	{
		result = 0;

#pragma omp parallel for
		for( int tx = 0; tx < numThread; tx++ )
		{
			subResult[tx] = reduceFun( Rin + start[tx], end[tx] - start[tx], OPERATOR );
 		}

		for( int i = 0; i < numThread; i++ )
		{
			result += subResult[i];
		}
	}
	else if( OPERATOR == REDUCE_MAX )
	{
#pragma omp parallel for
		for( int tx = 0; tx < numThread; tx++ )
		{
			subResult[tx] = reduceFun( Rin + start[tx], end[tx] - start[tx], OPERATOR );
 		}

		result = TEST_MIN;

		for( int i = 0; i < numThread; i++ )
		{
			result = (subResult[i] > result) ? subResult[i] : result;
		}
	}
	else if( OPERATOR == REDUCE_MIN )
	{
#pragma omp parallel for
		for( int tx = 0; tx < numThread; tx++ )
		{
			subResult[tx] = reduceFun( Rin + start[tx], end[tx] - start[tx], OPERATOR );
 		}

		result = TEST_MAX;

		for( int i = 0; i < numThread; i++ )
		{
			result = (subResult[i] < result) ? subResult[i] : result;
		}
	}	

	//validateReduce( Rin, rLen, result, OPERATOR );

	return result;
}