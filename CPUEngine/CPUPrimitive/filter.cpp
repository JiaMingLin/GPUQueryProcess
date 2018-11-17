#include "stdafx.h"
#include "MyThreadPool.h"
#include "Primitive.h"
#include "omp.h"
#include "common.h"
#include <iostream>

using namespace std;

//for point filter.
int filterFun_point( Record* Rin, int rLen, int keyValue, Record** Rout )
{
	unsigned int size = 0;

	for( int i = 0; i < rLen; i++ )
	{
		if( Rin[i].value==keyValue )
		{
			size++;
		}
	}

	(*Rout) = (Record*)malloc( sizeof(Record)*size ); 

	int idx = 0;
	for( int i = 0; i < rLen; i++ )
	{
		if( Rin[i].value==keyValue )
		{
			(*Rout)[idx] = Rin[i];
			idx++;
		}
	}

	return size;
}

//for range filter.
int filterFun_range( Record* Rin, int rLen, int lowerKey, int higherKey, Record** Rout )
{
	unsigned int size = 0;

	for( int i = 0; i < rLen; i++ )
	{
		if( lowerKey<=Rin[i].value && Rin[i].value<=higherKey)
		{
			size++;
		}
	}

	(*Rout) = (Record*)malloc( sizeof(Record)*size ); 

	int idx = 0;
	for( int i = 0; i < rLen; i++ )
	{
		if( lowerKey<=Rin[i].value && Rin[i].value<=higherKey)
		{
			(*Rout)[idx] = Rin[i];
			idx++;
		}
	}

	return size;
}

void copyResult( Record* Rout, Record* Rin, int numResult, int copyIdx )
{
	memcpy( Rout + copyIdx, Rin, sizeof(Record)*numResult);
}

void validateFilter( Record* Rin, int rLen, Record* Rout, int outSize)
{
	bool passed = true;

	int count = 0;
	for( int i = 0; i < rLen; i++ )
	{
		//the filter condition
		if( FILTER_CONDITION )
		{
			count++;
		}
	}

	if( count != outSize )
	{
		printf( "!!!filter error: the number error\n" );
		passed = false;
		exit(0);
	}

	Record* v_Rout = (Record*)malloc( sizeof(Record)*outSize );
	int j = 0;
	for( int i = 0; i < rLen; i++ )
	{
		//the filter condition
		if( FILTER_CONDITION )
		{
			v_Rout[j] = Rin[i];
			j++;
		}
	}

	for( int i = 0; i < outSize; i++ )
	{
		if( (v_Rout[i].rid != Rout[i].rid) || (v_Rout[i].value != Rout[i].value) )
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

/*
if lowerKey==higherKey, point filter.
if lowerKey<higherKey, range filter.
*/
int filter_openmp( Record* Rin, int rLen, int lowerKey, int higherKey, Record** Rout, int numThread )
{
	Record** subResult = new Record*[numThread];
	
	int* start = new int[numThread];
	int* end = new int[numThread]; 
	int* numResult = new int[numThread];
	int chunkSize = rLen/numThread;
	int remainer = rLen%numThread;

	for( int i = 0; i < numThread; i++ )
	{
		start[i] = i*chunkSize;
		end[i] = (i==(numThread - 1))?(rLen):((i + 1)*chunkSize);
	}

	omp_set_num_threads( numThread );

	//get the sub-result
	if(lowerKey==higherKey)
	{
		#pragma omp parallel for
		for( int tx = 0; tx < numThread; tx++ )
		{
			numResult[tx] = filterFun_point( Rin + start[tx], end[tx] - start[tx], lowerKey, &subResult[tx] );
		}
	}
	else
	{
		#pragma omp parallel for
		for( int tx = 0; tx < numThread; tx++ )
		{
			numResult[tx] = filterFun_range( Rin + start[tx], end[tx] - start[tx], lowerKey, higherKey, &subResult[tx] );
		}
	}

	unsigned int totalNumResult = 0;
	int* copyIdx = new int[numThread];

	for( int tx = 0; tx < numThread; tx++ )
	{
		copyIdx[tx] = totalNumResult;
		totalNumResult += numResult[tx];
	}
	(*Rout) = (Record*)malloc( sizeof(Record)*totalNumResult );

	//copy the result
#pragma omp parallel for
	for( int tx = 0; tx < numThread; tx++ )
	{
		copyResult( (*Rout), subResult[tx], numResult[tx], copyIdx[tx] );
	}

	return totalNumResult;

	//validateFilter(Rin, rLen, (*Rout), totalNumResult);
}