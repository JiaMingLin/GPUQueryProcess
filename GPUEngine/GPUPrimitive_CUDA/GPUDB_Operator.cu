#ifndef GPUDB_OPERATOR
#define  GPUDB_OPERATOR
/*
This file implements a common set of operators.
*/

/*
selection,
two versions:
point selection and range selection.
Rout point to the array of the results in the device memory.
*/

#include "GPU_Dll.h"

int point_selection(Record* d_Rin, int rLen, int matchingKeyValue, Record **d_Rout, 
					int numThreadPB, int numBlock)
{
	int* outSize = (int*)malloc( sizeof(int) );
	int beginPos = 0;

	filterImpl( d_Rin, beginPos, rLen, d_Rout, outSize, 
				numThreadPB, numBlock, matchingKeyValue, matchingKeyValue );

	return (*outSize);
}

extern "C"
int GPUOnly_PointSelection( Record* d_Rin, int rLen, int matchingKeyValue, Record** d_Rout, 
						   int numThreadPB, int numBlock)
{
	return point_selection(d_Rin, rLen, matchingKeyValue, d_Rout, numThreadPB, numBlock);
}

extern "C"
int GPUCopy_PointSelection( Record* h_Rin, int rLen, int matchingKeyValue, Record** h_Rout, 
						   int numThreadPB , int numBlock )
{
	Record* d_Rin;
	Record* d_Rout;

	GPUMALLOC( (void**)&d_Rin, sizeof(Record)*rLen );
	TOGPU( d_Rin, h_Rin, sizeof(Record)*rLen );

	int outSize = point_selection(d_Rin, rLen, matchingKeyValue, &d_Rout, numThreadPB, numBlock);

	//(*h_Rout) = (Record*)malloc( sizeof(Record)*outSize );
	CPUMALLOC( (void**)&(*h_Rout), sizeof(Record)*outSize );
	FROMGPU( *h_Rout, d_Rout, sizeof(Record)*outSize );

	GPUFREE( d_Rin );
	GPUFREE( d_Rout );

	return outSize;
}

int range_selection(Record* d_Rin, int rLen, int rangeSmallKey, int rangeLargeKey, Record **d_Rout,
					int numThreadPB , int numBlock )
{
	int* outSize = (int*)malloc( sizeof(int) );
	int beginPos = 0;

	filterImpl( d_Rin, beginPos, rLen, d_Rout, outSize, 
				numThreadPB, numBlock, rangeSmallKey, rangeLargeKey );

	return (*outSize);
}

extern "C"
int GPUOnly_RangeSelection(Record* d_Rin, int rLen, int rangeSmallKey, int rangeLargeKey, Record **d_Rout, 
						   int numThreadPB , int numBlock )
{
	return range_selection(d_Rin, rLen, rangeSmallKey, rangeLargeKey, d_Rout, 
		numThreadPB, numBlock);
}

extern "C"
int GPUCopy_RangeSelection( Record* h_Rin, int rLen, int rangeSmallKey, int rangeLargeKey, Record** h_Rout, 
						   int numThreadPB , int numBlock )
{
	Record* d_Rin;
	Record* d_Rout;
	GPUMALLOC( (void**)&d_Rin, sizeof(Record)*rLen );
	TOGPU( d_Rin, h_Rin, sizeof(Record)*rLen );

	int outSize = range_selection( d_Rin, rLen, rangeSmallKey, rangeLargeKey, &d_Rout, 
		numThreadPB, numBlock);

	//(*h_Rout) = (Record*)malloc( sizeof(Record)*outSize );
	CPUMALLOC( (void**)&(*h_Rout), sizeof(Record)*outSize );
	FROMGPU( *h_Rout, d_Rout, sizeof(Record)*outSize );

	GPUFREE( d_Rin );
	GPUFREE( d_Rout );

	return outSize;
}

/*
aggregation
*/

int agg_max(Record *d_Rin, int rLen, Record* d_Rout, int numThread, int numBlock)
{
	return reduceImpl( d_Rin, rLen, d_Rout, REDUCE_MAX, numThread, numBlock );
}

extern "C"
int GPUOnly_AggMax( Record* d_Rin, int rLen, Record** d_Rout, int numThread, int numBlock )
{
	GPUMALLOC( (void**)d_Rout, sizeof(Record) );
	return agg_max( d_Rin, rLen, *d_Rout, numThread, numBlock );
}

extern "C"
int GPUCopy_AggMax( Record* h_Rin, int rLen, Record** h_Rout, int numThread, int numBlock  )
{
	unsigned int timer = 0;

	startTimer( &timer );
	Record* d_Rin;
	GPUMALLOC( (void**)&d_Rin, sizeof(Record)*rLen );
	Record* d_Rout;
	TOGPU( d_Rin, h_Rin, sizeof(Record)*rLen );
	endTimer( "copy to", &timer );

	startTimer( &timer );
	int result = GPUOnly_AggMax( d_Rin, rLen, &d_Rout, numThread, numBlock );
	endTimer( "aggregation", &timer );

	startTimer( &timer );
	CPUMALLOC( (void**)h_Rout, sizeof(Record) );
	FROMGPU( *h_Rout, d_Rout, sizeof(Record)  );
	endTimer( "copy back", &timer );
	
	GPUFREE( d_Rin );
	GPUFREE( d_Rout );

	return result;
}

int agg_min(Record *d_Rin, int rLen, Record* d_Rout, int numThread, int numBlock )
{
	return reduceImpl( d_Rin, rLen, d_Rout, REDUCE_MIN, numThread, numBlock );
}

extern "C"
int GPUOnly_AggMin( Record* d_Rin, int rLen, Record** d_Rout, int numThread, int numBlock  )
{
	GPUMALLOC( (void**)d_Rout, sizeof(Record) );
	return agg_min( d_Rin, rLen, *d_Rout, numThread, numBlock );
}

extern "C"
int GPUCopy_AggMin( Record* h_Rin, int rLen, Record** h_Rout, int numThread, int numBlock  )
{
	unsigned int timer = 0;

	startTimer( &timer );
	Record* d_Rin;
	GPUMALLOC( (void**)&d_Rin, sizeof(Record)*rLen );
	TOGPU( d_Rin, h_Rin, sizeof(Record)*rLen );
	Record* d_Rout;
	endTimer( "copy to", &timer );
	
	startTimer( &timer );
	int result = GPUOnly_AggMin( d_Rin, rLen, &d_Rout, numThread, numBlock );
	endTimer( "aggregation", &timer );

	startTimer( &timer );
	CPUMALLOC( (void**)h_Rout, sizeof(Record) );
	FROMGPU( *h_Rout, d_Rout, sizeof(Record) );
	endTimer( "copy back", &timer );

	GPUFREE( d_Rin );
	GPUFREE( d_Rout );

	return result;
}

int agg_sum(Record *d_Rin, int rLen, Record* d_Rout, int numThread, int numBlock )
{
	return reduceImpl( d_Rin, rLen, d_Rout, REDUCE_SUM, numThread, numBlock );
}

extern "C"
int GPUOnly_AggSum( Record* d_Rin, int rLen, Record** d_Rout, int numThread, int numBlock )
{
	GPUMALLOC( (void**)d_Rout, sizeof(Record) );
	return agg_sum(d_Rin, rLen, *d_Rout, numThread, numBlock);
}

extern "C"
int GPUCopy_AggSum( Record* h_Rin, int rLen, Record** h_Rout, int numThread, int numBlock )
{
	Record* d_Rin;
	GPUMALLOC( (void**)&d_Rin, sizeof(Record)*rLen );
	TOGPU( d_Rin, h_Rin, sizeof(Record)*rLen );

	Record* d_Rout;

	int result = GPUOnly_AggSum(d_Rin, rLen, &d_Rout, numThread, numBlock);

	CPUMALLOC( (void**)h_Rout, sizeof(Record) );
	FROMGPU( *h_Rout, d_Rout, sizeof(Record) );

	GPUFREE( d_Rin );
	GPUFREE( d_Rout );

	return result;
}

int agg_avg(Record *d_Rin, int rLen, Record* d_Rout, int numThread, int numBlock )
{
	return reduceImpl( d_Rin, rLen, d_Rout, REDUCE_AVERAGE, numThread, numBlock );
}

extern "C"
int GPUOnly_AggAvg( Record* d_Rin, int rLen, Record** d_Rout, int numThread, int numBlock  )
{
	GPUMALLOC( (void**)d_Rout, sizeof(Record) );
	return agg_avg( d_Rin, rLen, *d_Rout, numThread, numBlock );
}

extern "C"
int GPUCopy_AggAvg( Record* h_Rin, int rLen, Record** h_Rout, int numThread, int numBlock  )
{
	Record* d_Rin;
	GPUMALLOC( (void**)&d_Rin, sizeof(Record)*rLen );
	TOGPU( d_Rin, h_Rin, sizeof(Record)*rLen );

	Record* d_Rout;

	int result = GPUOnly_AggAvg( d_Rin, rLen, &d_Rout, numThread, numBlock );

	CPUMALLOC( (void**)h_Rout, sizeof(Record) );
	FROMGPU( *h_Rout, d_Rout, sizeof(Record) ); 

	GPUFREE( d_Rin );
	GPUFREE( d_Rout );

	return result;
}

/*
group by.
return the number of groups.
*/
int groupBy(Record* d_Rin, int rLen, Record * d_Rout, int** d_startPos, int numThread , int numBlock )
{
	return groupByImpl(d_Rin, rLen, d_Rout, d_startPos, numThread, numBlock);;
}

extern "C"
int GPUOnly_GroupBy( Record* d_Rin, int rLen, Record* d_Rout, int** d_startPos, 
					int numThread, int numBlock  )
{
	return groupBy( d_Rin, rLen, d_Rout, d_startPos, numThread, numBlock );
}

extern "C"
int	GPUCopy_GroupBy( Record* h_Rin, int rLen, Record* h_Rout, int** h_startPos, 
					int numThread , int numBlock  )
{
	int memSize = sizeof(Record)*rLen;
	Record* d_Rin;
	Record* d_Rout;
	int* d_startPos;

	GPUMALLOC( (void**)&d_Rin, memSize );
	GPUMALLOC( (void**)&d_Rout, memSize );
	TOGPU( d_Rin, h_Rin, memSize );

	int out = groupBy(d_Rin, rLen, d_Rout, &d_startPos, numThread, numBlock);

	//(*h_startPos) = (int*)malloc( sizeof(int)*out );
	CPUMALLOC( (void**)&(*h_startPos), sizeof(int)*out );
	FROMGPU( *h_startPos, d_startPos, sizeof(int)*out );

	FROMGPU( h_Rout, d_Rout, sizeof(Record)*rLen );

	GPUFREE( d_Rin );
	GPUFREE( d_Rout );
	GPUFREE( d_startPos );

	return out;
}

/*
aggregation after group by.
with the known number of groups, we can allocate the output for advance: d_aggResults.
*/
void agg_max_afterGroupBy(Record *d_Rin, int rLen, int* d_startPos, int numGroups, Record* d_Ragg, int* d_aggResults, int numThread)
{
	aggAfterGroupByImpl(d_Rin, rLen, d_startPos, numGroups, d_Ragg, d_aggResults, REDUCE_MAX, numThread);
}

void agg_min_afterGroupBy(Record *d_Rin, int rLen, int* d_startPos, int numGroups, Record* d_Ragg, int* d_aggResults, int numThread)
{
	aggAfterGroupByImpl(d_Rin, rLen, d_startPos, numGroups, d_Ragg, d_aggResults, REDUCE_MIN, numThread);
}

void agg_sum_afterGroupBy(Record *d_Rin, int rLen, int* d_startPos, int numGroups, Record* d_Ragg, int* d_aggResults, int numThread)
{
	aggAfterGroupByImpl(d_Rin, rLen, d_startPos, numGroups, d_Ragg, d_aggResults, REDUCE_SUM, numThread);
}

void agg_avg_afterGroupBy(Record *d_Rin, int rLen, int* d_startPos, int numGroups, Record* d_Ragg, int* d_aggResults, int numThread)
{
	aggAfterGroupByImpl(d_Rin, rLen, d_startPos, numGroups, d_Ragg, d_aggResults, REDUCE_AVERAGE, numThread);
}


///GPUOnly_ for agg after group by
extern "C"
void GPUOnly_agg_max_afterGroupBy( Record* d_Rin, int rLen, int* d_startPos, int numGroups, Record* d_Ragg, int* d_aggResults, 
								  int numThread  )
{
	agg_max_afterGroupBy(d_Rin, rLen, d_startPos, numGroups, d_Ragg, d_aggResults, numThread);
}

extern "C"
void GPUOnly_agg_min_afterGroupBy( Record* d_Rin, int rLen, int* d_startPos, int numGroups, Record* d_Ragg, int* d_aggResults, 
								  int numThread  )
{
	agg_min_afterGroupBy(d_Rin, rLen, d_startPos, numGroups, d_Ragg, d_aggResults, numThread);
}

extern "C"
void GPUOnly_agg_sum_afterGroupBy( Record* d_Rin, int rLen, int* d_startPos, int numGroups, Record* d_Ragg, int* d_aggResults, 
								  int numThread  )
{
	agg_sum_afterGroupBy(d_Rin, rLen, d_startPos, numGroups, d_Ragg, d_aggResults, numThread);
}

extern "C"
void GPUOnly_agg_avg_afterGroupBy( Record* d_Rin, int rLen, int* d_startPos, int numGroups, Record* d_Ragg, int* d_aggResults, 
								  int numThread  )
{
	agg_avg_afterGroupBy(d_Rin, rLen, d_startPos, numGroups, d_Ragg, d_aggResults, numThread);
}

//GPUCopy_ for agg after group by
extern "C"
void GPUCopy_agg_max_afterGroupBy( Record* h_Rin, int rLen, int* h_startPos, int numGroups, Record* h_Ragg, int* h_aggResults, 
								  int numThread  ) 
{
	Record* d_Rin;
	int* d_startPos;
	Record* d_Ragg;
	int* d_aggResults;

	GPUMALLOC( (void**)&d_Rin, sizeof(Record)*rLen );
	GPUMALLOC( (void**)&d_startPos, sizeof(int)*numGroups );
	GPUMALLOC( (void**)&d_Ragg, sizeof(Record)*rLen );
	GPUMALLOC( (void**)&d_aggResults, sizeof(int)*numGroups );

	TOGPU( d_Rin, h_Rin, sizeof(Record)*rLen );
	TOGPU( d_startPos, h_startPos, sizeof(int)*numGroups );
	TOGPU( d_Ragg, h_Ragg, sizeof(Record)*rLen );

	agg_max_afterGroupBy(d_Rin, rLen, d_startPos, numGroups, d_Ragg, d_aggResults, numThread);

	FROMGPU( h_aggResults, d_aggResults, sizeof(int)*numGroups );

	GPUFREE( d_Rin );
	GPUFREE( d_startPos );
	GPUFREE( d_Ragg );
	GPUFREE( d_aggResults );
}

extern "C"
void GPUCopy_agg_min_afterGroupBy( Record* h_Rin, int rLen, int* h_startPos, int numGroups, Record* h_Ragg, int* h_aggResults, 
								  int numThread  ) 
{
	Record* d_Rin;
	int* d_startPos;
	Record* d_Ragg;
	int* d_aggResults;

	GPUMALLOC( (void**)&d_Rin, sizeof(Record)*rLen );
	GPUMALLOC( (void**)&d_startPos, sizeof(int)*numGroups );
	GPUMALLOC( (void**)&d_Ragg, sizeof(Record)*rLen );
	GPUMALLOC( (void**)&d_aggResults, sizeof(int)*numGroups );

	TOGPU( d_Rin, h_Rin, sizeof(Record)*rLen );
	TOGPU( d_startPos, h_startPos, sizeof(int)*numGroups );
	TOGPU( d_Ragg, h_Ragg, sizeof(Record)*rLen );

	agg_min_afterGroupBy(d_Rin, rLen, d_startPos, numGroups, d_Ragg, d_aggResults, numThread);

	FROMGPU( h_aggResults, d_aggResults, sizeof(int)*numGroups );

	GPUFREE( d_Rin );
	GPUFREE( d_startPos );
	GPUFREE( d_Ragg );
	GPUFREE( d_aggResults );
}

extern "C"
void GPUCopy_agg_sum_afterGroupBy( Record* h_Rin, int rLen, int* h_startPos, int numGroups, Record* h_Ragg, int* h_aggResults, 
								  int numThread  ) 
{
	Record* d_Rin;
	int* d_startPos;
	Record* d_Ragg;
	int* d_aggResults;

	GPUMALLOC( (void**)&d_Rin, sizeof(Record)*rLen );
	GPUMALLOC( (void**)&d_startPos, sizeof(int)*numGroups );
	GPUMALLOC( (void**)&d_Ragg, sizeof(Record)*rLen );
	GPUMALLOC( (void**)&d_aggResults, sizeof(int)*numGroups );

	TOGPU( d_Rin, h_Rin, sizeof(Record)*rLen );
	TOGPU( d_startPos, h_startPos, sizeof(int)*numGroups );
	TOGPU( d_Ragg, h_Ragg, sizeof(Record)*rLen );

	agg_sum_afterGroupBy(d_Rin, rLen, d_startPos, numGroups, d_Ragg, d_aggResults, numThread);

	FROMGPU( h_aggResults, d_aggResults, sizeof(int)*numGroups );

	GPUFREE( d_Rin );
	GPUFREE( d_startPos );
	GPUFREE( d_Ragg );
	GPUFREE( d_aggResults );
}

extern "C"
void GPUCopy_agg_avg_afterGroupBy( Record* h_Rin, int rLen, int* h_startPos, int numGroups, Record* h_Ragg, int* h_aggResults, 
								  int numThread  ) 
{
	Record* d_Rin;
	int* d_startPos;
	Record* d_Ragg;
	int* d_aggResults;

	GPUMALLOC( (void**)&d_Rin, sizeof(Record)*rLen );
	GPUMALLOC( (void**)&d_startPos, sizeof(int)*numGroups );
	GPUMALLOC( (void**)&d_Ragg, sizeof(Record)*rLen );
	GPUMALLOC( (void**)&d_aggResults, sizeof(int)*numGroups );

	TOGPU( d_Rin, h_Rin, sizeof(Record)*rLen );
	TOGPU( d_startPos, h_startPos, sizeof(int)*numGroups );
	TOGPU( d_Ragg, h_Ragg, sizeof(Record)*rLen );

	agg_avg_afterGroupBy(d_Rin, rLen, d_startPos, numGroups, d_Ragg, d_aggResults, numThread);

	FROMGPU( h_aggResults, d_aggResults, sizeof(int)*numGroups );

	GPUFREE( d_Rin );
	GPUFREE( d_startPos );
	GPUFREE( d_Ragg );
	GPUFREE( d_aggResults );
}

/*
joins and sort are defined in other files.
*/

//write your testing code here.
void test_Operators(int argc, char **argv)
{
	int rLen = 1024*1024*16;
	int numThread = 512;
	int numBlock = 256;

	//testGroupByImpl( rLen, numThread, numBlock );
	int OPERATOR = REDUCE_AVERAGE;
	testAggAfterGroupByImpl( rLen, OPERATOR, numThread, numBlock );
}


#endif


