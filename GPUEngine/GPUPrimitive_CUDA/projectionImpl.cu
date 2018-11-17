#ifndef _PROJECTION_CU_
#define _PROJECTION_CU_


#include "GPU_Dll.h"

__global__
void projection_map_kernel( Record* d_projTable, int pLen, int* d_loc, int* d_temp )
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int gridSize = blockDim.x*gridDim.x;

	for( int idx = bx*blockDim.x + tx; idx < pLen; idx += gridSize )
	{
		d_loc[idx] = d_projTable[idx].x;
		d_temp[idx] = d_projTable[idx].y;
	}
}

void projectionImpl( Record* d_Rin, int rLen, Record* d_projTable, int pLen, 
					int numThread, int numBlock)
{
	int* d_loc;
	GPUMALLOC( (void**)&d_loc, sizeof(int)*pLen ) ;

	//unsigned int timer = 0;

	//startTimer( &timer );
	int* d_temp;
	GPUMALLOC( (void**)&d_temp, sizeof(int)*pLen );
	projection_map_kernel<<<numBlock, numThread>>>( d_projTable, pLen, d_loc, d_temp );
	cudaThreadSynchronize();
	GPUFREE( d_temp );
	//endTimer( "map", &timer );

	gatherImpl( d_Rin, rLen, d_loc, d_projTable,pLen, numThread, numBlock);
	cudaThreadSynchronize();

	GPUFREE( d_loc );
}

extern "C"
void GPUOnly_Projection( Record* d_Rin, int rLen, Record* d_projTable, int pLen, int numThread, int numBlock )
{
	projectionImpl( d_Rin, rLen, d_projTable, pLen, numThread, numBlock);
}

extern "C"
void GPUCopy_Projection( Record* h_Rin, int rLen, Record* h_projTable, int pLen, int numThread, int numBlock )
{
	Record* d_Rin;
	Record* d_projTable;

	unsigned timer = 0;
	startTimer( &timer );
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

	GPUFREE( d_Rin );
	GPUFREE( d_projTable );
}

#endif