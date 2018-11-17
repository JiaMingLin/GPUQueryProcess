#ifndef TABLE_OP_IMPL_CU
#define TABLE_OP_IMPL_CU

#include "GPU_Dll.h"

/*
* the kernels.
*/

//kernel for interface 1:GPUOnly_getRIDList
__global__ void 
getRIDList_kernel(Record *d_R, int delta, int rLen,int *d_RIDList, int *d_output)  
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	//Record value;
	for(int pos=resultID;pos<rLen;pos+=delta)
	{
		//value=d_R[pos];
		d_RIDList[pos]=d_R[pos].x;
		d_output[pos]=d_R[pos].y;
	}	
}

//kernel for interface 2:GPUOnly_copyRelation
__global__ void 
copyRelation_kernel(Record *d_R, int delta, int rLen, Record* d_output)  
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	//Record value;
	for(int pos=resultID;pos<rLen;pos+=delta)
	{
		d_output[pos]=d_R[pos];
	}	
}


//kernel for interface 3:GPUOnly_setRIDList
__global__ void 
setRIDList_kernel(int *d_RIDList, int *d_intput, int delta, int rLen, Record *d_R)  
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	//Record value;
	for(int pos=resultID;pos<rLen;pos+=delta)
	{
		//value=d_R[pos];
		d_R[pos].x=d_RIDList[pos];
		d_R[pos].y=d_intput[pos];
	}	
}


//kernel for interface 3:GPUOnly_setRIDList
__global__ void 
setValueList_kernel(int *d_ValueList, int delta, int rLen, Record *d_R)  
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	//Record value;
	for(int pos=resultID;pos<rLen;pos+=delta)
	{
		//value=d_R[pos];
		d_R[pos].x=pos;
		d_R[pos].y=d_ValueList[pos];
	}	
}


/*
* the interfaces.
*/


//Interface 1: get all RIDs into an array. You need to allocate d_RIDList.
extern "C"
void GPUOnly_getRIDList(Record* d_Rin, int rLen, int** d_RIDList, int numThreadPerBlock, int numBlock)
{
	int outputSize=sizeof(int)*rLen;
	GPUMALLOC((void**)d_RIDList, outputSize);
	int* d_tempOutput;
	GPUMALLOC((void**)&d_tempOutput, outputSize);

	int numThread=numThreadPerBlock*numBlock;
	dim3  thread( numThreadPerBlock, 1, 1);
	dim3  grid( numBlock, 1 , 1);
	getRIDList_kernel<<<grid,thread>>>(d_Rin, numThread, rLen, *d_RIDList, d_tempOutput);
	GPUFREE(d_tempOutput);
}

//Interface 2: copy a relation to another relation. You need to allocate d_destRin.
extern "C"
void GPUOnly_copyRelation(Record* d_srcRin, int rLen, Record** d_destRin, int numThreadPerBlock, int numBlock)
{
	int outputSize=sizeof(Record)*rLen;
	GPUMALLOC((void**)d_destRin, outputSize);
	
	int numThread=numThreadPerBlock*numBlock;
	dim3  thread( numThreadPerBlock, 1, 1);
	dim3  grid( numBlock, 1 , 1);
	copyRelation_kernel<<<grid,thread>>>(d_srcRin, numThread, rLen, *d_destRin);
}

//Interface3: set the RID according to the RID list.
extern "C"
void GPUOnly_setRIDList(int* d_RIDList, int rLen, Record* d_destRin, int numThreadPerBlock, int numBlock)
{
	int outputSize=sizeof(int)*rLen;
	int* d_tempOutput;
	GPUMALLOC((void**)&d_tempOutput, outputSize);

	int numThread=numThreadPerBlock*numBlock;
	dim3  thread( numThreadPerBlock, 1, 1);
	dim3  grid( numBlock, 1 , 1);
	setRIDList_kernel<<<grid,thread>>>(d_RIDList, d_tempOutput, numThread, rLen, d_destRin);
	GPUFREE(d_tempOutput);
}

void GPUOnly_setValueList(int* d_ValueList, int rLen, Record* d_destRin, int numThreadPerBlock , int numBlock)
{
	int numThread=numThreadPerBlock*numBlock;
	dim3  thread( numThreadPerBlock, 1, 1);
	dim3  grid( numBlock, 1 , 1);
	setValueList_kernel<<<grid,thread>>>(d_ValueList, numThread, rLen, d_destRin);
}

__global__ void
getValueList_kernel(Record *d_R, int delta, int rLen,int *d_ValueList, int *d_output) 

{

	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y; 

	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;

	//Record value;

	for(int pos=resultID;pos<rLen;pos+=delta)
	{
		d_output[pos]=d_R[pos].x;

		d_ValueList[pos]=d_R[pos].y;                   
	}          
}


extern "C"
void GPUOnly_getValueList(Record* d_Rin, int rLen, int** d_ValueList, int numThreadPerBlock, int numBlock)
{
	int outputSize=sizeof(int)*rLen;
	GPUMALLOC((void**)d_ValueList, outputSize);
	int* d_tempOutput;
	GPUMALLOC((void**)&d_tempOutput, outputSize);

	int numThread=numThreadPerBlock*numBlock;
	dim3  thread( numThreadPerBlock, 1, 1);
	dim3  grid( numBlock, 1 , 1);

	getValueList_kernel<<<grid,thread>>>(d_Rin, numThread, rLen, *d_ValueList, d_tempOutput);

	GPUFREE(d_tempOutput);
}

#endif

