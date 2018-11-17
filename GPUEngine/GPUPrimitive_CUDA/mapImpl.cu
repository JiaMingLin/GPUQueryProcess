#ifndef MAP_IMPL_CU
#define MAP_IMPL_CU

//#include <QP_Utility.cu>

#ifndef COALESCED
__global__ void 
mapImpl_kernel(Record *d_R, int delta, int rLen,int *d_output1, int *d_output2)  
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
		d_output1[pos]=d_R[pos].x;
		d_output2[pos]=d_R[pos].x;
	}	
}
#else
__global__ void 
mapImpl_kernel(Record *d_R, int delta, int rLen,int *d_output1, int *d_output2)  
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
		d_output1[pos]=d_R[pos].x;
		d_output2[pos]=d_R[pos].y;
	}
	
}
#endif

void mapImpl_int(Record *d_R, int rLen, int *d_S1, int* d_S2, int numThreadPB=256, int numBlock=512)
{
#ifndef COALESCED
	printf("NO COALESCED");
#else
	printf("YES COALESCED");
#endif
	int numThreadsPerBlock_x=numThreadPB;
	int numThreadsPerBlock_y=1;
	int numBlock_x=numBlock;
	int numBlock_y=1;
	int numThread=numBlock_x*numThreadsPerBlock_x;
	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	mapImpl_kernel<<<grid,thread>>>(d_R, numThread, rLen, d_S1, d_S2);
}



__global__ void 
mapInit_kernel(Record *d_R, int beginPos, int delta, int rLen,Record value)  
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	for(int pos=resultID+beginPos;pos<rLen;pos+=delta)
	{
		d_R[pos]=value;
	}
	
}







void mapInit(Record *d_R, int beginPos, int rLen, Record value)
{
	int numThreadsPerBlock_x=256;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int numThread=numBlock_x*numThreadsPerBlock_x;
	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	mapInit_kernel<<<grid,thread>>>(d_R, beginPos, numThread, rLen, value);
}

/*
for testing whether the skew in the worklaod for each block will affect the execution time.
*/

__global__ void 
mapTest_kernel(Record *d_R, int beginPos, int delta, int rLen,Record value)  
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	if(bid&1023==1)
	{
		//for(int t=0;t<1024;t++)
		for(int pos=resultID+beginPos;pos<rLen;pos+=delta)
		{
			d_R[pos].x=sqrtf((int)value.x);
			//d_R[pos].y=sqrtf((int)value.y);
		}
	}
	else
	{
		for(int pos=resultID+beginPos;pos<rLen;pos+=delta)
		{
			d_R[pos].x=sqrtf((int)value.x);
			d_R[pos].y=sqrtf((int)value.y);
		}
	}
	
}


void mapTest(Record *d_R, int beginPos, int rLen, Record value)
{
	int numThreadsPerBlock_x=256;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int numThread=numBlock_x*numThreadsPerBlock_x;
	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	mapTest_kernel<<<grid,thread>>>(d_R, beginPos, numThread, rLen, value);
}


#endif

