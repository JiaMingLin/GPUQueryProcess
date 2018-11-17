#ifndef PICK_BOUND_KERNEL
#define PICK_BOUND_KERNEL

__global__ void countLargeBound_kernel(Record *d_iBound, int delta, int rLen, int threshold, int* d_flag)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int value=0;
	int result=0;
	for(int pos=resultID;pos<rLen;pos+=delta)
	{
		value=d_iBound[pos].y-d_iBound[pos].x;
		if(value>threshold)
			result=1;
		else
			result=0;
		d_flag[pos]=result;
	}
}

__global__ void writeLargeBound_kernel(Record *d_iBound, int delta, int rLen, int* d_flag, int* d_flagSum, Record* d_oBound)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	Record value;
	int targetPos;
	for(int pos=resultID;pos<rLen;pos+=delta)
	{
		value=d_iBound[pos];
		if(d_flag[pos]==1)
		{
			targetPos=d_flagSum[pos];
			d_oBound[targetPos]=value;
		}			
	}
}

//reverse the flag
__global__ void reverseFlag_kernel(int* d_flag, int delta, int rLen)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	//int value=0;
	for(int pos=resultID;pos<rLen;pos+=delta)
	{
		d_flag[pos]=1-d_flag[pos];
	}
}

//write the small bound
__global__ void writeSmallBound_kernel(Record *d_iBound, int delta, int rLen, int numLargeBound, int* d_flag, int* d_flagSum, Record* d_oBound)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	Record value;
	int targetPos;
	for(int pos=resultID;pos<rLen;pos+=delta)
	{
		value=d_iBound[pos];
		if(d_flag[pos]==1)
		{
			targetPos=d_flagSum[pos]+numLargeBound;
			d_oBound[targetPos]=value;
		}			
	}
}

#endif
