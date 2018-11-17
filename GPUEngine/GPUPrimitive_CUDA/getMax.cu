#ifndef GET_MAX_CU
#define GET_MAX_CU

#define NUM_THREAD_PER_BLOCK_MAX_MAP 512

__global__ void 
maxMap_kernel(Record *d_R, int delta, int rLen,int *d_output)  
{
	__shared__ Record partialMax[NUM_THREAD_PER_BLOCK_MAX_MAP];
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	if(resultID<rLen)
	{
		Record value=d_R[resultID];
		Record tempValue;
		for(int pos=resultID+delta;pos<rLen;pos+=delta)
		{
			tempValue=d_R[pos];
			if(tempValue.y>value.y)
			{
				value=tempValue;
			}
		}
		partialMax[tid]=value;
		__syncthreads();
		for(int d = (numThread >>1); d > 0; d >>= 1)
		{
			__syncthreads();
			if(tid < d)
				partialMax[tid].y = max(partialMax[tid].y, partialMax[tid + d].y);
		}
		if(tid == 0)
			d_output[bid] = partialMax[0].y;
	}
	else
		if(tid == 0)
		d_output[bid] = -1;
		
}

int getMax(Record *d_R, int rLen)
{
	int numThreadsPerBlock_x=256;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int numThread=numBlock_x*numThreadsPerBlock_x;
	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int memSize=numBlock_x*sizeof(int);
	int *d_output;
	GPUMALLOC((void**)&d_output, memSize);
	maxMap_kernel<<<grid,thread>>>(d_R, numThread, rLen, d_output);
	int *h_output;
	CPUMALLOC((void**)&h_output, memSize);
	FROMGPU(h_output,d_output, memSize);
	int maxValue=h_output[0];
	int i=0;
	for(i=1;i<numBlock_x;i++)
		if(maxValue<h_output[i])
			maxValue=h_output[i];
	GPUFREE(d_output);
	CPUFREE(h_output);
	return maxValue;
}


#endif

