#ifndef GET_QUANTILE_CU
#define GET_QUANTILE_CU



__global__ void 
quanMap_kernel(Record *d_R, int interval, int rLen,int *d_output)  
{
	extern __shared__ Record tempBuf[];
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	if(resultID<rLen)
		tempBuf[tid]=d_R[resultID];
	__syncthreads();
	if(tid==0)
	{
		Record value;
		value.x=tempBuf[0].y;
		int numElement=0;
		if(resultID+interval>rLen)
			numElement=rLen-resultID;
		else
			numElement=interval;
		value.y=tempBuf[numElement-1].y+1;
		int curQuanPos=(resultID/interval)<<1;
		d_output[curQuanPos]=value.x;			
		d_output[curQuanPos+1]=value.y;			
	}

}

void getQuantile(Record *d_R, int rLen, int interval, int* d_output, int numQuantile)
{
	int numThreadsPerBlock_x=interval;
	int numThreadsPerBlock_y=1;
	int numBlock_X=divRoundUp(rLen, interval);
	int numBlock_Y=1;
	if(numBlock_X>NLJ_MAX_NUM_BLOCK_PER_DIM)
	{
		numBlock_Y=numBlock_X/NLJ_MAX_NUM_BLOCK_PER_DIM;
		if(numBlock_X%NLJ_MAX_NUM_BLOCK_PER_DIM!=0)
			numBlock_Y++;
		numBlock_X=NLJ_MAX_NUM_BLOCK_PER_DIM;
	}
	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_X, numBlock_Y , 1);
	quanMap_kernel<<<grid,thread, interval*sizeof(Record)>>>(d_R, interval,rLen, d_output);
}


#endif

