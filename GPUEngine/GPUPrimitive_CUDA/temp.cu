typedef int2 Record;
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
