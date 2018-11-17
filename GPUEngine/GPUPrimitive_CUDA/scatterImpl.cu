#ifndef SCATTER_IMPL_CU
#define SCATTER_IMPL_CU

__global__ void
optScatter( Record *d_R, int delta, int rLen, int *loc, int from, int to, Record *d_S)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int targetLoc=0;
	for(int pos=resultID;pos<rLen;pos+=delta)
	{
		targetLoc=loc[pos];
		if(targetLoc>=from && targetLoc<to)
		d_S[targetLoc]=d_R[pos];
	}	
}


__global__ void
optGather( Record *d_R, int delta, int rLen, int *loc, int from, int to, Record *d_S, int sLen)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int targetLoc=0;
	for(int pos=resultID;pos<sLen;pos+=delta)
	{
		targetLoc=loc[pos];
		if(targetLoc>=from && targetLoc<to)
		d_S[pos]=d_R[targetLoc];
	}	
}

#ifndef COALESCED
	__global__ void
	optScatter_noCoalesced( Record *d_R, int delta, int rLen, int *loc, int from, int to, Record *d_S)
	{
		int numThread = blockDim.x;
		int numBlock = gridDim.x;
		int tid = blockIdx.x*numThread + threadIdx.x;
		int len = rLen/( numThread*numBlock );
		int start = tid*len;
		int end = start + len;

		__syncthreads();
		int targetLoc = 0;
		for( int pos = start; pos < end; pos++ )
		{
			targetLoc=loc[pos];
			if(targetLoc>=from && targetLoc<to)
			{
				d_S[targetLoc].x = d_R[pos].x;
				d_S[targetLoc].y = d_R[pos].y;
			}
			__syncthreads();
		}
	}

	__global__ void 
	optGather_noCoalesced( Record *d_R, int delta, int rLen, int *loc, int from, int to, Record *d_S, int sLen)
	{
		int numThread = blockDim.x;
		int numBlock = gridDim.x;
		int tid = blockIdx.x*numThread + threadIdx.x;
		int len = rLen/( numThread*numBlock );
		int start = tid*len;
		int end = start + len;

		//__syncthreads();
		int targetLoc = 0;
		for( int pos = start; pos < end; pos++ )
		{
			targetLoc=loc[pos];
			if(targetLoc>=from && targetLoc<to)
			d_S[pos]=d_R[targetLoc];
		}
	}

#endif


void scatterImpl(Record *d_R, int rLen, int *d_loc, Record *d_S, int numThreadPB=256, int numBlock=512)
{
	int numRun=8;
	if(rLen<256*1024)
		numRun=1;
	else
		if(rLen<1024*1024)
			numRun=2;
		else
			if(rLen<8192*1024)
				numRun=4;
	int runSize=rLen/numRun;	
	if(rLen%numRun!=0)
		runSize+=1;
	printf("run, %d\n", numRun);
	int from, to;
	int numThreadsPerBlock_x=numThreadPB;
	int numThreadsPerBlock_y=1;
	int numBlock_x=numBlock;
	int numBlock_y=1;
	int numThread=numBlock_x*numThreadsPerBlock_x;
	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);

#ifdef COALESCED
	printf( "YES, COALESCED, scatter\n" );
#else
	printf( "NO COALESCED, scatter\n" );
#endif

	for(int i=0;i<numRun;i++)
	{
		from=i*runSize;
		to=(i+1)*runSize;
#ifdef COALESCED
		optScatter<<<grid,thread>>>(d_R,numThread, rLen, d_loc,from, to, d_S);
#else
		optScatter_noCoalesced<<<grid,thread>>>(d_R,numThread, rLen, d_loc,from, to, d_S);
#endif
		CUDA_SAFE_CALL(cudaThreadSynchronize());
	}
}

void scatterImpl_forPart(Record *d_R, int rLen, int numPart, int *d_loc, Record *d_S)
{
	int numRun=8;
	if(numPart<=8)
		numRun=1;
	else if(numPart<=16)
			numRun=2;
	else if(numPart<=32)
			numRun=4;
	else
		numRun=8;
	int runSize=rLen/numRun;
	if(rLen%numRun!=0)
		runSize+=1;
	printf("run, %d\n", numRun);
	int from, to;
	int numThreadsPerBlock_x=256;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int numThread=numBlock_x*numThreadsPerBlock_x;
	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	for(int i=0;i<numRun;i++)
	{
		from=i*runSize;
		to=(i+1)*runSize;
		optScatter<<<grid,thread>>>(d_R,numThread, rLen, d_loc,from, to, d_S);
	}
}


void gatherImpl(Record *d_R, int rLen, int *d_loc, Record *d_S, int sLen, int numThreadsPerBlock_x = 32, int numBlock_x = 64)
{
	int numRun=8;
	if(sLen<256*1024)
		numRun=1;
	else
		if(sLen<1024*1024)
			numRun=2;
		else
			if(sLen<8192*1024)
				numRun=4;
	printf("run, %d\n", numRun);
	int runSize=rLen/numRun;	
	if(rLen%numRun!=0)
		runSize+=1;
	int from, to;
	//int numThreadsPerBlock_x=256;
	int numThreadsPerBlock_y=1;
	//int numBlock_x=512;
	int numBlock_y=1;
	int numThread=numBlock_x*numThreadsPerBlock_x;
	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);

#ifdef COALESCED
	printf( "YES, COALESCED, \n" );
#else
	printf( "NO COALESCED, \n" );
#endif

	for(int i=0;i<numRun;i++)
	{
		from=i*runSize;
		to=(i+1)*runSize;
#ifdef COALESCED
		optGather<<<grid,thread>>>(d_R,numThread, rLen, d_loc,from, to, d_S,sLen);
#else
		optGather_noCoalesced<<<grid,thread>>>(d_R,numThread, rLen, d_loc,from, to, d_S,sLen);
#endif
		CUDA_SAFE_CALL(cudaThreadSynchronize());
	}
}

	

#endif
