#ifndef JOIN_MATCHING_BLOCKS_CU
#define JOIN_MATCHING_BLOCKS_CU

#define NUM_DELTA_PER_BLOCK 8 
#define SMJ_NUM_THREADS_PER_BLOCK 512

#ifdef BINARY_SEARCH
__device__ 
int findNumResultInChunk(Record records[], int key)
{
	int min = 0;
	int max = SMJ_NUM_THREADS_PER_BLOCK;
	int mid;
	int cut;
	while(max - min > 1) {
		mid = (min + max) / 2;
		cut = records[mid].y;

		if(key > cut)
			min = mid;
		else
			max = mid;
	}

	if(records[min].y >= key)
		return min;
	else
		return max;
}
#else
__device__ 
int findNumResultInChunk(Record records[], int key)
{
	int numResult=0;
	for(int i=0;i<SMJ_NUM_THREADS_PER_BLOCK;i++)
		if(records[i].y==key)
			numResult++;
		else
			if(records[i].y>key)
				break;	
	return numResult;
}
#endif

//the best, with shared memory, with coalesced access
__global__ void 
joinMBCount_kernel(Record *d_R, int rLen, Record* d_S, int sLen, 
				   int *d_quanLocS, int numQuan, int *d_n)  
{
	__shared__ Record tempBuf_R[SMJ_NUM_THREADS_PER_BLOCK];
	__shared__ int sStart;
	__shared__ int sEnd;
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int numResult=0;
	if(bid<numQuan)
	{
		if(resultID<rLen)
			tempBuf_R[tid]=d_R[resultID];
		else
			tempBuf_R[tid].y=TEST_MAX;
		if(tid==0)
		{
			sStart=d_quanLocS[bid<<1];
			sEnd=d_quanLocS[(bid<<1)+1];
		}
		__syncthreads();
		int pos=0;
		Record tempValue;
		int i=0;
		int startPos=0;
		for(pos=sStart;(pos+tid)<sEnd;pos+=SMJ_NUM_THREADS_PER_BLOCK)
		{
			tempValue=d_S[pos+tid];
			//for(i=0;i<SMJ_NUM_THREADS_PER_BLOCK;i++)
			//	if(tempValue.y==tempBuf_R[i].y)
			//		numResult++;
			//numResult+=findNumResultInChunk_seq(tempBuf_R,tempValue.y);
			startPos=findNumResultInChunk(tempBuf_R,tempValue.y);
			for(i=startPos;i<SMJ_NUM_THREADS_PER_BLOCK;i++)
				if(tempBuf_R[i].y==tempValue.y)
					numResult++;
				else
					if(tempBuf_R[i].y>tempValue.y)
						break;
		}
	}
	else
		numResult=0;
	d_n[resultID]=numResult;
}

#ifndef SAHRED
__global__ void joinMBCount_kernel_noShared(Record* d_tempBuf_R, Record *d_R, int rLen, Record* d_S, int sLen, 
				   int *d_quanLocS, int numQuan, int *d_n)
{
	//__shared__ Record tempBuf_R[SMJ_NUM_THREADS_PER_BLOCK];
	Record* tempBuf_R;
	tempBuf_R = d_tempBuf_R + blockIdx.x*SMJ_NUM_THREADS_PER_BLOCK;
	__shared__ int sStart;
	__shared__ int sEnd;
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int numResult=0;
	if(bid<numQuan)
	{
		if(resultID<rLen)
			tempBuf_R[tid]=d_R[resultID];
		else
			tempBuf_R[tid].y=TEST_MAX;

		if(tid==0)
		{
			sStart=d_quanLocS[bid<<1];
			sEnd=d_quanLocS[(bid<<1)+1];
		}
		__syncthreads();
		int pos=0;
		Record tempValue;
		int i=0;
		int startPos=0;
		for(pos=sStart;(pos+tid)<sEnd;pos+=SMJ_NUM_THREADS_PER_BLOCK)
		{
			tempValue=d_S[pos+tid];
			//for(i=0;i<SMJ_NUM_THREADS_PER_BLOCK;i++)
			//	if(tempValue.y==tempBuf_R[i].y)
			//		numResult++;
			//numResult+=findNumResultInChunk_seq(tempBuf_R,tempValue.y);
			startPos=findNumResultInChunk(tempBuf_R,tempValue.y);
			for(i=startPos;i<SMJ_NUM_THREADS_PER_BLOCK;i++)
				if(tempBuf_R[i].y==tempValue.y)
					numResult++;
				else
					if(tempBuf_R[i].y>tempValue.y)
						break;
		}
	}
	else
		numResult=0;
	d_n[resultID]=numResult;
}
#endif

#ifndef COALESCED
	__global__ void 
	joinMBCount_noCoalesced_kernel(Record *d_R, int rLen, Record* d_S, int sLen, 
					   int *d_quanLocS, int numQuan, int *d_n)  
	{
		__shared__ Record tempBuf_R[SMJ_NUM_THREADS_PER_BLOCK];
		__shared__ int sStart;
		__shared__ int sEnd;
		const int by = blockIdx.y;
		const int bx = blockIdx.x;
		const int tx = threadIdx.x;
		const int ty = threadIdx.y;	
		const int tid=tx+ty*blockDim.x;
		const int bid=bx+by*gridDim.x;
		const int numThread=blockDim.x;
		const int resultID=(bid)*numThread+tid;
		int numResult=0;
		if(bid<numQuan)
		{
			if(resultID<rLen)
				tempBuf_R[tid]=d_R[resultID];
			else
				tempBuf_R[tid].y=TEST_MAX;
			if(tid==0)
			{
				sStart=d_quanLocS[bid<<1];
				sEnd=d_quanLocS[(bid<<1)+1];
			}
			__syncthreads();
			int pos=0;
			Record tempValue;
			int i=0;
			int startPos=0;

			int len = (sEnd - sStart)/numThread;
			int start = sStart + len*tid;
			int end = start + len;
			if( tid == (numThread - 1) )
			{
				end = sEnd;
			}

			//for(pos=sStart + tid; pos < sEnd;pos+=SMJ_NUM_THREADS_PER_BLOCK)
			for(pos=start; pos < end;pos++)
			{
				tempValue=d_S[pos];
				startPos=findNumResultInChunk(tempBuf_R,tempValue.y);

				for(i=startPos;i<SMJ_NUM_THREADS_PER_BLOCK;i++)
					if(tempBuf_R[i].y==tempValue.y)
						numResult++;
					else
						if(tempBuf_R[i].y>tempValue.y)
							break;
			}
		}
		else
			numResult=0;
		d_n[resultID]=numResult;
	}
#endif


//best, with shared memory, with coalesced
	__global__ void 
joinMBWrite_kernel(Record *d_R, int rLen, Record* d_S, int sLen, 
				   int *d_quanLocS, int numQuan, int *d_sum, Record *d_output)  
{
	__shared__ Record tempBuf_R[SMJ_NUM_THREADS_PER_BLOCK];
	__shared__ int sStart;
	__shared__ int sEnd;
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	//int numResult=0;
	if(bid<numQuan)
	{
		if(resultID<rLen)
			tempBuf_R[tid]=d_R[resultID];
		else
			tempBuf_R[tid].y=TEST_MAX;
		if(tid==0)
		{
			sStart=d_quanLocS[bid<<1];
			sEnd=d_quanLocS[(bid<<1)+1];
		}
		__syncthreads();
		int pos=0;
		Record tempValue;
		int i=0;
		int startPos=0;
		int base=d_sum[resultID];
		for(pos=sStart;(pos+tid)<sEnd;pos+=SMJ_NUM_THREADS_PER_BLOCK)
		{
			tempValue=d_S[pos+tid];
			//for(i=0;i<SMJ_NUM_THREADS_PER_BLOCK;i++)
			//	if(tempValue.y==tempBuf_R[i].y)
			//		numResult++;
			//numResult+=findNumResultInChunk_seq(tempBuf_R,tempValue.y);
			startPos=findNumResultInChunk(tempBuf_R,tempValue.y);
			for(i=startPos;i<SMJ_NUM_THREADS_PER_BLOCK;i++)
				if(tempBuf_R[i].y==tempValue.y)
				{
					d_output[base].x=tempBuf_R[i].x;
					d_output[base].y=tempValue.x;
					base++;
				}
				else
					if(tempBuf_R[i].y>tempValue.y)
						break;
		}
	}
}

#ifndef COALESCED
	__global__ void 
	joinMBWrite_noCoalesced_kernel(Record *d_R, int rLen, Record* d_S, int sLen, 
					   int *d_quanLocS, int numQuan, int *d_sum, Record *d_output)  
	{
		__shared__ Record tempBuf_R[SMJ_NUM_THREADS_PER_BLOCK];
		__shared__ int sStart;
		__shared__ int sEnd;
		const int by = blockIdx.y;
		const int bx = blockIdx.x;
		const int tx = threadIdx.x;
		const int ty = threadIdx.y;	
		const int tid=tx+ty*blockDim.x;
		const int bid=bx+by*gridDim.x;
		const int numThread=blockDim.x;
		const int resultID=(bid)*numThread+tid;
		//int numResult=0;
		if(bid<numQuan)
		{
			if(resultID<rLen)
				tempBuf_R[tid]=d_R[resultID];
			else
				tempBuf_R[tid].y=TEST_MAX;
			if(tid==0)
			{
				sStart=d_quanLocS[bid<<1];
				sEnd=d_quanLocS[(bid<<1)+1];
			}
			__syncthreads();
			int pos=0;
			Record tempValue;
			int i=0;
			int startPos=0;
			int base=d_sum[resultID];

			int len = (sEnd - sStart)/numThread;
			int start = sStart + len*tid;
			int end = start + len;
			if( tid == (numThread - 1) )
			{
				end = sEnd;
			}

			for(pos = start; pos < end; pos++ )
			{
				tempValue=d_S[pos];

				startPos=findNumResultInChunk(tempBuf_R,tempValue.y);
				for(i=startPos;i<SMJ_NUM_THREADS_PER_BLOCK;i++)
					if(tempBuf_R[i].y==tempValue.y)
					{
						d_output[base].x=tempBuf_R[i].x;
						d_output[base].y=tempValue.x;
						base++;
					}
					else
						if(tempBuf_R[i].y>tempValue.y)
							break;
			}
		}
	}
#endif

#ifndef SHARED_MEM
	__global__ void 
	joinMBWrite_kernel_noShared(Record* d_tempBuf_R, Record *d_R, int rLen, Record* d_S, int sLen, 
					   int *d_quanLocS, int numQuan, int *d_sum, Record *d_output)  
	{
		//__shared__ Record tempBuf_R[SMJ_NUM_THREADS_PER_BLOCK];
		Record* tempBuf_R;
		tempBuf_R = d_tempBuf_R + blockIdx.x*SMJ_NUM_THREADS_PER_BLOCK;
		__shared__ int sStart;
		__shared__ int sEnd;
		const int by = blockIdx.y;
		const int bx = blockIdx.x;
		const int tx = threadIdx.x;
		const int ty = threadIdx.y;	
		const int tid=tx+ty*blockDim.x;
		const int bid=bx+by*gridDim.x;
		const int numThread=blockDim.x;
		const int resultID=(bid)*numThread+tid;
		//int numResult=0;
		if(bid<numQuan)
		{
			if(resultID<rLen)
				tempBuf_R[tid]=d_R[resultID];
			else
				tempBuf_R[tid].y=TEST_MAX;
			if(tid==0)
			{
				sStart=d_quanLocS[bid<<1];
				sEnd=d_quanLocS[(bid<<1)+1];
			}
			__syncthreads();
			int pos=0;
			Record tempValue;
			int i=0;
			int startPos=0;
			int base=d_sum[resultID];
			for(pos=sStart;(pos+tid)<sEnd;pos+=SMJ_NUM_THREADS_PER_BLOCK)
			{
				tempValue=d_S[pos+tid];
				//for(i=0;i<SMJ_NUM_THREADS_PER_BLOCK;i++)
				//	if(tempValue.y==tempBuf_R[i].y)
				//		numResult++;
				//numResult+=findNumResultInChunk_seq(tempBuf_R,tempValue.y);
				startPos=findNumResultInChunk(tempBuf_R,tempValue.y);
				for(i=startPos;i<SMJ_NUM_THREADS_PER_BLOCK;i++)
					if(tempBuf_R[i].y==tempValue.y)
					{
						d_output[base].x=tempBuf_R[i].x;
						d_output[base].y=tempValue.x;
						base++;
					}
					else
						if(tempBuf_R[i].y>tempValue.y)
							break;
			}
		}
	}
#endif

int joinMatchingBlocks(Record *d_R, int rLen, Record *d_S, int sLen, 
					   int *d_quanLocS, int numQuan, Record** d_Rout)
{
	int numResults=0;
	int numThreadPerBlock =SMJ_NUM_THREADS_PER_BLOCK;
	int numBlock_X=numQuan;
	int numBlock_Y=1;
	if(numBlock_X>NLJ_MAX_NUM_BLOCK_PER_DIM)
	{
		numBlock_Y=numBlock_X/NLJ_MAX_NUM_BLOCK_PER_DIM;
		if(numBlock_X%NLJ_MAX_NUM_BLOCK_PER_DIM!=0)
			numBlock_Y++;
		numBlock_X=NLJ_MAX_NUM_BLOCK_PER_DIM;
	}
	dim3  threads_NLJ( numThreadPerBlock, 1, 1);
	dim3  grid_NLJ( numBlock_X, numBlock_Y, 1);
	int resultBuf=grid_NLJ.x*grid_NLJ.y*numThreadPerBlock;
	printf("numThreadPerBlock,%d,  numBlock_X, %d, numBlock_Y, %d\n", numThreadPerBlock, numBlock_X,numBlock_Y);
	int* d_n;
	GPUMALLOC((void**)&d_n, sizeof(int)*resultBuf );
	//the prefix sum for d_n
	int *d_sum;//the prefix sum for d_n[1,...,n]
	GPUMALLOC((void**)&d_sum, sizeof(int)*resultBuf );
	
	
	int* h_n ;
	CPUMALLOC((void**)&h_n, sizeof(int));
	int* h_sum ;
	CPUMALLOC((void**)&h_sum, sizeof(int));

	unsigned int timer=0;
	//saven_initialPrefixSum(resultBuf);	
	startTimer(&timer);
#ifdef SHARED_MEM
	printf( "YES, SHARED MEMORY, joinMBCount \n" );
	#ifdef COALESCED
		printf( "YES, COALESCED, joinMBCount\n" );
		joinMBCount_kernel<<< grid_NLJ, threads_NLJ >>>(d_R, rLen, d_S, sLen, 
				d_quanLocS, numQuan, d_n);
	#else
		printf( "NO COALESCED, joinMBCount\n" );
		joinMBCount_noCoalesced_kernel<<< grid_NLJ, threads_NLJ >>>(d_R, rLen, d_S, sLen, 
				d_quanLocS, numQuan, d_n);
#endif
#else
	printf( "NO SHARED MEMORY, jonMBCount \n" );
	Record* d_tempBuf_R;
	GPUMALLOC( (void**)&d_tempBuf_R, sizeof(Record)*SMJ_NUM_THREADS_PER_BLOCK*grid_NLJ.x );
	joinMBCount_kernel_noShared<<< grid_NLJ, threads_NLJ >>>(d_tempBuf_R, d_R, rLen, d_S, sLen, 
			d_quanLocS, numQuan, d_n);
	GPUFREE( d_tempBuf_R );
#endif
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	endTimer("joinMBCount_kernel", &timer);
	//gpuPrint(d_n,512,"d_n");
	startTimer(&timer);
	//prescanArray( d_sum,d_n, resultBuf);
	scanImpl(d_n, resultBuf, d_sum);
	FROMGPU(h_n, (d_n+resultBuf-1), sizeof(int));
	FROMGPU(h_sum, (d_sum+resultBuf-1), sizeof(int));
	numResults=*h_n+*h_sum;
	printf("numResults=%d, ", numResults);
	endTimer("prescanArray", &timer);
	Record *d_outBuf;
	if(numResults>0)
	{
		GPUMALLOC((void**) &d_outBuf, sizeof(Record)*numResults );
		*d_Rout=d_outBuf;
		startTimer(&timer);
#ifdef SHARED_MEM
		printf( "YES, SHARED MEMORY, joinMBWrite\n" );
	#ifdef COALESCED
			printf( "YES, COALESCED, joinMBWrite\n" );
			joinMBWrite_kernel<<< grid_NLJ, threads_NLJ >>>(d_R, rLen, d_S, sLen, 
				d_quanLocS, numQuan,d_sum, d_outBuf);
	#else
			printf( "NO COALESCED, joinMBWrite\n" );
			joinMBWrite_noCoalesced_kernel<<< grid_NLJ, threads_NLJ >>>(d_R, rLen, d_S, sLen, 
				d_quanLocS, numQuan,d_sum, d_outBuf);
	#endif
#else
		printf( "NO SHARED MEMORY, joinMBWrite\n" );
		Record* d_tempBuf_R;
		GPUMALLOC( (void**)&d_tempBuf_R, sizeof(Record)*SMJ_NUM_THREADS_PER_BLOCK*grid_NLJ.x );
		joinMBWrite_kernel_noShared<<< grid_NLJ, threads_NLJ >>>(d_tempBuf_R, d_R, rLen, d_S, sLen, 
			d_quanLocS, numQuan,d_sum, d_outBuf);
		GPUFREE( d_tempBuf_R );

#endif
		CUDA_SAFE_CALL(cudaThreadSynchronize());
		endTimer("joinMBWrite_kernel", &timer);
	}
	GPUFREE(d_n);
	GPUFREE(d_sum);
	CPUFREE(h_n);
	CPUFREE(h_sum);
	return numResults;
}

#endif

