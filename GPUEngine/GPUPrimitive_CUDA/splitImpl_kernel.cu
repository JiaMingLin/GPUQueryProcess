#ifndef SPLIT_IMPL_KERNEL_CU
#define SPLIT_IMPL_KERNEL_CU


__device__ int getPartID(int key, int numPart)
{
	return (key%numPart);
}

__device__
unsigned int RSHash(int value, int mask)
{
    unsigned int b=378551;
    unsigned int a=63689;
    unsigned int hash=0;

    int i=0;

	for(i=0;i<4;i++)
    {

        hash=hash*a+(value>>(24-(i<<3)));
        a*=b;
    }

    return (hash & mask);
}

#ifdef COALESCED
__global__ 
void partition_kernel(Record *d_R, int delta, int rLen, int numPart, int *d_output1, int *d_output2)  
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;

	for(int pos=resultID;pos<rLen;pos+=delta)
	{
		d_output1[pos] = d_R[pos].x;
		d_output2[pos] = RSHash(d_R[pos].y, numPart - 1);
	}	
}

__global__ void 
mapPart_kernel(Record *d_R, int delta, int rLen, int numPart, int *d_output1, int *d_output2)  
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	for(int pos=resultID;pos<rLen;pos+=delta)
	{
		d_output1[pos]=d_R[pos].x;
		d_output2[pos]=getPartID(d_R[pos].y, numPart);
	}	
}
#else //NO COALESCED
	__global__ 
	void partition_kernel(Record *d_R, int delta, int rLen, int numPart, int *d_output1, int *d_output2)  
	{
		int numThread = blockDim.x;
		int numBlock = gridDim.x;
		int tid = blockIdx.x*numThread + threadIdx.x;
		int len = rLen/( numThread*numBlock );
		int start = tid*len;
		int end = start + len;
		if( (threadIdx.x == numThread - 1) && (blockIdx.x == numBlock - 1) )
		{
			end = rLen;
		}

		__syncthreads();
		for( int idx = start; idx < end; idx++ )
		{
			d_output1[idx] = d_R[idx].x;
			d_output2[idx] = RSHash(d_R[idx].y, numPart - 1);	
		}
	}

	__global__ void 
	mapPart_kernel(Record *d_R, int delta, int rLen, int numPart, int *d_output1, int *d_output2)  
	{
		int numThread = blockDim.x;
		int numBlock = gridDim.x;
		int tid = blockIdx.x*numThread + threadIdx.x;
		int len = rLen/( numThread*numBlock );
		int start = tid*len;
		int end = start + len;
		if( (threadIdx.x == numThread - 1) && (blockIdx.x == numBlock - 1) )
		{
			end = rLen;
		}

		for( int idx = start; idx < end; idx++ )
		{
			d_output1[idx] = d_R[idx].x;
			d_output2[idx] = getPartID(d_R[idx].y, numPart);
		}	
	}
#endif

//use the shared memory.
#ifdef SHARED_MEM
	//compute the histogram.
	//d_hist layout: d_hist[thread global ID + total number threads*partition ID] is the count.
#ifdef COALESCED
	__global__ void 
	countHist_kernel(int *d_pidArray, int delta, int rLen, int numPart, int *d_hist)
	{
		extern __shared__ int shared_hist[];
		const int by = blockIdx.y;
		const int bx = blockIdx.x;
		const int tx = threadIdx.x;
		const int ty = threadIdx.y;	
		const int tid=tx+ty*blockDim.x;
		const int bid=bx+by*gridDim.x;
		const int numThread=blockDim.x;
		const int resultID=(bid)*numThread+tid;

		int pid=0;
		int offset=tid*numPart;
		
		for(pid=0;pid<numPart;pid++)
		{
			shared_hist[pid+offset]=0;
		}
			

		for(int pos=resultID;pos<rLen;pos+=delta)
		{
			pid=d_pidArray[pos];
			shared_hist[pid+offset]++;
		}

		for(pid=0;pid<numPart;pid++)
		{
			d_hist[resultID+delta*pid]=shared_hist[pid+offset];
		}			
	}
	//having the prefix sum, compute the write location.
	__global__ void 
	writeHist_kernel(int *d_pidArray, int delta, int rLen, int numPart, int *d_psSum, int* d_loc)
	{
		extern __shared__ int shared_hist[];
		const int by = blockIdx.y;
		const int bx = blockIdx.x;
		const int tx = threadIdx.x;
		const int ty = threadIdx.y;	
		const int tid=tx+ty*blockDim.x;
		const int bid=bx+by*gridDim.x;
		const int numThread=blockDim.x;
		const int resultID=(bid)*numThread+tid;
		int pid=0;
		int offset=tid*numPart;
		for(pid=0;pid<numPart;pid++)
			shared_hist[pid+offset]=d_psSum[resultID+delta*pid];
		for(int pos=resultID;pos<rLen;pos+=delta)
		{
			pid=d_pidArray[pos];
			d_loc[pos]=shared_hist[pid+offset];
			shared_hist[pid+offset]++;
		}
	}


	/*
	//
	//
	// multiple block version
	*/

	__global__ void 
	countHist_MB_kernel(int *d_pidArray, Record *d_iBound, int numPart, int *d_hist)
	{
		extern __shared__ int shared_hist[];//the first two is pStart, pEnd
		const int by = blockIdx.y;
		const int bx = blockIdx.x;
		const int tx = threadIdx.x;
		const int ty = threadIdx.y;	
		const int tid=tx+ty*blockDim.x;
		//const int bid=bx+by*gridDim.x;
		const int numThreadPerPartition=blockDim.x*gridDim.x*blockDim.y;
		const int resultID=bx*blockDim.x+tid; //!! not bid!!, the thread ID within the x blocks.
		int pid=0;
		int offsetInBlock=tid*numPart+2;
		//(numPart*numThreadPerPartition) is the number of entries for each partition. 
		int offsetGlobal=numPart*numThreadPerPartition*by;
		for(pid=0;pid<numPart;pid++)
			shared_hist[pid+offsetInBlock]=0;
		if(tid==0)
		{
			Record tempValue=d_iBound[by];
			shared_hist[0]=tempValue.x;
			shared_hist[1]=tempValue.y;
		}
		__syncthreads();
		int start=shared_hist[0];
		int end=shared_hist[1];
		//printf("bx, %d, by,%d, start, %d, end, %d\n", bx, by, start, end);
		for(int pos=start+resultID;pos<end;pos+=numThreadPerPartition)
		{
			pid=d_pidArray[pos];
			shared_hist[pid+offsetInBlock]++;
		}
		int sum=0;
		for(pid=0;pid<numPart;pid++)
		{	
			sum+=shared_hist[pid+offsetInBlock];
			d_hist[resultID+numThreadPerPartition*pid+offsetGlobal]=shared_hist[pid+offsetInBlock];
			//printf("d_hist: %d, %d; ", resultID+numThreadPerPartition*pid+offsetGlobal,d_hist[resultID+numThreadPerPartition*pid+offsetGlobal]); 
		}
		//printf("\n sum, %d\n", sum);
	}
	//having the prefix sum, compute the write location.
	__global__ void 
	writeHist_MB_kernel(int *d_pidArray, Record *d_iBound, int numPart, int *d_psSum, int* d_loc)
	{
		extern __shared__ int shared_hist[];//the first two is pStart, pEnd
		const int by = blockIdx.y;
		const int bx = blockIdx.x;
		const int tx = threadIdx.x;
		const int ty = threadIdx.y;	
		const int tid=tx+ty*blockDim.x;
		//const int bid=bx+by*gridDim.x;
		const int numThreadPerPartition=blockDim.x*gridDim.x*blockDim.y;
		const int resultID=bx*blockDim.x+tid; //!! not bid!!, the thread ID within the x blocks.
		int pid=0;
		int offsetInBlock=tid*numPart+2;
		//(numPart*numThreadPerPartition) is the number of entries for each partition. 
		int offsetGlobal=numPart*numThreadPerPartition*by;
		for(pid=0;pid<numPart;pid++)
		{
			shared_hist[pid+offsetInBlock]=d_psSum[resultID+numThreadPerPartition*pid+offsetGlobal];
		}
		if(tid==0)
		{
			Record tempValue=d_iBound[by];
			shared_hist[0]=tempValue.x;
			shared_hist[1]=tempValue.y;
		}
		__syncthreads();
		int start=shared_hist[0];
		int end=shared_hist[1];
		for(int pos=start+resultID;pos<end;pos+=numThreadPerPartition)
		{
			pid=d_pidArray[pos];
			d_loc[pos]=shared_hist[pid+offsetInBlock];
			//printf("d_loc: %d, %d; ", pos, d_loc[pos]);
			shared_hist[pid+offsetInBlock]++;
		}
		//printf("\n");
	}
#else //no coalesced with shared memory
	__global__ void
	countHist_kernel(int *d_pidArray, int delta, int rLen, int numPart, int *d_hist)
	{
		extern __shared__ int shared_hist[];
		const int by = blockIdx.y;
		const int bx = blockIdx.x;
		const int tx = threadIdx.x;
		const int ty = threadIdx.y;	
		const int tid=tx+ty*blockDim.x;
		const int bid=bx+by*gridDim.x;
		const int numThread=blockDim.x;
		const int resultID=(bid)*numThread+tid;

		int numBlock = gridDim.x;
		int totalTx = bx*numThread + tx;
		int threadSize = rLen/( numThread*numBlock );
		int start = totalTx*threadSize;
		int end = start + threadSize;

		int pid=0;
		int offset=tid*numPart;
		
		for(pid=0;pid<numPart;pid++)
		{
			shared_hist[pid+offset]=0;
		}
			

		for(int pos = start;pos < end;pos++ )
		{
			pid=d_pidArray[pos];
			shared_hist[pid+offset]++;
		}

		for(pid=0;pid<numPart;pid++)
		{
			d_hist[resultID+delta*pid]=shared_hist[pid+offset];
		}			
	}
	//having the prefix sum, compute the write location.
	__global__ void 
	writeHist_kernel(int *d_pidArray, int delta, int rLen, int numPart, int *d_psSum, int* d_loc)
	{
		extern __shared__ int shared_hist[];
		const int by = blockIdx.y;
		const int bx = blockIdx.x;
		const int tx = threadIdx.x;
		const int ty = threadIdx.y;	
		const int tid=tx+ty*blockDim.x;
		const int bid=bx+by*gridDim.x;
		const int numThread=blockDim.x;
		const int resultID=(bid)*numThread+tid;
		int pid=0;
		int offset=tid*numPart;

		int numBlock = gridDim.x;
		int totalTx = bx*numThread + tx;
		int threadSize = rLen/( numThread*numBlock );
		int start = totalTx*threadSize;
		int end = start + threadSize;

		for(pid=0;pid<numPart;pid++)
		{
			shared_hist[pid+offset]=d_psSum[resultID+delta*pid];
		}		
			
		for(int pos = start; pos < end; pos++ )
		{
			pid=d_pidArray[pos];
			d_loc[pos]=shared_hist[pid+offset];
			shared_hist[pid+offset]++;
		}
	}

	/*
	//
	//
	// multiple block version
	*/

	__global__ void 
	countHist_MB_kernel(int *d_pidArray, Record *d_iBound, int numPart, int *d_hist)
	{
		extern __shared__ int shared_hist[];//the first two is pStart, pEnd
		const int by = blockIdx.y;
		const int bx = blockIdx.x;
		const int tx = threadIdx.x;
		const int ty = threadIdx.y;	
		const int tid=tx+ty*blockDim.x;
		//const int bid=bx+by*gridDim.x;
		const int numThreadPerPartition=blockDim.x*gridDim.x*blockDim.y;
		const int resultID=bx*blockDim.x+tid; //!! not bid!!, the thread ID within the x blocks.
		int pid=0;
		int offsetInBlock=tid*numPart+2;
		//(numPart*numThreadPerPartition) is the number of entries for each partition. 
		int offsetGlobal=numPart*numThreadPerPartition*by;
		for(pid=0;pid<numPart;pid++)
			shared_hist[pid+offsetInBlock]=0;
		if(tid==0)
		{
			Record tempValue=d_iBound[by];
			shared_hist[0]=tempValue.x;
			shared_hist[1]=tempValue.y;
		}
		__syncthreads();
		int start=shared_hist[0];
		int end=shared_hist[1];

		int rLen = end - start;
		int totalTx = by*bx*blockDim.x + tx;
		int threadSize = rLen/( numThreadPerPartition );
		int startIdx = start + totalTx*threadSize;
		int endIdx = startIdx + threadSize;


		//printf("bx, %d, by,%d, start, %d, end, %d\n", bx, by, start, end);
		//for(int pos=start+resultID;pos<end;pos+=numThreadPerPartition)
		for(int pos=startIdx;pos<endIdx;pos++)
		{
			pid=d_pidArray[pos];
			shared_hist[pid+offsetInBlock]++;
		}
		int sum=0;
		for(pid=0;pid<numPart;pid++)
		{	
			sum+=shared_hist[pid+offsetInBlock];
			d_hist[resultID+numThreadPerPartition*pid+offsetGlobal]=shared_hist[pid+offsetInBlock];
			//printf("d_hist: %d, %d; ", resultID+numThreadPerPartition*pid+offsetGlobal,d_hist[resultID+numThreadPerPartition*pid+offsetGlobal]); 
		}
		//printf("\n sum, %d\n", sum);
	}
	//having the prefix sum, compute the write location.
	__global__ void 
	writeHist_MB_kernel(int *d_pidArray, Record *d_iBound, int numPart, int *d_psSum, int* d_loc)
	{
		extern __shared__ int shared_hist[];//the first two is pStart, pEnd
		const int by = blockIdx.y;
		const int bx = blockIdx.x;
		const int tx = threadIdx.x;
		const int ty = threadIdx.y;	
		const int tid=tx+ty*blockDim.x;
		//const int bid=bx+by*gridDim.x;
		const int numThreadPerPartition=blockDim.x*gridDim.x*blockDim.y;
		const int resultID=bx*blockDim.x+tid; //!! not bid!!, the thread ID within the x blocks.
		int pid=0;
		int offsetInBlock=tid*numPart+2;
		//(numPart*numThreadPerPartition) is the number of entries for each partition. 
	
		int offsetGlobal=numPart*numThreadPerPartition*by;
		for(pid=0;pid<numPart;pid++)
		{
			shared_hist[pid+offsetInBlock]=d_psSum[resultID+numThreadPerPartition*pid+offsetGlobal];
		}

		if(tid==0)
		{
			Record tempValue=d_iBound[by];
			shared_hist[0]=tempValue.x;
			shared_hist[1]=tempValue.y;
		}
		__syncthreads();

		int start=shared_hist[0];
		int end=shared_hist[1];

		int rLen = end - start;
		int totalTx = by*bx*blockDim.x + tx;
		int threadSize = rLen/( numThreadPerPartition );
		int startIdx = start + totalTx*threadSize;
		int endIdx = startIdx + threadSize;

		for(int pos=startIdx;pos<endIdx;pos++)
		{
			pid=d_pidArray[pos];
			d_loc[pos]=shared_hist[pid+offsetInBlock];
			//printf("d_loc: %d, %d; ", pos, d_loc[pos]);
			shared_hist[pid+offsetInBlock]++;
		}
		//printf("\n");
	}
#endif
#else//no shared memory with coalesced
	__global__ void 
	countHist_kernel(int *d_pidArray, int delta, int rLen, int numPart, int *d_hist)
	{
		//extern __shared__ int shared_hist[];
		const int by = blockIdx.y;
		const int bx = blockIdx.x;
		const int tx = threadIdx.x;
		const int ty = threadIdx.y;	
		const int tid=tx+ty*blockDim.x;
		const int bid=bx+by*gridDim.x;
		const int numThread=blockDim.x;
		const int resultID=(bid)*numThread+tid;
		int pid=0;
		int offset=tid*numPart;
		for(pid=0;pid<numPart;pid++)
			//shared_hist[pid+offset]=0;
			d_hist[resultID+delta*pid]=0;
		for(int pos=resultID;pos<rLen;pos+=delta)
		{
			pid=d_pidArray[pos];
			//shared_hist[pid+offset]++;
			d_hist[resultID+delta*pid]++;
		}
		//for(pid=0;pid<numPart;pid++)
		//	d_hist[resultID+delta*pid]=shared_hist[pid+offset];
	}
	//no shared memory
	__global__ void 
	writeHist_kernel(int *d_pidArray, int delta, int rLen, int numPart, int *d_psSum, int* d_loc)
	{
		//extern __shared__ int shared_hist[];
		const int by = blockIdx.y;
		const int bx = blockIdx.x;
		const int tx = threadIdx.x;
		const int ty = threadIdx.y;	
		const int tid=tx+ty*blockDim.x;
		const int bid=bx+by*gridDim.x;
		const int numThread=blockDim.x;
		const int resultID=(bid)*numThread+tid;
		int pid=0;
		int offset=tid*numPart;
		//for(pid=0;pid<numPart;pid++)
		//	shared_hist[pid+offset]=d_psSum[resultID+delta*pid];
		for(int pos=resultID;pos<rLen;pos+=delta)
		{
			pid=d_pidArray[pos];
			//d_loc[pos]=shared_hist[pid+offset];
			d_loc[pos]=d_psSum[resultID+delta*pid];
			//shared_hist[pid+offset]++;
			d_psSum[resultID+delta*pid]++;
		}
	}


	/*
	//
	//
	// multiple block version without shared memory
	*/

	__global__ void 
	countHist_MB_kernel(int *d_pidArray, Record *d_iBound, int numPart, int *d_hist)
	{
		extern __shared__ int shared_hist[];//the first two is pStart, pEnd
		const int by = blockIdx.y;
		const int bx = blockIdx.x;
		const int tx = threadIdx.x;
		const int ty = threadIdx.y;	
		const int tid=tx+ty*blockDim.x;
		//const int bid=bx+by*gridDim.x;
		const int numThreadPerPartition=blockDim.x*gridDim.x*blockDim.y;
		const int resultID=bx*blockDim.x+tid; //!! not bid!!, the thread ID within the x blocks.
		int pid=0;
		int offsetInBlock=tid*numPart+2;
		//(numPart*numThreadPerPartition) is the number of entries for each partition. 
		int offsetGlobal=numPart*numThreadPerPartition*by;
		for(pid=0;pid<numPart;pid++)
			d_hist[resultID+numThreadPerPartition*pid+offsetGlobal]=0;
		if(tid==0)
		{
			Record tempValue=d_iBound[by];
			shared_hist[0]=tempValue.x;
			shared_hist[1]=tempValue.y;
		}
		__syncthreads();
		int start=shared_hist[0];
		int end=shared_hist[1];
		//printf("bx, %d, by,%d, start, %d, end, %d\n", bx, by, start, end);
		for(int pos=start+resultID;pos<end;pos+=numThreadPerPartition)
		{
			pid=d_pidArray[pos];
			//shared_hist[pid+offsetInBlock]++;
			d_hist[resultID+numThreadPerPartition*pid+offsetGlobal]++;
		}
		int sum=0;
		//for(pid=0;pid<numPart;pid++)
		//{	
			//sum+=shared_hist[pid+offsetInBlock];
			//d_hist[resultID+numThreadPerPartition*pid+offsetGlobal]=shared_hist[pid+offsetInBlock];
			//printf("d_hist: %d, %d; ", resultID+numThreadPerPartition*pid+offsetGlobal,d_hist[resultID+numThreadPerPartition*pid+offsetGlobal]); 
		//}
		//printf("\n sum, %d\n", sum);
	}
	//having the prefix sum, compute the write location.
	__global__ void 
	writeHist_MB_kernel(int *d_pidArray, Record *d_iBound, int numPart, int *d_psSum, int* d_loc)
	{
		extern __shared__ int shared_hist[];//the first two is pStart, pEnd
		const int by = blockIdx.y;
		const int bx = blockIdx.x;
		const int tx = threadIdx.x;
		const int ty = threadIdx.y;	
		const int tid=tx+ty*blockDim.x;
		//const int bid=bx+by*gridDim.x;
		const int numThreadPerPartition=blockDim.x*gridDim.x*blockDim.y;
		const int resultID=bx*blockDim.x+tid; //!! not bid!!, the thread ID within the x blocks.
		int pid=0;
		int offsetInBlock=tid*numPart+2;
		//(numPart*numThreadPerPartition) is the number of entries for each partition. 
		int offsetGlobal=numPart*numThreadPerPartition*by;
		for(pid=0;pid<numPart;pid++)
		{
			//shared_hist[pid+offsetInBlock]=d_psSum[resultID+numThreadPerPartition*pid+offsetGlobal];
		}
		if(tid==0)
		{
			Record tempValue=d_iBound[by];
			shared_hist[0]=tempValue.x;
			shared_hist[1]=tempValue.y;
		}
		__syncthreads();
		int start=shared_hist[0];
		int end=shared_hist[1];
		for(int pos=start+resultID;pos<end;pos+=numThreadPerPartition)
		{
			pid=d_pidArray[pos];
			//d_loc[pos]=shared_hist[pid+offsetInBlock];
			d_loc[pos]=d_psSum[resultID+numThreadPerPartition*pid+offsetGlobal];
			//printf("d_loc: %d, %d; ", pos, d_loc[pos]);
			//shared_hist[pid+offsetInBlock]++;
			d_psSum[resultID+numThreadPerPartition*pid+offsetGlobal]++;
		}
		//printf("\n");
	}
#endif

//here we assume numPart<the number of threads.
//rLen is the number of tuples. 
__global__ void 
getBound_kernel(int *d_psSum, int interval, int rLen, int numPart, Record* d_bound)
{
	extern __shared__ int shared_hist[];
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	if(resultID<numPart)
	{
		int start=0;
		if(resultID==0)
			start=0;
		else
			start=d_psSum[interval*resultID];
		int end=0;
		if((resultID+1)==numPart)
			end=rLen;
		else
			end=d_psSum[interval*(resultID+1)];			
		d_bound[resultID].x=start;
		d_bound[resultID].y=end;
	}	
}


//here we assume numPart<the number of threads.
//rLen is the number of tuples. 
__global__ void 
getBound_MB_kernel(int *d_psSum, int interval, int rLen, int numPart, Record* d_bound)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	if(resultID<numPart)
	{
		int start=0;
		if(resultID==0)
			start=0;
		else
			start=d_psSum[interval*resultID];
		int end=0;
		if((resultID+1)==numPart)
			end=rLen;
		else
			end=d_psSum[interval*(resultID+1)];			
		d_bound[resultID].x=start;
		d_bound[resultID].y=end;
	}	
}

#endif

