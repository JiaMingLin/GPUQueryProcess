
#ifndef _NINLJ_CHECK_H_
#define _NINLJ_CHECK_H_


#ifndef SHARED_MEM

	__global__ void
	gpuNLJ_kernel(int* d_temp, Record* d_shared_s, Record *d_R, Record *d_S, int sStart, int rLen, int sLen, int *d_n) 
	{
		//__shared__ Record shared_s[NLJ_S_BLOCK_SIZE];
		Record* shared_s;
		shared_s = d_shared_s + blockIdx.y*blockDim.x*NLJ_S_BLOCK_SIZE+blockIdx.x*NLJ_S_BLOCK_SIZE;
		int tx = threadIdx.x;
		int ty = threadIdx.y;
		int bx = blockIdx.x;
		int by = blockIdx.y;
		int bid=bx+by*gridDim.x;
		int tid=tx+ty*blockDim.x;
		int resultID=(bid)*NLJ_NUM_THREADS_PER_BLOCK+tid;
		int j=0;
		int i=0;
		int numResult=0;
		Record rTmp;
		sStart+=bid*NLJ_S_BLOCK_SIZE;
		int curPosInShared=0;
		for(i=0;i<NLJ_NUM_TUPLE_PER_THREAD;i++)
		{
			curPosInShared=tid+NLJ_NUM_THREADS_PER_BLOCK*i;
			if((curPosInShared+sStart)<sLen)
				shared_s[curPosInShared]=d_S[(curPosInShared+sStart)];
			else
				shared_s[curPosInShared].y=-1;
		}
		__syncthreads();
		for(i = 0; (i+tid) < rLen; i=i+NLJ_R_BLOCK_SIZE)
		{
			//printf("%d, ", i);
			rTmp=d_R[i+tid];
			d_temp[i] = d_R[i].x;
			for(j=0;j<NLJ_S_BLOCK_SIZE;j++)
			{
				if(PRED_EQUAL(rTmp.y, shared_s[j].y))
				{
					numResult++;
				}
			}
		}
		__syncthreads();
		d_n[resultID]=numResult;
	}	
#endif

#ifndef COALESCED
	__global__ void
	gpuNLJ_noCoalesced_kernel(Record *d_R, Record *d_S, int sStart, int rLen, int sLen, int *d_n) 
	{
		__shared__ Record shared_s[NLJ_S_BLOCK_SIZE];
		int tx = threadIdx.x;
		int ty = threadIdx.y;
		int bx = blockIdx.x;
		int by = blockIdx.y;
		int bid=bx+by*gridDim.x;
		int tid=tx+ty*blockDim.x;
		int resultID=(bid)*NLJ_NUM_THREADS_PER_BLOCK+tid;
		int j=0;
		int i=0;
		int numResult=0;
		Record rTmp;
		sStart+=bid*NLJ_S_BLOCK_SIZE;
		int curPosInShared=0;
		for(i=0;i<NLJ_NUM_TUPLE_PER_THREAD;i++)
		{
			curPosInShared=tid+NLJ_NUM_THREADS_PER_BLOCK*i;
			if((curPosInShared+sStart)<sLen)
				shared_s[curPosInShared]=d_S[(curPosInShared+sStart)];
			else
				shared_s[curPosInShared].y=-1;
		}
		__syncthreads();

		for(i = tid; (i) < rLen; i=i+NLJ_R_BLOCK_SIZE)
		{
			rTmp=d_R[i];
			for(j=0;j<NLJ_S_BLOCK_SIZE;j++)
			{
				if(PRED_EQUAL(rTmp.y, shared_s[j].y))
				{
					numResult++;
				}
			}
		}
		__syncthreads();
		d_n[resultID]=numResult;
	}
#endif

//best with shared memory , with coaesced
__global__ void
gpuNLJ_kernel(int* d_temp, Record *d_R, Record *d_S, int sStart, int rLen, int sLen, int *d_n) 
{
	__shared__ Record shared_s[NLJ_S_BLOCK_SIZE];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bid=bx+by*gridDim.x;
	int tid=tx+ty*blockDim.x;
	int resultID=(bid)*NLJ_NUM_THREADS_PER_BLOCK+tid;
	int j=0;
	int i=0;
	int numResult=0;
	Record rTmp;
	sStart+=bid*NLJ_S_BLOCK_SIZE;
	int curPosInShared=0;
	for(i=0;i<NLJ_NUM_TUPLE_PER_THREAD;i++)
	{
		curPosInShared=tid+NLJ_NUM_THREADS_PER_BLOCK*i;
		if((curPosInShared+sStart)<sLen)
			shared_s[curPosInShared]=d_S[(curPosInShared+sStart)];
		else
			shared_s[curPosInShared].y=-1;
	}
	__syncthreads();

	for(i = tid; (i) < rLen; i=i+NLJ_R_BLOCK_SIZE)
	{
		rTmp=d_R[i];
		d_temp[i] = d_R[i].x;
		for(j=0;j<NLJ_S_BLOCK_SIZE;j++)
		{
			if(PRED_EQUAL(rTmp.y, shared_s[j].y))
			{
				numResult++;
			}
		}
	}
	__syncthreads();
	d_n[resultID]=numResult;
}


#ifndef SHARED_MEM
	__global__ void
	write(Record* d_shared_s, Record *d_R, Record *d_S,  int sStart, int rLen, int sLen, int *d_sum, Record *output)
	{
		//__shared__ Record shared_s[NLJ_S_BLOCK_SIZE];
		Record* shared_s;
		shared_s = d_shared_s + blockIdx.y*blockDim.x*NLJ_S_BLOCK_SIZE+blockIdx.x*NLJ_S_BLOCK_SIZE;;
		int tx = threadIdx.x;
		int ty = threadIdx.y;
		int bx = blockIdx.x;
		int by = blockIdx.y;
		int bid=bx+by*gridDim.x;
		int tid=tx+ty*blockDim.x;
		int resultID=(bid)*NLJ_NUM_THREADS_PER_BLOCK+tid;
		int j=0;
		int i=0;
		Record rTmp;
		
		int base=d_sum[resultID];
		//if(d_sum[bstartSum]!=d_sum[bendSum])
		//for(int sg=0;sg<NLJ_NUM_GRID_S;sg++)	
		{
			sStart+=bid*NLJ_S_BLOCK_SIZE;
			int curPosInShared=0;
			for(i=0;i<NLJ_NUM_TUPLE_PER_THREAD;i++)
			{
				curPosInShared=tid+NLJ_NUM_THREADS_PER_BLOCK*i;
				if((curPosInShared+sStart)<sLen)
					shared_s[curPosInShared]=d_S[(curPosInShared+sStart)];
				else
					shared_s[curPosInShared].y=-1;
			}		
			__syncthreads();
			for(i = 0; (i+tid) < rLen; i=i+NLJ_R_BLOCK_SIZE)
			{
				//printf("%d, ", i);
				rTmp=d_R[i+tid];
				for(j=0;j<NLJ_S_BLOCK_SIZE;j++)
				{
					if(PRED_EQUAL(rTmp.y, shared_s[j].y))
					{
						output[base].x=rTmp.x;
						output[base].y=shared_s[j].x;
						base++;
					}
				}
			}
			__syncthreads();
		}
	}
#endif

#ifndef COALESCED
__global__ void
write_noCoalesced(Record *d_R, Record *d_S,  int sStart, int rLen, int sLen, int *d_sum, Record *output)
{
	__shared__ Record shared_s[NLJ_S_BLOCK_SIZE];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bid=bx+by*gridDim.x;
	int tid=tx+ty*blockDim.x;
	int resultID=(bid)*NLJ_NUM_THREADS_PER_BLOCK+tid;
	int j=0;
	int i=0;
	Record rTmp;
	
	int base=d_sum[resultID];
	//if(d_sum[bstartSum]!=d_sum[bendSum])
	//for(int sg=0;sg<NLJ_NUM_GRID_S;sg++)	
	{
		sStart+=bid*NLJ_S_BLOCK_SIZE;
		int curPosInShared=0;
		for(i=0;i<NLJ_NUM_TUPLE_PER_THREAD;i++)
		{
			curPosInShared=tid+NLJ_NUM_THREADS_PER_BLOCK*i;
			if((curPosInShared+sStart)<sLen)
				shared_s[curPosInShared]=d_S[(curPosInShared+sStart)];
			else
				shared_s[curPosInShared].y=-1;
		}		
		__syncthreads();

		int numThread = blockDim.x;
		int len = rLen/numThread;
		int start = len*threadIdx.x;
		int end = start + len;
		if( threadIdx.x == numThread - 1 )
		{
			end = rLen;
		}
		
		//for(i = tid; i < rLen; i=i+NLJ_R_BLOCK_SIZE)
		for( i = start; i < end; i++ )
		{
			rTmp=d_R[i];
			for(j=0;j<NLJ_S_BLOCK_SIZE;j++)
			{
				if(PRED_EQUAL(rTmp.y, shared_s[j].y))
				{
					output[base].x=rTmp.x;
					output[base].y=shared_s[j].x;
					base++;
				}
			}
		}
		__syncthreads();
	}
}
#endif

__global__ void
write(Record *d_R, Record *d_S,  int sStart, int rLen, int sLen, int *d_sum, Record *output)
{
	__shared__ Record shared_s[NLJ_S_BLOCK_SIZE];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bid=bx+by*gridDim.x;
	int tid=tx+ty*blockDim.x;
	int resultID=(bid)*NLJ_NUM_THREADS_PER_BLOCK+tid;
	int j=0;
	int i=0;
	Record rTmp;
	
	int base=d_sum[resultID];
	//if(d_sum[bstartSum]!=d_sum[bendSum])
	//for(int sg=0;sg<NLJ_NUM_GRID_S;sg++)	
	{
		sStart+=bid*NLJ_S_BLOCK_SIZE;
		int curPosInShared=0;
		for(i=0;i<NLJ_NUM_TUPLE_PER_THREAD;i++)
		{
			curPosInShared=tid+NLJ_NUM_THREADS_PER_BLOCK*i;
			if((curPosInShared+sStart)<sLen)
				shared_s[curPosInShared]=d_S[(curPosInShared+sStart)];
			else
				shared_s[curPosInShared].y=-1;
		}		
		__syncthreads();
		
		for(i = tid; i < rLen; i=i+NLJ_R_BLOCK_SIZE)
		{
			rTmp=d_R[i];
			for(j=0;j<NLJ_S_BLOCK_SIZE;j++)
			{
				if(PRED_EQUAL(rTmp.y, shared_s[j].y))
				{
					output[base].x=rTmp.x;
					output[base].y=shared_s[j].x;
					base++;
				}
			}
		}
		__syncthreads();
	}
}

__global__ void
matchCount_kernel(Record *R, Record *S, int sStart, int rLen, int sLen, int *d_n) 
{
	__shared__ Record match_ss[NLJ_S_BLOCK_SIZE];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bid=bx+by*gridDim.x;
	int tid=tx+ty*blockDim.x;
	int resultID=(bid)*NLJ_NUM_THREADS_PER_BLOCK+tid;
	int j=0;
	int i=0;
	int numResult=0;
	Record rTmp;
	Record sTmp;
	sStart+=bid*NLJ_S_BLOCK_SIZE;
	if(sStart+tid<sLen)
	{
		match_ss[tid]=S[sStart+tid];
	}
	else
	{
		match_ss[tid].y=-1;
	}
	__syncthreads();
	for(i = 0; (i+tid) < rLen; i=i+NLJ_R_BLOCK_SIZE)
	{
		//printf("%d, ", i);
		rTmp=R[i+tid];
		for(j=0;j<NLJ_S_BLOCK_SIZE;j++)
		{
			sTmp=match_ss[j];
			if(sTmp.y!=-1)
			{
				if((sTmp.x<=rTmp.x && rTmp.x<=sTmp.y) || (rTmp.x<=sTmp.x && sTmp.x<=rTmp.y))
				{
					numResult++;
					//printf("S%d, %d, %d, R%d,%d, %d\n", sStart+j, sTmp.x, sTmp.y, i+tid, rTmp.x, rTmp.y);
				}
			}
		}
	}
	__syncthreads();
	d_n[resultID]=numResult;
}



__global__ void
matchWrite_kernel(Record *R, Record *S,  int sStart, int rLen, int sLen, int *d_sum, Record *output)
{
	__shared__ Record match_ss[NLJ_S_BLOCK_SIZE];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bid=bx+by*gridDim.x;
	int tid=tx+ty*blockDim.x;
	int resultID=(bid)*NLJ_NUM_THREADS_PER_BLOCK+tid;
	int j=0;
	int i=0;
	Record rTmp;
	Record sTmp;
	
	int base=d_sum[resultID];
	//if(d_sum[bstartSum]!=d_sum[bendSum])
	//for(int sg=0;sg<NLJ_NUM_GRID_S;sg++)	
	{
		sStart+=bid*NLJ_S_BLOCK_SIZE;
		if(sStart+tid<sLen)
		{
			match_ss[tid]=S[sStart+tid];
		}
		else
		{
			match_ss[tid].y=-1;
		}		
		__syncthreads();
		for(i = 0; (i+tid) < rLen; i=i+NLJ_R_BLOCK_SIZE)
		{
			rTmp=R[i+tid];
			for(j=0;j<NLJ_S_BLOCK_SIZE;j++)
			{
				sTmp=match_ss[j];
				if(sTmp.y!=-1)
				{
					if((sTmp.x<=rTmp.x && rTmp.x<=sTmp.y) || (rTmp.x<=sTmp.x && sTmp.x<=rTmp.y))
					{
						output[base].x=i+tid;
						output[base].y=sStart+j;
						base++;
					}
				}
			}
		}
		__syncthreads();
	}
}


//use constant memory
__global__ void
gpuNLJ_Constant_kernel(Record *d_R, Record *d_S, int sStart, int rLen, int sLen, int *d_n) 
{
	__shared__ Record shared_s[NLJ_S_BLOCK_SIZE];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bid=bx+by*gridDim.x;
	int tid=tx+ty*blockDim.x;
	int resultID=(bid)*NLJ_NUM_THREADS_PER_BLOCK+tid;
	int j=0;
	int i=0;
	int numResult=0;
	Record rTmp;
	sStart+=bid*NLJ_S_BLOCK_SIZE;
	if(sStart+tid<sLen)
	{
		shared_s[tid]=d_S[sStart+tid];
	}
	else
	{
		shared_s[tid].y=-1;
	}
	__syncthreads();
	for(i = 0; (i+tid) < rLen; i=i+NLJ_R_BLOCK_SIZE)
	{
		//printf("%d, ", i);
		rTmp=d_R[i+tid];
		for(j=0;j<NLJ_S_BLOCK_SIZE;j++)
		{
			if(PRED_EQUAL(rTmp.y, shared_s[j].y))
			{
				numResult++;
			}
		}
	}
	__syncthreads();
	d_n[resultID]=numResult;
}



__global__ void
write_Constant_kernel(Record *d_R, Record *d_S,  int sStart, int rLen, int sLen, int *d_sum, Record *output)
{
	__shared__ Record shared_s[NLJ_S_BLOCK_SIZE];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bid=bx+by*gridDim.x;
	int tid=tx+ty*blockDim.x;
	int resultID=(bid)*NLJ_NUM_THREADS_PER_BLOCK+tid;
	int j=0;
	int i=0;
	Record rTmp;
	
	int base=d_sum[resultID];
	//if(d_sum[bstartSum]!=d_sum[bendSum])
	//for(int sg=0;sg<NLJ_NUM_GRID_S;sg++)	
	{
		sStart+=bid*NLJ_S_BLOCK_SIZE;
		if(sStart+tid<sLen)
		{
			shared_s[tid]=d_S[sStart+tid];
		}
		else
		{
			shared_s[tid].y=-1;
		}		
		__syncthreads();
		for(i = 0; (i+tid) < rLen; i=i+NLJ_R_BLOCK_SIZE)
		{
			//printf("%d, ", i);
			rTmp=d_R[i+tid];
			for(j=0;j<NLJ_S_BLOCK_SIZE;j++)
			{
				if(PRED_EQUAL(rTmp.y, shared_s[j].y))
				{
					output[base].x=rTmp.x;
					output[base].y=shared_s[j].x;
					base++;
				}
			}
		}
		__syncthreads();
	}
}

#endif
