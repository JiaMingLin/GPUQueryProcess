
#ifndef _RadixHJ_KERNEL_CU_
#define _RadixHJ_KERNEL_CU_

#include "Header.cu"


////////////////////////////////////////////////////////////////////////////////////////////Build

//	Histo: scan R and get d_PidHisto[pn] of each thread
//		p=hash(r); histo[p]++; sync; d_PidHisto[snakewise p]=histo;
__global__ void Histo(int* d_HistoMat, Record* d_R, const int nR, const int pn, const int shift)
{
	const int gridLen = gridDim.x * blockDim.x;
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int pid;
	extern __shared__ int s_histo[];
	for(int pi = 0; pi < pn; ++pi)
	{
		s_histo[threadIdx.x * pn + pi] = 0;
	}
	while(offset < nR)
	{
		pid = (HASH(d_R[offset].y) >> shift) & (pn - 1);	
		++s_histo[threadIdx.x * pn + pid];
		offset = offset + gridLen;
	}
	__syncthreads();
	offset = blockIdx.x * blockDim.x + threadIdx.x;
	for(int pi = 0; pi < pn; ++pi)
	{
		d_HistoMat[pi * gridLen + offset] = s_histo[threadIdx.x * pn + pi];
	}
}

#ifndef SHARED_MEM
__global__ void Histo_noShared(int* d_histo, int sharedSize, int* d_HistoMat, Record* d_R, const int nR, const int pn, const int shift)
{
	const int gridLen = gridDim.x * blockDim.x;
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int pid;
	//extern __shared__ int s_histo[];
	int* s_histo;
	s_histo = d_histo + blockIdx.x*(sharedSize/sizeof(int));

	for(int pi = 0; pi < pn; ++pi)
	{
		s_histo[threadIdx.x * pn + pi] = 0;
	}
	while(offset < nR)
	{
		pid = (HASH(d_R[offset].y) >> shift) & (pn - 1);	
		++s_histo[threadIdx.x * pn + pid];
		offset = offset + gridLen;
	}
	__syncthreads();
	offset = blockIdx.x * blockDim.x + threadIdx.x;
	for(int pi = 0; pi < pn; ++pi)
	{
		d_HistoMat[pi * gridLen + offset] = s_histo[threadIdx.x * pn + pi];
	}
}
#endif

__global__ void Reorder(Record* d_R1, int* d_PBound, Record* d_R, int* d_WriteLoc, const int nR, const int pn, const int shift)
{
	int gridLen = gridDim.x * blockDim.x;
	int pid;
	Record recR;
	extern __shared__ int s_writeLoc[];
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	for(int pi = 0; pi < pn; ++pi)
	{
		s_writeLoc[threadIdx.x * pn + pi] = d_WriteLoc[pi * gridLen + offset];
	}
    if(offset == 0)//this thread's corresponding HistMatPre contains d_partbound2
	{
		for(int pi = 0; pi < pn; ++pi)
		{
			d_PBound[pi] = s_writeLoc[threadIdx.x * pn + pi];
		}
		d_PBound[pn] = nR;
	}

	//v0: single pass
	while(offset < nR)
	{
		recR = d_R[offset];
		pid = threadIdx.x * pn + ((HASH(recR.y) >> shift) & (pn - 1));
		//pid = threadIdx.x * pn + d_Pid[offset];
		d_R1[s_writeLoc[pid]] = recR;
		++s_writeLoc[pid];
		offset = offset + gridLen;
	}

	////v1: multipass on s_writeLoc: slow. when nPass==1, it's as fast as no-multipass
	//const int nPass = 2;
	//const int nSubR = nR / nPass;
	//__shared__ int subStart, subEnd;
	//int loc;
	//for(int pass = 0; pass < nPass; ++pass)
	//{
	//	__syncthreads();
	//	subStart = pass * nSubR;
	//	subEnd = (pass + 1) * nSubR;

	//	offset = blockIdx.x * blockDim.x + threadIdx.x;
	//	while(offset < nR)
	//	{
	//		recR = d_R[offset];
	//		pid = ((HASH(recR.y) >> shift) & (pn - 1));
	//		loc = threadIdx.x * pn + pid;
	//		if(	(s_writeLoc[loc] >= subStart) && (s_writeLoc[loc] < subEnd) )
	//		{
	//			d_R1[s_writeLoc[loc]] = recR;
	//			++s_writeLoc[loc];
	//		}
	//		offset = offset + gridLen;
	//	}
	//}

	////v2: multipass on pid: samely slow
	//const int nPass = 1;
	//const int nSubPart = pn / nPass;
	//__shared__ int subStart, subEnd;
	//for(int pass = 0; pass < nPass; ++pass)
	//{
	//	__syncthreads();
	//	subStart = pass * nSubPart;
	//	subEnd = (pass + 1) * nSubPart;

	//	offset = blockIdx.x * blockDim.x + threadIdx.x;
	//	while(offset < nR)
	//	{
	//		recR = d_R[offset];
	//		pid = ((HASH(recR.y) >> shift) & (pn - 1));
	//		if(	(pid >= subStart) && (pid < subEnd) )
	//		{
	//			d_R1[s_writeLoc[threadIdx.x * pn + pid]] = recR;
	//			++s_writeLoc[threadIdx.x * pn + pid];
	//		}
	//		offset = offset + gridLen;
	//	}
	//}
}

#ifndef SHARED_MEM
__global__ void Reorder_noShared(int* d_writeLoc, int sharedSize, Record* d_R1, int* d_PBound, Record* d_R, int* d_WriteLoc, const int nR, const int pn, const int shift)
{
	int gridLen = gridDim.x * blockDim.x;
	int pid;
	Record recR;
	//extern __shared__ int s_writeLoc[];
	int* s_writeLoc;
	s_writeLoc = d_writeLoc + blockIdx.x*(sharedSize/sizeof(int));

	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	for(int pi = 0; pi < pn; ++pi)
	{
		s_writeLoc[threadIdx.x * pn + pi] = d_WriteLoc[pi * gridLen + offset];
	}
    if(offset == 0)//this thread's corresponding HistMatPre contains d_partbound2
	{
		for(int pi = 0; pi < pn; ++pi)
		{
			d_PBound[pi] = s_writeLoc[threadIdx.x * pn + pi];
		}
		d_PBound[pn] = nR;
	}

	//v0: single pass
	while(offset < nR)
	{
		recR = d_R[offset];
		pid = threadIdx.x * pn + ((HASH(recR.y) >> shift) & (pn - 1));
		//pid = threadIdx.x * pn + d_Pid[offset];
		d_R1[s_writeLoc[pid]] = recR;
		++s_writeLoc[pid];
		offset = offset + gridLen;
	}
}
#endif

// histo all parents in 1 kernel. Parent i corresponds to HistM[i*Bp]
//	for each parent, scan R and get d_PID and d_PidHisto of each thread
//		p=d_PID[r]=hash(r); histo[p]++; sync; d_PidHisto[snakewise p]=histo;
__global__ void Histo3(int* d_HistoMat, Record* d_R, int* d_PBound, const int nParent, const int pn, const int shift)
{
	const int bp = gridDim.x / nParent;			//num blocks per part
	const int partIdx = blockIdx.x / bp;		//part#
	const int partStart = d_PBound[partIdx];	//start rec idx
	const int partEnd =  d_PBound[partIdx + 1];	//end rec idx
	const int bx = blockIdx.x;					//blockIdx
	const int bx0 = partIdx * bp ;				//start block idx
	const int tn = blockDim.x;					//num threads in this block
	const int tx = threadIdx.x;

	extern __shared__ int s_histo[];
	//reset histo of this block
	for(int pi = 0; pi < pn; ++pi)
		s_histo[tx * pn + pi] = 0;

	//cumu
	int offset = partStart + (bx - bx0) * tn + tx;	//recIdx
	int pid;
	while(offset < partEnd)
	{
		pid = (HASH(d_R[offset].y) >> shift) & (pn - 1);	
		++s_histo[tx * pn + pid];
		offset += bp * tn;
	}
	
	//write
	offset = bx0 * tn * pn + (bx - bx0) * tn + tx;
	for(int pi = 0; pi < pn; ++pi)
	{
		d_HistoMat[pi * (bp * tn) + offset] = s_histo[tx * pn + pi];
	}
}

#ifndef SHARED_MEM
__global__ void Histo3_noShared(int* d_histo, int sharedSize, int* d_HistoMat, Record* d_R, int* d_PBound, const int nParent, const int pn, const int shift)
{
	const int bp = gridDim.x / nParent;			//num blocks per part
	const int partIdx = blockIdx.x / bp;		//part#
	const int partStart = d_PBound[partIdx];	//start rec idx
	const int partEnd =  d_PBound[partIdx + 1];	//end rec idx
	const int bx = blockIdx.x;					//blockIdx
	const int bx0 = partIdx * bp ;				//start block idx
	const int tn = blockDim.x;					//num threads in this block
	const int tx = threadIdx.x;

	//extern __shared__ int s_histo[];
	int* s_histo;
	s_histo = d_histo + blockIdx.x*(sharedSize/sizeof(int));

	//reset histo of this block
	for(int pi = 0; pi < pn; ++pi)
		s_histo[tx * pn + pi] = 0;

	//cumu
	int offset = partStart + (bx - bx0) * tn + tx;	//recIdx
	int pid;
	while(offset < partEnd)
	{
		pid = (HASH(d_R[offset].y) >> shift) & (pn - 1);	
		++s_histo[tx * pn + pid];
		offset += bp * tn;
	}
	
	//write
	offset = bx0 * tn * pn + (bx - bx0) * tn + tx;
	for(int pi = 0; pi < pn; ++pi)
	{
		d_HistoMat[pi * (bp * tn) + offset] = s_histo[tx * pn + pi];
	}
}

#endif

//Permute all parents in 1 kernel. Parent i corresponds to HistM[i*Bp]
//	for each parent, scan R and get d_PID and d_PidHisto of each thread
//		p=d_PID[r]=hash(r); histo[p]++; sync; d_PidHisto[snakewise p]=histo;

//write d_R to d_R1 according to d_Loc; write d_RDir to d_RDir1 using d_Loc
__global__ void Reorder3(Record* d_R1, int* d_RDir1, Record* d_R, int* d_RDir, 
						 int* d_Loc, const int nParent, const int pn, const int shift)
{
	const int bp = gridDim.x / nParent;			//num blocks per part
	const int partIdx = blockIdx.x / bp;		//part#
	const int partStart = d_RDir[partIdx];	//start rec idx
	const int partEnd =  d_RDir[partIdx + 1];	//end rec idx
	const int bx = blockIdx.x;					//blockIdx
	const int bx0 = partIdx * bp ;				//start block idx
	const int tn = blockDim.x;					//num threads in this block
	const int tx = threadIdx.x;
	
	extern __shared__ int s_writeLoc[];
	int offset = bx0 * tn * pn + (bx - bx0) * tn + tx;
	for(int pi = 0; pi < pn; ++pi)
		s_writeLoc[tx * pn + pi] = partStart + d_Loc[pi * (bp * tn) + offset];
	
    if((bx == bx0) && (tx == 0))//this writeLoc is also the child partition's start
	{
		for(int pi = 0; pi < pn; ++pi)
		{
			d_RDir1[partIdx * pn + pi] = s_writeLoc[tx * pn + pi];
		}
		d_RDir1[partIdx * pn + pn] = partEnd;
	}

	//scatter
	offset = partStart + (bx - bx0) * tn + tx;	//recIdx
	int pid;
	Record rec;
	while(offset < partEnd)
	{		
		rec = d_R[offset];
		pid = ((HASH(rec.y) >> shift) & (pn - 1));
		d_R1[s_writeLoc[tx * pn + pid]] = rec;
		++s_writeLoc[tx * pn + pid];
		offset += bp * tn;
	}
}

#ifndef SHARED_MEM
__global__ void Reorder3_noShared(int* d_writeLoc, int sharedSize, Record* d_R1, int* d_RDir1, Record* d_R, int* d_RDir, 
						 int* d_Loc, const int nParent, const int pn, const int shift)
{
	const int bp = gridDim.x / nParent;			//num blocks per part
	const int partIdx = blockIdx.x / bp;		//part#
	const int partStart = d_RDir[partIdx];	//start rec idx
	const int partEnd =  d_RDir[partIdx + 1];	//end rec idx
	const int bx = blockIdx.x;					//blockIdx
	const int bx0 = partIdx * bp ;				//start block idx
	const int tn = blockDim.x;					//num threads in this block
	const int tx = threadIdx.x;
	
	//extern __shared__ int s_writeLoc[];
	int* s_writeLoc;
	s_writeLoc = d_writeLoc + blockIdx.x*( sharedSize/sizeof(int) );

	int offset = bx0 * tn * pn + (bx - bx0) * tn + tx;
	for(int pi = 0; pi < pn; ++pi)
		s_writeLoc[tx * pn + pi] = partStart + d_Loc[pi * (bp * tn) + offset];
	
    if((bx == bx0) && (tx == 0))//this writeLoc is also the child partition's start
	{
		for(int pi = 0; pi < pn; ++pi)
		{
			d_RDir1[partIdx * pn + pi] = s_writeLoc[tx * pn + pi];
		}
		d_RDir1[partIdx * pn + pn] = partEnd;
	}

	//scatter
	offset = partStart + (bx - bx0) * tn + tx;	//recIdx
	int pid;
	Record rec;
	while(offset < partEnd)
	{		
		rec = d_R[offset];
		pid = ((HASH(rec.y) >> shift) & (pn - 1));
		d_R1[s_writeLoc[tx * pn + pid]] = rec;
		++s_writeLoc[tx * pn + pid];
		offset += bp * tn;
	}
}
#endif

//////////////////////////////////////////////////////////////////////////////////Sort

__device__ inline void swapRec(Record & a, Record & b)
{
	// Alternative swap doesn't use a temporary register:
	 //a ^= b;
	 //b ^= a;
	 //a ^= b;
	
    Record tmp = a;
    a = b;
    b = tmp;
}

__global__ void ProbePreSort(Record* d_R, int* d_RDir, int nR, Record* d_S, int* d_SDir, int nS, int PnG)
{
	const int partId = blockIdx.y * gridDim.x + blockIdx.x;
	if(partId >= PnG)
		return;

	const int r1 = d_RDir[partId];
	const int r2 = d_RDir[partId + 1];
	const int s1 = d_SDir[partId];
	const int s2 = d_SDir[partId + 1];

	if((r2 - r1 > _maxPartLen) && (s2 - s1 > _maxPartLen))	//overflow shared memory
		return;

	const bool bLoadR = ((r2 - r1) <= (s2 - s1));	//load r or s into SM
	extern __shared__ Record shared[];
    const int tid = threadIdx.x;
	
	shared[tid].y = INT_MAX;
	if(bLoadR)
	{
		if(tid < r2 - r1)
			shared[tid] = d_R[r1 + tid];
	}
	else
	{
		if(tid < s2 - s1) 
			shared[tid] = d_S[s1 + tid];
	}

    __syncthreads();

    // Parallel bitonic sort.

	for (int k = 2; k <= blockDim.x; k *= 2)
	{
		// Bitonic merge:
		for (int j = k / 2; j>0; j /= 2)
		{
			int ixj = tid ^ j;
 			if (ixj > tid)	//should carry the INTMAX together, o/w will err!:  if ((ixj > tid) && (tid < partLen))
			{
				if ((tid & k) == 0)
				{
					if (shared[tid].y > shared[ixj].y)
					{
						swapRec(shared[tid], shared[ixj]);
					}
				}
				else
				{
					if (shared[tid].y < shared[ixj].y)
					{
						swapRec(shared[tid], shared[ixj]);
					}
				}
			}
            
			__syncthreads();
		}
	}

	// Write result.
	if(bLoadR)
	{
		if(tid < r2 - r1)
			d_R[r1 + tid] = shared[tid];
	}
	else
	{
		if(tid < s2 - s1) 
			d_S[s1 + tid] = shared[tid];
	}
}

/////////////////////////////////////////////////////////////////////////////Probe
/* error
//bisearch 1st hitting key in keys[len]
__device__ 
int FirstHit(Record keys[], int len, int& key)
{
	int min = 0;
	int max = len;
	int mid;
	int cut;
	while(max - min > 1) {
		mid = (min + max) / 2;
		cut = keys[mid].y;

		if(key > cut)
			min = mid;
		else
			max = mid;
	}

	if(keys[min].y == key)
		return min;

	return -1; //return max;

}
*/

//bisearch 1st hitting key in keys[len]; return -1 if not found
__device__ 
int FirstHit(Record* A, int N, int& val)
{
	//v1: 300ms
	int low = 0;
	int high = N - 1;
	int p = low + (high - low) / 2;	//Initial probe position
    while(low <= high)
	{
		if(A[p].y > val)
			high = p -1;
		else if(A[p].y < val)
			low = p + 1;
		else
			return p;
		p = low + (high - low) / 2;
	}
	return -1;
	
	////v2: 340ms
	//for(int i = 0; i < N; ++i)
	//	if(A[i].y == val)
	//		return i;
	//return -1;
}


//find cnt for each threads
__global__ void Probe_Cnt(int* d_ThreadCnts, int* d_skewPid, Record* d_R, const int* d_PBoundR, Record* d_S, 
							const int* d_PBoundS, const int PnG)
{
	extern __shared__ Record s_Table[];
	int count = 0;	//important
	int px = blockIdx.x;	//part idx
	Record tmpRec;
	int r1, r2, s1, s2;
	int offset/*, firstHit*/;
	while(px < PnG)
	{
		r1 = d_PBoundR[px];
		r2 = d_PBoundR[px + 1];
		s1 = d_PBoundS[px];
		s2 = d_PBoundS[px + 1];
		if((r2 - r1 > _maxPartLen) && (s2 - s1 > _maxPartLen))
		{
			d_skewPid[px] = 1;
			__syncthreads();	//avoid next s_Table[] be dirtied
			px = px + gridDim.x;	//skip, leave to later
		}
		else if((r2 - r1) <= (s2 - s1))	//read R, scan S
		{
			offset = r1 + threadIdx.x;
			for(int i = 0; offset < r2; ++i)
			{
				s_Table[threadIdx.x + i * blockDim.x] = d_R[offset];
				offset = offset + blockDim.x;
			}
			__syncthreads();
			//read S
			offset = s1 + threadIdx.x;
			for(int i = 0; offset < s2; ++i)
			{
				tmpRec = d_S[offset];
#if _bSortPart
				firstHit = FirstHit(s_Table, r2 - r1, tmpRec.y);
				if(firstHit >= 0)
				{
					do{
						++count;
						++firstHit;
					}while(s_Table[firstHit].y == tmpRec.y);
				}
#else
				for(int j = 0; j < (r2 - r1); ++j)
				{
					if(s_Table[j].y == tmpRec.y)
					{
						++count; //lazy: tune: to use break, only find unique results!
					}
				}
#endif
				offset = offset + blockDim.x;
			}
			__syncthreads();	//avoid next s_Table[] be dirtied
			px = px + gridDim.x;
		}
		else//((r2 - r1) >= (s2 - s1))	//read S, scan R
		{
			offset = s1 + threadIdx.x;
			for(int i = 0; offset < s2; ++i)
			{
				s_Table[threadIdx.x + i * blockDim.x] = d_S[offset];
				offset = offset + blockDim.x;
			}
			__syncthreads();
			//read R
			offset = r1 + threadIdx.x;
			for(int i = 0; offset < r2; ++i)
			{
				tmpRec = d_R[offset];
#if _bSortPart
				firstHit = FirstHit(s_Table, s2 - s1, tmpRec.y);
				if(firstHit >= 0)
				{
					do{
						++count;
						++firstHit;
					}while(s_Table[firstHit].y == tmpRec.y);
				}
#else
				for(int j = 0; j < (s2 - s1); ++j)
				{
					if(s_Table[j].y == tmpRec.y)
					{
						++count; //lazy: tune: to use break, only find unique results!
					}
				}
#endif
				offset = offset + blockDim.x;
			}
			__syncthreads();	//avoid next s_Table[] be dirtied
			px = px + gridDim.x;
		}
	}
	d_ThreadCnts[blockIdx.x * blockDim.x + threadIdx.x] = count;
}

#ifndef SHARED_MEM
__global__ void Probe_Cnt_noShared(Record* d_Table, int sharedSize, int* d_ThreadCnts, int* d_skewPid, Record* d_R, const int* d_PBoundR, Record* d_S, 
							const int* d_PBoundS, const int PnG)
{
	//extern __shared__ Record s_Table[];
	Record* s_Table;
	s_Table = d_Table + blockIdx.x*( sharedSize/sizeof(Record) );

	int count = 0;	//important
	int px = blockIdx.x;	//part idx
	Record tmpRec;
	int r1, r2, s1, s2;
	int offset/*, firstHit*/;
	while(px < PnG)
	{
		r1 = d_PBoundR[px];
		r2 = d_PBoundR[px + 1];
		s1 = d_PBoundS[px];
		s2 = d_PBoundS[px + 1];
		if((r2 - r1 > _maxPartLen) && (s2 - s1 > _maxPartLen))
		{
			d_skewPid[px] = 1;
			__syncthreads();	//avoid next s_Table[] be dirtied
			px = px + gridDim.x;	//skip, leave to later
		}
		else if((r2 - r1) <= (s2 - s1))	//read R, scan S
		{
			offset = r1 + threadIdx.x;
			for(int i = 0; offset < r2; ++i)
			{
				s_Table[threadIdx.x + i * blockDim.x] = d_R[offset];
				offset = offset + blockDim.x;
			}
			__syncthreads();
			//read S
			offset = s1 + threadIdx.x;
			for(int i = 0; offset < s2; ++i)
			{
				tmpRec = d_S[offset];
#if _bSortPart
				firstHit = FirstHit(s_Table, r2 - r1, tmpRec.y);
				if(firstHit >= 0)
				{
					do{
						++count;
						++firstHit;
					}while(s_Table[firstHit].y == tmpRec.y);
				}
#else
				for(int j = 0; j < (r2 - r1); ++j)
				{
					if(s_Table[j].y == tmpRec.y)
					{
						++count; //lazy: tune: to use break, only find unique results!
					}
				}
#endif
				offset = offset + blockDim.x;
			}
			__syncthreads();	//avoid next s_Table[] be dirtied
			px = px + gridDim.x;
		}
		else//((r2 - r1) >= (s2 - s1))	//read S, scan R
		{
			offset = s1 + threadIdx.x;
			for(int i = 0; offset < s2; ++i)
			{
				s_Table[threadIdx.x + i * blockDim.x] = d_S[offset];
				offset = offset + blockDim.x;
			}
			__syncthreads();
			//read R
			offset = r1 + threadIdx.x;
			for(int i = 0; offset < r2; ++i)
			{
				tmpRec = d_R[offset];
#if _bSortPart
				firstHit = FirstHit(s_Table, s2 - s1, tmpRec.y);
				if(firstHit >= 0)
				{
					do{
						++count;
						++firstHit;
					}while(s_Table[firstHit].y == tmpRec.y);
				}
#else
				for(int j = 0; j < (s2 - s1); ++j)
				{
					if(s_Table[j].y == tmpRec.y)
					{
						++count; //lazy: tune: to use break, only find unique results!
					}
				}
#endif
				offset = offset + blockDim.x;
			}
			__syncthreads();	//avoid next s_Table[] be dirtied
			px = px + gridDim.x;
		}
	}
	d_ThreadCnts[blockIdx.x * blockDim.x + threadIdx.x] = count;
}
#endif

//use s to probe r, write to global positions that d_ThreadCntsPresum indicates
__global__ void Probe_Write(Record* d_RS, int* d_ThreadCntsPresum, Record* d_R, const int* d_PBoundR,
						   Record* d_S, const int* d_PBoundS, const int PnG)
{
	extern __shared__ Record s_Table[];
	int outputPos = d_ThreadCntsPresum[blockIdx.x * blockDim.x + threadIdx.x];	//output's starting offset
	int px = blockIdx.x;	//part idx
	Record tmpRec;
	Record tmpRS;
	int r1, r2, s1, s2;
	int offset/*, firstHit*/;
	while(px < PnG)
	{
		r1 = d_PBoundR[px];
		r2 = d_PBoundR[px + 1];
		s1 = d_PBoundS[px];
		s2 = d_PBoundS[px + 1];
		if((r2 - r1 > _maxPartLen) && (s2 - s1 > _maxPartLen))
		{
			__syncthreads();	//avoid next s_Table[] be dirtied
			px = px + gridDim.x;	//skip, leave to later
		}
		else if((r2 - r1) <= (s2 - s1))	//read R, scan S
		{
			offset = r1 + threadIdx.x;

			for(int i = 0; offset < r2; ++i)
			{
				s_Table[threadIdx.x + i * blockDim.x] = d_R[offset];
				offset = offset + blockDim.x;
			}
			__syncthreads();
			//scan S
			offset = s1 + threadIdx.x;

			for(int i = 0; offset < s2; ++i)
			{
				tmpRec = d_S[offset];
#if _bSortPart
				firstHit = FirstHit(s_Table, r2 - r1, tmpRec.y);
				if(firstHit >= 0)
				{
					do{
						tmpRS.x = s_Table[firstHit].x;	//x
						tmpRS.y = tmpRec.x;	//sid
						d_RS[outputPos] = tmpRS;
						++outputPos;
						++firstHit;
					}
					while(s_Table[firstHit].y == tmpRec.y);
				}
#else
				for(int j = 0; j < (r2 - r1); ++j)
				{
					if(s_Table[j].y == tmpRec.y)
					{
						tmpRS.x = s_Table[j].x;	//x
						tmpRS.y = tmpRec.x;	//sid
						d_RS[outputPos] = tmpRS;
						++outputPos;
					}
				}
#endif
				offset = offset + blockDim.x;
			}
			__syncthreads();	//avoid next s_Table[] be dirtied
			px = px + gridDim.x;
		}
		else//((r2 - r1) >= (s2 - s1))	//read S, scan R
		{
			offset = s1 + threadIdx.x;
			for(int i = 0; offset < s2; ++i)
			{
				s_Table[threadIdx.x + i * blockDim.x] = d_S[offset];
				offset = offset + blockDim.x;
			}
			__syncthreads();
			//scan R
			offset = r1 + threadIdx.x;
			for(int i = 0; offset < r2; ++i)
			{
				tmpRec = d_R[offset];
#if _bSortPart
				firstHit = FirstHit(s_Table, s2 - s1, tmpRec.y);
				if(firstHit >= 0)
				{
					do{
						tmpRS.x = tmpRec.x;	//x
						tmpRS.y = s_Table[firstHit].x;	//sid
						d_RS[outputPos] = tmpRS;
						++outputPos;
						++firstHit;
					}
					while(s_Table[firstHit].y == tmpRec.y);
				}

#else
				for(int j = 0; j < (s2 - s1); ++j)
				{
					if(s_Table[j].y == tmpRec.y)
					{
						tmpRS.x = tmpRec.x;			//x
						tmpRS.y = s_Table[j].x;	//sid
						d_RS[outputPos] = tmpRS;
						++outputPos;
					}
				}
#endif
				offset = offset + blockDim.x;
			}
			__syncthreads();	//avoid next s_Table[] be dirtied
			px = px + gridDim.x;
		}
	}
}

#ifndef SHARED_MEM
__global__ void Probe_Write_noShared(Record* d_Table, int sharedSize, Record* d_RS, int* d_ThreadCntsPresum, Record* d_R, const int* d_PBoundR,
						   Record* d_S, const int* d_PBoundS, const int PnG)
{
	//extern __shared__ Record s_Table[];
	Record* s_Table;
	s_Table = d_Table + blockIdx.x*( sharedSize/sizeof(Record) );
	int outputPos = d_ThreadCntsPresum[blockIdx.x * blockDim.x + threadIdx.x];	//output's starting offset
	int px = blockIdx.x;	//part idx
	Record tmpRec;
	Record tmpRS;
	int r1, r2, s1, s2;
	int offset/*, firstHit*/;
	while(px < PnG)
	{
		r1 = d_PBoundR[px];
		r2 = d_PBoundR[px + 1];
		s1 = d_PBoundS[px];
		s2 = d_PBoundS[px + 1];
		if((r2 - r1 > _maxPartLen) && (s2 - s1 > _maxPartLen))
		{
			__syncthreads();	//avoid next s_Table[] be dirtied
			px = px + gridDim.x;	//skip, leave to later
		}
		else if((r2 - r1) <= (s2 - s1))	//read R, scan S
		{
			offset = r1 + threadIdx.x;

			for(int i = 0; offset < r2; ++i)
			{
				s_Table[threadIdx.x + i * blockDim.x] = d_R[offset];
				offset = offset + blockDim.x;
			}
			__syncthreads();
			//scan S
			offset = s1 + threadIdx.x;

			for(int i = 0; offset < s2; ++i)
			{
				tmpRec = d_S[offset];
#if _bSortPart
				firstHit = FirstHit(s_Table, r2 - r1, tmpRec.y);
				if(firstHit >= 0)
				{
					do{
						tmpRS.x = s_Table[firstHit].x;	//x
						tmpRS.y = tmpRec.x;	//sid
						d_RS[outputPos] = tmpRS;
						++outputPos;
						++firstHit;
					}
					while(s_Table[firstHit].y == tmpRec.y);
				}
#else
				for(int j = 0; j < (r2 - r1); ++j)
				{
					if(s_Table[j].y == tmpRec.y)
					{
						tmpRS.x = s_Table[j].x;	//x
						tmpRS.y = tmpRec.x;	//sid
						d_RS[outputPos] = tmpRS;
						++outputPos;
					}
				}
#endif
				offset = offset + blockDim.x;
			}
			__syncthreads();	//avoid next s_Table[] be dirtied
			px = px + gridDim.x;
		}
		else//((r2 - r1) >= (s2 - s1))	//read S, scan R
		{
			offset = s1 + threadIdx.x;
			for(int i = 0; offset < s2; ++i)
			{
				s_Table[threadIdx.x + i * blockDim.x] = d_S[offset];
				offset = offset + blockDim.x;
			}
			__syncthreads();
			//scan R
			offset = r1 + threadIdx.x;
			for(int i = 0; offset < r2; ++i)
			{
				tmpRec = d_R[offset];
#if _bSortPart
				firstHit = FirstHit(s_Table, s2 - s1, tmpRec.y);
				if(firstHit >= 0)
				{
					do{
						tmpRS.x = tmpRec.x;	//x
						tmpRS.y = s_Table[firstHit].x;	//sid
						d_RS[outputPos] = tmpRS;
						++outputPos;
						++firstHit;
					}
					while(s_Table[firstHit].y == tmpRec.y);
				}

#else
				for(int j = 0; j < (s2 - s1); ++j)
				{
					if(s_Table[j].y == tmpRec.y)
					{
						tmpRS.x = tmpRec.x;			//x
						tmpRS.y = s_Table[j].x;	//sid
						d_RS[outputPos] = tmpRS;
						++outputPos;
					}
				}
#endif
				offset = offset + blockDim.x;
			}
			__syncthreads();	//avoid next s_Table[] be dirtied
			px = px + gridDim.x;
		}
	}
}
#endif

//the overflowed partition falls naturally into some fragments, each length parLen/shareMemsize. then each block handles one frag in strip-mining way.
__global__ void Probe_CntOverflow(int* d_ThreadCnts, const int skewPid, Record* d_R, const int* d_PBoundR, Record* d_S, 
							const int* d_PBoundS)
{
	extern __shared__ Record s_Table[];
	const int rStart = d_PBoundR[skewPid];
	const int rEnd = d_PBoundR[skewPid + 1];
	const int sStart = d_PBoundS[skewPid];
	const int sEnd = d_PBoundS[skewPid + 1];
	int count = 0;
	Record tmpRec;
	if((rEnd - rStart) <= (sEnd - sStart))
	{
		const int nFrag = (int)ceilf((float)(rEnd - rStart) / _maxPartLen);
		int FIdx = blockIdx.x;
		while(FIdx < nFrag)
		{
			//read r fragment
			int fragStart = rStart + FIdx * _maxPartLen;
			int fragEnd = (fragStart + _maxPartLen < rEnd)? (fragStart + _maxPartLen) : rEnd;
			int offset = fragStart + threadIdx.x;	//global
			for(int i = 0; offset < fragEnd; ++i)
			{
				s_Table[threadIdx.x + i * blockDim.x] = d_R[offset];
				offset = offset + blockDim.x;
			}
			__syncthreads();
			//read whole S partition
			offset = sStart + threadIdx.x;
			for(int i = 0; offset < sEnd; ++i)
			{
				tmpRec = d_S[offset];
				for(int j = 0; j < fragEnd - fragStart; ++j)
				{
					if(s_Table[j].y == tmpRec.y)
						++count;
				}
				offset = offset + blockDim.x;
			}
			__syncthreads();
			FIdx = FIdx + gridDim.x;
		}
	}
	else //(rEnd - rStart) >= (sEnd - sStart)), use s as inner table
	{
		int nFrag = (int)ceilf((float)(sEnd - sStart) / _maxPartLen);
		int FIdx = blockIdx.x;
		while(FIdx < nFrag)
		{
			//read s fragment
			int fragStart = sStart + FIdx * _maxPartLen;
			int fragEnd = (fragStart + _maxPartLen < sEnd)? (fragStart + _maxPartLen) : sEnd;
			int offset = fragStart + threadIdx.x;	//global
			for(int i = 0; offset < fragEnd; ++i)
			{
				s_Table[threadIdx.x + i * blockDim.x] = d_S[offset];
				offset = offset + blockDim.x;
			}
			__syncthreads();
			//read whole R partition
			offset = rStart + threadIdx.x;
			for(int i = 0; offset < rEnd; ++i)
			{
				tmpRec = d_R[offset];
				for(int j = 0; j < fragEnd - fragStart; ++j)
				{
					if(s_Table[j].y == tmpRec.y)
						++count;
				}
				offset = offset + blockDim.x;
			}
			__syncthreads();
			FIdx = FIdx + gridDim.x;
		}
	}
	d_ThreadCnts[blockIdx.x * blockDim.x + threadIdx.x] = count;
}

//the overflowed partition falls naturally into some fragments, each length parLen/shareMemsize. then each block handles one frag in strip-mining way.
__global__ void Probe_WriteOverflow(Record* d_RS, int* d_WriteLoc, const int skewPid, Record* d_R, const int* d_PBoundR, Record* d_S, 
							const int* d_PBoundS)
{
	extern __shared__ Record s_Table[];
	const int rStart = d_PBoundR[skewPid];
	const int rEnd = d_PBoundR[skewPid + 1];
	const int sStart = d_PBoundS[skewPid];
	const int sEnd = d_PBoundS[skewPid + 1];
	int outputPos = d_WriteLoc[blockIdx.x * blockDim.x + threadIdx.x];
		
	Record tmpRec, tmpRS;
	if((rEnd - rStart) <= (sEnd - sStart))
	{
		const int nFrag = (int)ceilf((float)(rEnd - rStart) / _maxPartLen);
		int FIdx = blockIdx.x;
		while(FIdx < nFrag)
		{
			//read r fragment
			int fragStart = rStart + FIdx * _maxPartLen;
			int fragEnd = min(fragStart + _maxPartLen, rEnd);
			int offset = fragStart + threadIdx.x;	//global
			for(int i = 0; offset < fragEnd; ++i)
			{
				s_Table[threadIdx.x + i * blockDim.x] = d_R[offset];
				offset = offset + blockDim.x;
			}
			__syncthreads();
			//read whole S partition
			offset = sStart + threadIdx.x;
			for(int i = 0; offset < sEnd; ++i)
			{
				tmpRec = d_S[offset];
				for(int j = 0; j < fragEnd - fragStart; ++j)
				{
					if(s_Table[j].y == tmpRec.y)
					{
						tmpRS.x = s_Table[j].x;	//x
						tmpRS.y = tmpRec.x;	//sid
						d_RS[outputPos] = tmpRS;
						++outputPos;
					}
				}
				offset = offset + blockDim.x;
			}
			__syncthreads();
			FIdx = FIdx + gridDim.x;
		}
	}
	else //(rEnd - rStart) >= (sEnd - sStart)), use s as inner table
	{
		int nFrag = (int)ceilf((float)(sEnd - sStart) / _maxPartLen);
		int FIdx = blockIdx.x;
		while(FIdx < nFrag)
		{
			//read s fragment
			int fragStart = sStart + FIdx * _maxPartLen;
			int fragEnd = (fragStart + _maxPartLen < sEnd)? (fragStart + _maxPartLen) : sEnd;
			int offset = fragStart + threadIdx.x;	//global
			for(int i = 0; offset < fragEnd; ++i)
			{
				s_Table[threadIdx.x + i * blockDim.x] = d_S[offset];
				offset = offset + blockDim.x;
			}
			__syncthreads();
			//read whole R partition
			offset = rStart + threadIdx.x;
			for(int i = 0; offset < rEnd; ++i)
			{
				tmpRec = d_R[offset];
				for(int j = 0; j < fragEnd - fragStart; ++j)
				{
					if(s_Table[j].y == tmpRec.y)
					{
						tmpRS.x = tmpRec.x;			//x
						tmpRS.y = s_Table[j].x;	//sid
						d_RS[outputPos] = tmpRS;
						++outputPos;
					}
				}
				offset = offset + blockDim.x;
			}
			__syncthreads();
			FIdx = FIdx + gridDim.x;
		}
	}
}

#ifndef SHARED_MEM
__global__ void Probe_WriteOverflow_noShared(Record* d_Table, int sharedSize, Record* d_RS, int* d_WriteLoc, const int skewPid, Record* d_R, const int* d_PBoundR, Record* d_S, 
							const int* d_PBoundS)
{
	//extern __shared__ Record s_Table[];
	Record* s_Table;
	s_Table = d_Table + blockIdx.x*( sharedSize/sizeof(Record) );

	const int rStart = d_PBoundR[skewPid];
	const int rEnd = d_PBoundR[skewPid + 1];
	const int sStart = d_PBoundS[skewPid];
	const int sEnd = d_PBoundS[skewPid + 1];
	int outputPos = d_WriteLoc[blockIdx.x * blockDim.x + threadIdx.x];
		
	Record tmpRec, tmpRS;
	if((rEnd - rStart) <= (sEnd - sStart))
	{
		const int nFrag = (int)ceilf((float)(rEnd - rStart) / _maxPartLen);
		int FIdx = blockIdx.x;
		while(FIdx < nFrag)
		{
			//read r fragment
			int fragStart = rStart + FIdx * _maxPartLen;
			int fragEnd = min(fragStart + _maxPartLen, rEnd);
			int offset = fragStart + threadIdx.x;	//global
			for(int i = 0; offset < fragEnd; ++i)
			{
				s_Table[threadIdx.x + i * blockDim.x] = d_R[offset];
				offset = offset + blockDim.x;
			}
			__syncthreads();
			//read whole S partition
			offset = sStart + threadIdx.x;
			for(int i = 0; offset < sEnd; ++i)
			{
				tmpRec = d_S[offset];
				for(int j = 0; j < fragEnd - fragStart; ++j)
				{
					if(s_Table[j].y == tmpRec.y)
					{
						tmpRS.x = s_Table[j].x;	//x
						tmpRS.y = tmpRec.x;	//sid
						d_RS[outputPos] = tmpRS;
						++outputPos;
					}
				}
				offset = offset + blockDim.x;
			}
			__syncthreads();
			FIdx = FIdx + gridDim.x;
		}
	}
	else //(rEnd - rStart) >= (sEnd - sStart)), use s as inner table
	{
		int nFrag = (int)ceilf((float)(sEnd - sStart) / _maxPartLen);
		int FIdx = blockIdx.x;
		while(FIdx < nFrag)
		{
			//read s fragment
			int fragStart = sStart + FIdx * _maxPartLen;
			int fragEnd = (fragStart + _maxPartLen < sEnd)? (fragStart + _maxPartLen) : sEnd;
			int offset = fragStart + threadIdx.x;	//global
			for(int i = 0; offset < fragEnd; ++i)
			{
				s_Table[threadIdx.x + i * blockDim.x] = d_S[offset];
				offset = offset + blockDim.x;
			}
			__syncthreads();
			//read whole R partition
			offset = rStart + threadIdx.x;
			for(int i = 0; offset < rEnd; ++i)
			{
				tmpRec = d_R[offset];
				for(int j = 0; j < fragEnd - fragStart; ++j)
				{
					if(s_Table[j].y == tmpRec.y)
					{
						tmpRS.x = tmpRec.x;			//x
						tmpRS.y = s_Table[j].x;	//sid
						d_RS[outputPos] = tmpRS;
						++outputPos;
					}
				}
				offset = offset + blockDim.x;
			}
			__syncthreads();
			FIdx = FIdx + gridDim.x;
		}
	}
}

#endif

#endif
