
#ifndef _RadixHJ_KERNEL_CU_
#define _RadixHJ_KERNEL_CU_

#include "cutil.h"

#define _bDebug 0
#define _JoinPartMax (_bDebug? 2: 512)	//SM / sizeof(rec) = 8k/8 =1k; bucket skew-> /2 = 0.5k
#define HASH(v) (_bDebug ? ((unsigned int) v) : ((unsigned int)( (v >> 7) ^ (v >> 13) ^ (v >>21) ^ (v) )) )

//	Histo: scan R and get d_PidHisto[pn] of each thread
//		p=hash(r); histo[p]++; sync; d_PidHisto[snakewise p]=histo;
__global__ void Histo(int* d_HistoMat, Record* d_R, const int nR, const int pn, const int shift)
{
	const int gridLen = gridDim.x * blockDim.x;
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int pid;
	extern __shared__ int histo[];
	for(int pi = 0; pi < pn; ++pi)
	{
		histo[threadIdx.x * pn + pi] = 0;
	}
	while(offset < nR)
	{
		pid = (HASH(d_R[offset].y) >> shift) & (pn - 1);	
		++histo[threadIdx.x * pn + pid];
		offset = offset + gridLen;
	}
	__syncthreads();
	offset = blockIdx.x * blockDim.x + threadIdx.x;
	for(int pi = 0; pi < pn; ++pi)
	{
		d_HistoMat[pi * gridLen + offset] = histo[threadIdx.x * pn + pi];
	}
}

__global__ void Reorder(Record* d_R1, int* d_PBound, Record* d_R, int* d_WriteLoc, const int nR, const int pn, const int shift)
{
	int gridLen = gridDim.x * blockDim.x;
	int pid;
	Record recR;
	extern __shared__ int writeLoc[];
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	for(int pi = 0; pi < pn; ++pi)
	{
		writeLoc[threadIdx.x * pn + pi] = d_WriteLoc[pi * gridLen + offset];
	}
    if(offset == 0)//this thread's corresponding HistMatPre contains d_partbound2
	{
		for(int pi = 0; pi < pn; ++pi)
		{
			d_PBound[pi] = writeLoc[threadIdx.x * pn + pi];
		}
		d_PBound[pn] = nR;
	}
	while(offset < nR)
	{
		recR = d_R[offset];
		pid = threadIdx.x * pn + ((HASH(recR.y) >> shift) & (pn - 1));
		//pid = threadIdx.x * pn + d_Pid[offset];
		d_R1[writeLoc[pid]] = recR;
		++writeLoc[pid];
		offset = offset + gridLen;
	}
}

//	Histo: scan R and get d_PID and d_PidHisto of each thread
//		p=d_PID[r]=hash(r); histo[p]++; sync; d_PidHisto[snakewise p]=histo;
__global__ void Histo2(int* d_HistoMat, Record* d_R, int* d_PBound, int unitId, const int pn, const int shift)
{
	const int PStart = d_PBound[unitId];
	const int nR = d_PBound[unitId + 1] -  PStart;
	const int gridLen = gridDim.x * blockDim.x;
	d_R = d_R + PStart;
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int pid;
	extern __shared__ int histo[];	//tune: int->tn?
	for(int pi = 0; pi < pn; ++pi)
	{
		histo[threadIdx.x * pn + pi] = 0;
	}
	while(offset < nR)
	{
		pid = (HASH(d_R[offset].y) >> shift) & (pn - 1);	
		++histo[threadIdx.x * pn + pid];	//tune: sharemem bank coalesce?
		offset = offset + gridLen;
	}
	offset = blockIdx.x * blockDim.x + threadIdx.x;
	for(int pi = 0; pi < pn; ++pi)
	{
		d_HistoMat[pi * gridLen + offset] = histo[threadIdx.x * pn + pi];
	}
}

//write d_R1 to d_R according to d_WriteLoc and d_Pid; write d_PboundR2 using d_PBoundR and d_WL
__global__ void Reorder2(Record* d_R, int* d_PBoundR2, Record* d_R1, int* d_PBoundR, int unitId, 
						 int* d_WriteLoc, const int Pn2, const int shift)
{
	Record recR;
	const int PStart = d_PBoundR[unitId];
	d_R = d_R + PStart;
	d_R1 = d_R1 + PStart;
	const int gridLen = gridDim.x * blockDim.x;
	const int nSub = d_PBoundR[unitId + 1] - PStart;
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int pid;
	extern __shared__ int s_writeLoc[];
	for(int pi = 0; pi < Pn2; ++pi)
	{
		s_writeLoc[threadIdx.x * Pn2 + pi] = d_WriteLoc[pi * gridLen + offset];
	}
    if(offset == 0)//this thread's corresponding HistMatPre contains d_partbound2
	{
		for(int pi = 0; pi < Pn2; ++pi)
		{
			d_PBoundR2[pi] = PStart + s_writeLoc[threadIdx.x * Pn2 + pi];
		}
		d_PBoundR2[Pn2] = PStart + nSub;
	}
	while(offset < nSub)
	{
		recR = d_R1[offset];
		pid = threadIdx.x * Pn2 + ((HASH(recR.y) >> shift) & (Pn2 - 1));
		d_R[s_writeLoc[pid]] = recR;//tune: sharemem bank coalesce?
		++s_writeLoc[pid];
		offset = offset + gridLen;
	}
}


//find cnt for each threads
__global__ void Joining_Cnt(int* d_ThreadCnts, int* d_skewPid, Record* d_R, const int* d_PBoundR, Record* d_S, 
							const int* d_PBoundS, const int PnG)
{
	extern __shared__ Record s_Table[];
	int count = 0;	//important
	int px = blockIdx.x;	//part idx
	Record tmpRec;
	int r1, r2, s1, s2;
	int offset;
	while(px < PnG)
	{
		r1 = d_PBoundR[px];
		r2 = d_PBoundR[px + 1];
		s1 = d_PBoundS[px];
		s2 = d_PBoundS[px + 1];
		if((r2 - r1 > _JoinPartMax) && (s2 - s1 > _JoinPartMax))
		//ykdeb if((r2 - r1 > _JoinPartMax) || (s2 - s1 > _JoinPartMax))
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
				for(int j = 0; j < (r2 - r1); ++j)
				{
					if(s_Table[j].y == tmpRec.y)
					{
						++count; //lazy: tune: to use break, only find unique results!
					}
				}
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
				for(int j = 0; j < (s2 - s1); ++j)
				{
					if(s_Table[j].y == tmpRec.y)
					{
						++count; //lazy: tune: to use break, only find unique results!
					}
				}
				offset = offset + blockDim.x;
			}
			__syncthreads();	//avoid next s_Table[] be dirtied
			px = px + gridDim.x;
		}
	}
	d_ThreadCnts[blockIdx.x * blockDim.x + threadIdx.x] = count;
}

//use s to probe r, write to global positions that d_ThreadCntsPresum indicates
__global__ void Join_Write(Record* d_RS, int* d_ThreadCntsPresum, Record* d_R, const int* d_PBoundR,
						   Record* d_S, const int* d_PBoundS, const int PnG)
{
	extern __shared__ Record s_Table[];
	int outputPos = d_ThreadCntsPresum[blockIdx.x * blockDim.x + threadIdx.x];	//output's starting offset
	int px = blockIdx.x;	//part idx
	Record tmpRec;
	Record tmpRS;
	int r1, r2, s1, s2;
	int offset;
	while(px < PnG)
	{
		r1 = d_PBoundR[px];
		r2 = d_PBoundR[px + 1];
		s1 = d_PBoundS[px];
		s2 = d_PBoundS[px + 1];
		if((r2 - r1 > _JoinPartMax) && (s2 - s1 > _JoinPartMax))
		//if((r2 - r1 > _JoinPartMax) || (s2 - s1 > _JoinPartMax))
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
				offset = offset + blockDim.x;
			}
			__syncthreads();	//avoid next s_Table[] be dirtied
			px = px + gridDim.x;
		}
	}
}

//the overflowed partition falls naturally into some fragments, each length parLen/shareMemsize. then each block handles one frag in strip-mining way.
__global__ void Joining_CntOverflow(int* d_ThreadCnts, const int skewPid, Record* d_R, const int* d_PBoundR, Record* d_S, 
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
		const int nFrag = (int)ceilf((float)(rEnd - rStart) / _JoinPartMax);
		int FIdx = blockIdx.x;
		while(FIdx < nFrag)
		{
			//read r fragment
			int fragStart = rStart + FIdx * _JoinPartMax;
			int fragEnd = (fragStart + _JoinPartMax < rEnd)? (fragStart + _JoinPartMax) : rEnd;
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
		int nFrag = (int)ceilf((float)(sEnd - sStart) / _JoinPartMax);
		int FIdx = blockIdx.x;
		while(FIdx < nFrag)
		{
			//read s fragment
			int fragStart = sStart + FIdx * _JoinPartMax;
			int fragEnd = (fragStart + _JoinPartMax < sEnd)? (fragStart + _JoinPartMax) : sEnd;
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

#endif
