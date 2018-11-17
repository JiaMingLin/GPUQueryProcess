
//implementations of radix-clustered hash join

//Ke Yang

#ifndef _RadixClusteredHashJoin_H_
#define _RadixClusteredHashJoin_H_

#include "Header.cu"
#include "Util.cu"
#include "RadixClusteredHashJoin_kernel.cu"
#include "scan.cu"
#include "GPU_Dll.h"


int myLog2(int a)	//a should be 2^x. return x
{
	int x = 0;
	while (a = (a>>1))
		++x;
	return x;
}

//down large to multiples of small
int DownToMulti(int large, int small)
{
	if(large < small)
		return large;
	int res = 0;
	do{
		res += small;
	}	
	while(res < large);
	return res;
}

//return nearest (no less than) pow of 2 from a
int UpNearestPowOf2(int a)
{
	int tmp = 1;
	while(tmp < a)
		tmp *= 2;
	return tmp;
}


//make a plan of num of passes, and each pass performs on a how long radix.
//input:
//PnG: num of total partitions of R
//output:
//nPasses: num of passes to go through
//Pns: PnG of each pass, should be 2^
void RadixPlan(int* Pns, int& nPass, int PnG)
{
	// why the max partition number is 256?
	int maxPn = _bDebug? 2: 256;
	int curTotal = 1; //current total num of partitions
	nPass = 0;
	while(curTotal < PnG)
	{
		Pns[nPass] = min(maxPn, IntCeilDiv(PnG, curTotal));
		curTotal *= Pns[nPass];
		++nPass;
	}
	printf("radix plan: %d, %d, %d\n", Pns[0], Pns[1], Pns[2]);
}

void PartitionPlan(int& Bn, int& Tn, const int Pn, const int nParent)	//good
{
	Bn = _bDebug ? 2: max(nParent, 256); //16-1024 are the same
	Tn = _bDebug ? 2: DownToMulti(int(2048 / Pn), 16);
	//printf("partition Bn: %d, Tn: %d\n", Bn, Tn);
}

//to avoid wasting, Tn < partLen, about _maxPartLen
void ProbePlan(int& Bn, int& Tn)
{
	Bn = _bDebug ? 2: 512; //512; //>64 the same
	Tn = _bDebug ? 2: 128; //384; //>64 the same, but 512 will be wrong
	//printf("probe Bn: %d, Tn: %d\n", Bn, Tn);
}

//if _bDebug watch elements in device
inline void DebugRecs(Record* d_A, int nA)
{
#ifdef DEBUG
	printf("\n~~~~~~~~~~~~~~debug~~~~~~~~~~~~~~~~\n");
	Record* h_test = (Record*)malloc( sizeof(Record) * nA);
	CUDA_SAFE_CALL(cudaMemcpy(h_test, d_A, sizeof(Record) * nA, cudaMemcpyDeviceToHost));
	SAFE_FREE(h_test);
#endif	
}

inline void DebugInts(int* d_A, int nA)
{
#ifdef DEBUG
	printf("\n~~~~~~~~~~~~~~debug~~~~~~~~~~~~~~~~\n");
	int* h_test = (int*)malloc( sizeof(int) * nA);
	CUDA_SAFE_CALL(cudaMemcpy(h_test, d_A, sizeof(int) * nA, cudaMemcpyDeviceToHost));
	SAFE_FREE(h_test);
#endif	
}

//partition and reorder d_R[nR] into d_R1, with Pn partitions starting from d_RDir. The radix of clustering starts from radixShift from the rightmost.
void cuda_Partition(
		Record** d_R1,
		int*& d_RDir,
		Record*& d_R,
		const int nR,
		const int Pn,
		const int radixShift)
{
	unsigned int timer = 0;
	//CUT_SAFE_CALL(cutCreateTimer(&timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

	//paras
	int Bn, Tn;
	PartitionPlan(Bn, Tn, Pn, 1);

	//histoMat
	dim3  Dg(Bn, 1, 1);
	dim3  Db(Tn, 1, 1);
	int* d_HistM;	//histomat: bn*tn rows, Pn cols
	GPUMALLOC((void**) &d_HistM, sizeof(int) * Pn * Bn * Tn);
	int Ns = Tn * Pn * sizeof(int);	//each thread has a int[pn1]
	assert(Ns <= 8192); 
#ifdef SHARED_MEM
	Histo<<<Dg, Db, Ns>>> (d_HistM, d_R, nR, Pn, radixShift);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#else
	int* d_histo;
	GPUMALLOC( (void**)&d_histo, Ns*Dg.x );
	Histo_noShared<<<Dg, Db>>> (d_histo, Ns, d_HistM, d_R, nR, Pn, radixShift);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	GPUFREE( d_histo );
#endif
	CUT_CHECK_ERROR("Histo");
	DebugInts(d_HistM, Bn * Tn * Pn);

	//CUT_SAFE_CALL(cutStopTimer(timer));
	//printf("	histo1: %f (ms)\n", cutGetTimerValue(timer));
	//CUT_SAFE_CALL(cutResetTimer(timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

	//presum
	int* d_Loc = NULL;
    GPUMALLOC( (void**) &d_Loc, sizeof(int) * Pn * Bn * Tn);
	//saven_initialPrefixSum(Pn * Bn * Tn);
	//prescanArray(d_Loc, d_HistM, 16);
	scanImpl(d_HistM,Pn * Bn * Tn,d_Loc);
	//prescanArray(d_Loc, d_HistM, Pn * Bn * Tn);
	//deallocBlockSums();
	CUT_CHECK_ERROR("kernel error");
	SAFE_CUDA_FREE((d_HistM));
	DebugInts(d_Loc, Bn * Tn * Pn);

	//CUT_SAFE_CALL(cutStopTimer(timer));
	//printf("	presum1: %f (ms)\n", cutGetTimerValue(timer));
	//CUT_SAFE_CALL(cutResetTimer(timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

	//reorder
	GPUMALLOC((void**) d_R1, sizeof(Record) * nR);

#ifdef SHARED_MEM
	Reorder<<<Dg, Db, Ns>>> (*d_R1, d_RDir, d_R, d_Loc, nR, Pn, radixShift);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#else
	int* d_writeLoc;
	GPUMALLOC( (void**)&d_writeLoc, Ns*Dg.x );
	Reorder_noShared<<<Dg, Db>>> (d_writeLoc, Ns, *d_R1, d_RDir, d_R, d_Loc, nR, Pn, radixShift);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	GPUFREE( d_writeLoc );
#endif
	DebugRecs(*d_R1, nR);
	DebugInts(d_RDir, Pn + 1);
	SAFE_CUDA_FREE((d_Loc));

	//CUT_SAFE_CALL(cutStopTimer(timer));
	//printf("	reorder1: %f (ms)\n", cutGetTimerValue(timer));
	//CUT_SAFE_CALL(cutResetTimer(timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

}

//partition nParent parents, d_R[RDir[pId]] into d_R1 and dRDir1[Pn].
void cuda_Partition2(Record*& d_R1, int*& d_RDir1, Record*& d_R, const int nR, const int nParent, const int Pn, const int radixShift, int* const d_RDir)
{
	unsigned int timer = 0;
	//CUT_SAFE_CALL(cutCreateTimer(&timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

	int Bn, Tn;
	PartitionPlan(Bn, Tn, Pn, nParent);
	//each parent uses Bp blocks
	int Bp = Bn / nParent;
	assert(Bp > 0);
	assert(Bp == int(float(Bn)/nParent));	//Bn devides nParent

	dim3  Dg(Bn, 1, 1);
	dim3  Db(Tn, 1, 1);
	
	//output histoMat
	int* d_HistM;	//histomat: bn*tn rows, Pn cols
	GPUMALLOC((void**) &d_HistM, sizeof(int) * Pn * Bn * Tn);
	int Ns = Tn * Pn * sizeof(int);	//each thread has a int[pn1]
	assert(Ns <= 8192); 
	if(Bn * Tn > nR)
		printf("Bn * Tn > nR, waste threads\n");

	//1. histo all parents in 1 kernel. Parent i corresponds to HistM[i*Bp]
#ifdef SHARED_MEM
	Histo3<<<Dg, Db, Ns>>> (d_HistM, d_R, d_RDir, nParent, Pn, radixShift);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#else
	int* d_histo;
	GPUMALLOC( (void**)&d_histo, Dg.x*Ns );
	Histo3_noShared<<<Dg, Db>>> (d_histo, Ns, d_HistM, d_R, d_RDir, nParent, Pn, radixShift);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	GPUFREE( d_histo );
#endif
	DebugInts(d_HistM, Bn * Tn * Pn);

	//CUT_SAFE_CALL(cutStopTimer(timer));
	//printf("		histo2: %f (ms)\n", cutGetTimerValue(timer));
	//CUT_SAFE_CALL(cutResetTimer(timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

	//output presum
	int* d_Loc = NULL;
	GPUMALLOC( (void**) &d_Loc, sizeof(int) * Pn * Bn * Tn);
	//saven_initialPrefixSum(Pn * Bn * Tn);
	//prescanArray(d_Loc, d_HistM, _bDebug ? 8 : 16);
	//2. presum every parent, sized (Bp * Tn * Pn). note the resulting d_Loc is still local, partition range!
	clock_t l_startTime = clock();
	preallocBlockSums(Bp * Tn * Pn);
	for(int pi = 0; pi < nParent; ++pi)
	{
		prescanArray(d_Loc + pi * Bp * Tn * Pn, d_HistM + pi * Bp * Tn * Pn, Bp * Tn * Pn);
		//scanImpl(d_HistM + pi * Bp * Tn * Pn, Bp * Tn * Pn,d_Loc + pi * Bp * Tn * Pn);
		DebugInts(d_Loc, Bn * Tn * Pn);
	}
	deallocBlockSums();
	CUT_CHECK_ERROR("kernel error");
	clock_t l_endTime = clock();
	printf("SAVEN, time, %.3f ms, \n",(l_endTime-l_startTime)*1000/ (double)CLOCKS_PER_SEC);

	//CUT_SAFE_CALL(cutStopTimer(timer));
	//printf("		presum2: %f (ms)\n", cutGetTimerValue(timer));
	//CUT_SAFE_CALL(cutResetTimer(timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

	//3. permute
#ifdef SHARED_MEM
	Reorder3<<<Dg, Db, Ns>>> (d_R1, d_RDir1, d_R, d_RDir, d_Loc, nParent, Pn, radixShift);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#else
	int* d_writeLoc;
	GPUMALLOC( (void**)&d_writeLoc, Dg.x*Ns );
	Reorder3_noShared<<<Dg, Db>>> (d_writeLoc, Ns, d_R1, d_RDir1, d_R, d_RDir, d_Loc, nParent, Pn, radixShift);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	GPUFREE( d_writeLoc );
#endif

	DebugRecs(d_R1, nR);
	DebugInts(d_RDir1, nParent * Pn + 1);

	//deallocBlockSums();
	CUT_CHECK_ERROR("kernel error");
	SAFE_CUDA_FREE((d_HistM));
	SAFE_CUDA_FREE((d_Loc));

	//CUT_SAFE_CALL(cutStopTimer(timer));
	//printf("		reorder2: %f (ms)\n", cutGetTimerValue(timer));
	//CUT_SAFE_CALL(cutResetTimer(timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

}

//build hash table for d_R[nR]. store clustered&rearranged R in dd_R1, each bucket's starting offset in it in d_RDir
void cuda_BuildHashTable(Record** dd_R1, int** dd_RDir, Record*& d_R, const int nR, int* const Pns, const int nPass)
{
	unsigned int timer = 0;
	//CUT_SAFE_CALL(cutCreateTimer(&timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));
	
	//pass 1
	int* d_RDir1 = NULL;
	GPUMALLOC((void**) &d_RDir1, sizeof(int) * (Pns[0] + 1));
	cuda_Partition(dd_R1, d_RDir1, d_R, nR, Pns[0], 0);

	DebugRecs(*dd_R1, nR);
	DebugInts(d_RDir1, Pns[0] * Pns[1] + 1);

	//CUT_SAFE_CALL(cutStopTimer(timer));
	//printf("pass1: %f (ms)\n", cutGetTimerValue(timer));
	//CUT_SAFE_CALL(cutResetTimer(timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

	if(nPass == 1)
	{
		SAFE_CUDA_FREE(d_R);
		*dd_RDir = d_RDir1;
		d_RDir1 = NULL;
		return;
	}

	//pass 2: further-partition each partition
	int* d_RDir2 = NULL;
	GPUMALLOC((void**) &d_RDir2, sizeof(int) * (Pns[0] * Pns[1] + 1));
	cuda_Partition2(d_R, d_RDir2, *dd_R1, nR, Pns[0], Pns[1], myLog2(Pns[0]), d_RDir1);	//pingpong between dd_R1 and d_R
	DebugRecs(d_R, nR);
	DebugInts(d_RDir2, Pns[0] * Pns[1] + 1);

	//CUT_SAFE_CALL(cutStopTimer(timer));
	//printf("pass2: %f (ms)\n", cutGetTimerValue(timer));
	//CUT_SAFE_CALL(cutResetTimer(timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

	if(nPass == 2)
	{
		SAFE_CUDA_FREE((*dd_R1));
		*dd_R1 = d_R;
		d_R = NULL;
		SAFE_CUDA_FREE((d_RDir1));
		*dd_RDir = d_RDir2;
		d_RDir2 = NULL;
		return;
	}

	//pass3: further-partition each partition
	int* d_RDir3 = NULL;
	GPUMALLOC((void**) &d_RDir3, sizeof(int) * (Pns[0] * Pns[1] * Pns[2] + 1));
	cuda_Partition2(*dd_R1, d_RDir3, d_R, nR, Pns[0] * Pns[1], Pns[2], myLog2(Pns[0] * Pns[1]), d_RDir2);	//pingpong between dd_R1 and d_R
	DebugRecs(*dd_R1, nR);
	DebugInts(d_RDir3, Pns[0] * Pns[1] * Pns[2] + 1);

	//CUT_SAFE_CALL(cutStopTimer(timer));
	//printf("pass3: %f (ms)\n", cutGetTimerValue(timer));
	//CUT_SAFE_CALL(cutResetTimer(timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

	*dd_RDir = d_RDir3;

	SAFE_CUDA_FREE(d_R);
	SAFE_CUDA_FREE((d_RDir1));
	SAFE_CUDA_FREE((d_RDir2));
}


//input: 
//d_R[nR], d_S[nS]: records clusted as buckets
//d_RDir[PnG + 1], d_SDir[PnG + 1]: bucket starting indices in d_R. d_RDir[PnG] = nR.
//output:
//d_R/d_S: for every corresponding pair of R, S buckets, the shorter one is sorted by rec.y increasingly, for bisearch when loaded in shared memory in Probe.
void cuda_ProbePreSort(Record* d_R, int* d_RDir, const int nR, Record* d_S, int* d_SDir, const int nS, const int PnG)
{
	unsigned int timer = 0;
	//CUT_SAFE_CALL(cutCreateTimer(&timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

	//same efficiency with multipasses
	const int _maxBn = 65535;	//by cuda1.0
	int BnW = _maxBn;
	int BnH = (int)ceilf(float(PnG) / _maxBn);
	int Tn = _maxPartLen;
	dim3  Dg(BnW, BnH, 1);

	dim3  Db(Tn, 1, 1);	
	int Ns = sizeof(Record) * Tn;	//each thread for 1 record
	DebugInts(d_RDir, PnG + 1);
	DebugInts(d_SDir, PnG + 1);
	ProbePreSort<<<Dg, Db, Ns>>>(d_R, d_RDir, nR, d_S, d_SDir, nS, PnG);
	CUT_CHECK_ERROR("Kernel failed");

#ifdef DEBUG
	Record* h_R = (Record*)malloc( sizeof(Record) * nR);
	CUDA_SAFE_CALL(cudaMemcpy(h_R, d_R, sizeof(Record) * nR, cudaMemcpyDeviceToHost));
	int* h_RDir = (int*)malloc( sizeof(int) * (PnG + 1));
	CUDA_SAFE_CALL(cudaMemcpy(h_RDir, d_RDir, sizeof(int) * (PnG + 1), cudaMemcpyDeviceToHost));

	Record* h_S = (Record*)malloc( sizeof(Record) * nS);
	CUDA_SAFE_CALL(cudaMemcpy(h_S, d_S, sizeof(Record) * nS, cudaMemcpyDeviceToHost));
	int* h_SDir = (int*)malloc( sizeof(int) * (PnG + 1));
	CUDA_SAFE_CALL(cudaMemcpy(h_SDir, d_SDir, sizeof(int) * (PnG + 1), cudaMemcpyDeviceToHost));

	int r1, r2, s1, s2;
	for(int partIdx = 0; partIdx < PnG; ++partIdx)
	{
		r1 = h_RDir[partIdx];
		r2 = h_RDir[partIdx + 1];
		s1 = h_SDir[partIdx];
		s2 = h_SDir[partIdx + 1];
		if((r2 - r1 > _maxPartLen) && (s2 - s1 > _maxPartLen))
			continue;	//skewed
		else if(r2 - r1 <= s2 - s1)	//put RPart in SM
		{
			for(int recIdx = r1; recIdx < r2 - 1; ++recIdx)
				assert(h_R[recIdx].y <= h_R[recIdx + 1].y);
		}
		else 	//put SPart in SM
		{
			for(int recIdx = s1; recIdx < s2 - 1; ++recIdx)
				assert(h_S[recIdx].y <= h_S[recIdx + 1].y);
		}
	}
	SAFE_FREE(h_R);
	SAFE_FREE(h_RDir);
	SAFE_FREE(h_S);
	SAFE_FREE(h_SDir);

#endif	

	//CUT_SAFE_CALL(cutStopTimer(timer));
	//printf("sort buckets: %f (ms)\n", cutGetTimerValue(timer));
}

int cuda_ProbeHJ(Record** d_RSout, Record* const d_R1, int* const d_RDir, const int nR, Record* const d_S1, int* const d_SDir, const int nS, const int PnG)
{
	unsigned int timer = 0;
	//CUT_SAFE_CALL(cutCreateTimer(&timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

	int Bn, Tn;
	ProbePlan(Bn, Tn);

	//Cnt
	dim3  DgJoinCnt(Bn, 1, 1);
	dim3  DbJoinCnt(Tn, 1, 1);	
	int* d_ThreadCnts;
	GPUMALLOC((void**)&d_ThreadCnts, Bn * Tn * sizeof(int));
	
	int Ns = _maxPartLen * sizeof(Record);
	int* d_skewPid;
	GPUMALLOC((void**)&d_skewPid, sizeof(int) * PnG);
	CUDA_SAFE_CALL(cudaMemset(d_skewPid, 0, sizeof(int) * PnG));

#ifdef SHARED_MEM
	Probe_Cnt<<<DgJoinCnt, DbJoinCnt, Ns>>>(d_ThreadCnts, d_skewPid, d_R1, d_RDir, d_S1, d_SDir, PnG);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#else
	Record* d_Table;
	GPUMALLOC( (void**)&d_Table, DgJoinCnt.x*Ns );
	Probe_Cnt_noShared<<<DgJoinCnt, DbJoinCnt>>>(d_Table, Ns, d_ThreadCnts, d_skewPid, d_R1, d_RDir, d_S1, d_SDir, PnG);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	GPUFREE(d_Table );
#endif
	CUT_CHECK_ERROR("Kernel failed");
	DebugInts(d_ThreadCnts, Bn * Tn);

	//CUT_SAFE_CALL(cutStopTimer(timer));
	//printf("probe cnt: %f (ms)\n", cutGetTimerValue(timer));
	//CUT_SAFE_CALL(cutResetTimer(timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

	//presum
    int* d_ThreadCntsPresum = NULL;
    GPUMALLOC( (void**) &d_ThreadCntsPresum, sizeof(int) * Bn * Tn);
    //saven_initialPrefixSum(Tn * Tn);
	//prescanArray(d_ThreadCntsPresum, d_ThreadCnts, 16);
	//prescanArray(d_ThreadCntsPresum, d_ThreadCnts, Bn * Tn);
	scanImpl(d_ThreadCnts, Bn * Tn,d_ThreadCntsPresum);
	//deallocBlockSums();	
	
	//nRS
	int h_SumButLast = 0, h_LastCnt = 0;
	CUDA_SAFE_CALL(cudaMemcpy(&h_SumButLast, d_ThreadCntsPresum + Bn * Tn - 1, sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy((void*)&h_LastCnt, d_ThreadCnts + Bn * Tn - 1, sizeof(int), cudaMemcpyDeviceToHost));
	int nRS = h_SumButLast + h_LastCnt;
	//printf("\nNon-overflow result part: %d\n", nRS);
	DebugInts(d_ThreadCntsPresum, Bn * Tn);

	//CUT_SAFE_CALL(cutStopTimer(timer));
	//printf("presum and get nRS: %f (ms)\n", cutGetTimerValue(timer));
	//CUT_SAFE_CALL(cutResetTimer(timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

	//write
	dim3  DgJoinWrite(Bn, 1, 1);
	dim3  DbJoinWrite(Tn, 1, 1);

	Record* d_RS = NULL;
	GPUMALLOC( (void**) &d_RS, sizeof(Record) * nRS);
#ifdef SHARED_MEM
	Probe_Write<<<DgJoinWrite, DbJoinWrite, Ns>>>(d_RS, d_ThreadCntsPresum, d_R1, d_RDir, d_S1, d_SDir, PnG);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
#else
	Record* dd_Table;
	GPUMALLOC( (void**)&dd_Table, DgJoinWrite.x*Ns );
	Probe_Write_noShared<<<DgJoinWrite, DbJoinWrite>>>(dd_Table, Ns, d_RS, d_ThreadCntsPresum, d_R1, d_RDir, d_S1, d_SDir, PnG);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	GPUFREE( dd_Table );
#endif
	CUT_CHECK_ERROR("Kernel failed");

	//CUT_SAFE_CALL(cutStopTimer(timer));
	//printf("probe write: %f (ms)\n", cutGetTimerValue(timer));
	//CUT_SAFE_CALL(cutResetTimer(timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

	//handle overflowed buckets
	//CUT_SAFE_CALL(cutResetTimer(timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));
	Record* d_RSOverflow = NULL;
	int nOverflowRS = 0;

	int* h_skewPid = (int*) malloc(sizeof(int) * PnG);
	CUDA_SAFE_CALL(cudaMemcpy((void*)h_skewPid, d_skewPid, sizeof(int) * PnG, cudaMemcpyDeviceToHost));
	int nSkewBuck = 0;
    //saven_initialPrefixSum(Bn * Tn);
	for(int i = 0; i < PnG; ++i)
	{
		if(h_skewPid[i] != 0)
		{
			//count
			printf("partition %d overflow.\t", i);
			++nSkewBuck;
			Probe_CntOverflow<<<DgJoinCnt, DbJoinCnt, Ns>>>(d_ThreadCnts, i, d_R1, d_RDir, d_S1, d_SDir);
			DebugInts(d_ThreadCnts, Bn * Tn);

			//prescanArray(d_ThreadCntsPresum, d_ThreadCnts, Bn * Tn);
			scanImpl(d_ThreadCnts, Bn * Tn,d_ThreadCntsPresum);
			h_SumButLast = 0;
			h_LastCnt = 0;
			CUDA_SAFE_CALL(cudaMemcpy(&h_SumButLast, d_ThreadCntsPresum + Bn * Tn - 1, sizeof(int), cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL(cudaMemcpy((void*)&h_LastCnt, d_ThreadCnts + Bn * Tn - 1, sizeof(int), cudaMemcpyDeviceToHost));
			nOverflowRS += h_SumButLast + h_LastCnt;

			//write overflow:
			GPUMALLOC( (void**) &d_RSOverflow, sizeof(Record) * nOverflowRS);
#ifdef SHARED_MEM
			Probe_WriteOverflow<<<DgJoinCnt, DbJoinCnt, Ns>>>(d_RSOverflow, d_ThreadCntsPresum, i, d_R1, d_RDir, d_S1, d_SDir);
			CUDA_SAFE_CALL(cudaThreadSynchronize());
#else
			Record* ddd_Table;
			GPUMALLOC( (void**)&ddd_Table, DgJoinCnt.x*Ns );
			Probe_WriteOverflow_noShared<<<DgJoinCnt, DbJoinCnt>>>(ddd_Table, Ns, d_RSOverflow, d_ThreadCntsPresum, i, d_R1, d_RDir, d_S1, d_SDir);
			GPUFREE( ddd_Table );
			CUDA_SAFE_CALL(cudaThreadSynchronize());
#endif
		}
	}
	SAFE_FREE(h_skewPid);
	//deallocBlockSums();	
	printf("Overflowed nRS: %d, num of overflowed partitions: %d\n", nOverflowRS, nSkewBuck);

	//copyout
	GPUMALLOC( (void**)d_RSout, sizeof(Record) * (nRS + nOverflowRS));	//pinnd

	//normal part
	CUDA_SAFE_CALL(cudaMemcpy(*d_RSout, d_RS, nRS * sizeof(Record), cudaMemcpyDeviceToDevice));
	//overflow part
	CUDA_SAFE_CALL(cudaMemcpy(*d_RSout + nRS, d_RSOverflow, nOverflowRS * sizeof(Record), cudaMemcpyDeviceToDevice));
    
	/*CUDA_SAFE_CALL( cudaMallocHost( (void**)ph_RS, sizeof(Record) * (nRS + nOverflowRS)));	//pinned

	//CUT_SAFE_CALL(cutResetTimer(timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));
	//normal part
	CUDA_SAFE_CALL(cudaMemcpy(*ph_RS, d_RS, nRS * sizeof(Record), cudaMemcpyDeviceToHost));
	//overflow part
	CUDA_SAFE_CALL(cudaMemcpy(*ph_RS + nRS, d_RSOverflow, nOverflowRS * sizeof(Record), cudaMemcpyDeviceToHost));*/

    //free all
	SAFE_CUDA_FREE(d_RS);
	SAFE_CUDA_FREE(d_RSOverflow);
	SAFE_CUDA_FREE((d_ThreadCnts));
	SAFE_CUDA_FREE((d_ThreadCntsPresum));
	SAFE_CUDA_FREE((d_skewPid));
	return (nRS + nOverflowRS);	
}

//Partitioning:
//	Histo: scan R and get d_PID and d_PidHisto of each thread
//		p=d_PID[r]=hash(r); histo[p]++; sync; d_PidHisto[snakewise p]=histo;
//	presum (d_PidHisto);
//	Reorder:  reorder R according to PID
//		Off[]=d_PidHisto; p=d_Pid[r]; d_R1[Off[p]++]=r;
//	S the same
//Probing:
//	Cnt: read R, read S, find cnt
//	Offet: presum to get the d_offset
//	Write: read R again, read S, write output
int cuda_hj(Record*& d_R, const int nR, Record*& d_S, const int nS, Record** d_RSout)
{
	unsigned int timer = 0;
	//CUT_SAFE_CALL(cutCreateTimer(&timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

	int PnG = _bDebug?	UpNearestPowOf2(	(int)ceilf((float)min(nR, nS) / _maxPartLen)		) :
						UpNearestPowOf2(	(int)ceilf((float)min(nR, nS) / _maxPartLen * 2)		);//tune
	
	int Pns[256] = {0};	//num of partitions in each recursive split
	int nPass = 0;
	RadixPlan(Pns, nPass, PnG);
	
	Record* d_R1 = NULL;	//table
	int* d_RDir = NULL;
	cuda_BuildHashTable(&d_R1, &d_RDir, d_R, nR, Pns, nPass);

	//CUT_SAFE_CALL(cutStopTimer(timer));
	//printf("build table for R: %f (ms)\n", cutGetTimerValue(timer));

	//CUT_SAFE_CALL(cutResetTimer(timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

	DebugRecs(d_R1, nR);
	DebugInts(d_RDir, PnG + 1);

	Record* d_S1 = NULL;	//table
	int* d_SDir = NULL;
	cuda_BuildHashTable(&d_S1, &d_SDir, d_S, nS, Pns, nPass);

	//CUT_SAFE_CALL(cutStopTimer(timer));
	// printf("build table for S: %f (ms)\n", cutGetTimerValue(timer));

	if(_bSortPart)
		cuda_ProbePreSort(d_R1, d_RDir, nR, d_S1, d_SDir, nS, PnG);

	//CUT_SAFE_CALL(cutResetTimer(timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

	int nRS = cuda_ProbeHJ(d_RSout, d_R1, d_RDir, nR, d_S1, d_SDir, nS, PnG);

	//CUT_SAFE_CALL(cutStopTimer(timer));
	//printf("probe: %f (ms)\n", cutGetTimerValue(timer));
	//CUT_SAFE_CALL(cutDeleteTimer(timer));

	SAFE_CUDA_FREE((d_R1));
	SAFE_CUDA_FREE((d_RDir));
	SAFE_CUDA_FREE((d_S1));
	SAFE_CUDA_FREE((d_SDir));

	return nRS;
}

extern "C"
int GPUOnly_hj( Record* d_Rin, int rLen, Record* d_Sin, int sLen, Record** d_Rout )
{
	return cuda_hj(d_Rin, rLen, d_Sin, sLen, d_Rout);
}

extern "C"
int GPUCopy_hj( Record* h_Rin, int rLen, Record* h_Sin, int sLen, Record** h_Rout )
{
	int rMemSize = sizeof(Record)*rLen;
	int sMemSize = sizeof(Record)*sLen;

	Record* d_Rin;
	Record* d_Sin;
	Record* d_Rout;

	GPUMALLOC( (void**)&d_Rin, rMemSize );
	GPUMALLOC( (void**)&d_Sin, sMemSize );
	TOGPU( d_Rin, h_Rin, rMemSize );
	TOGPU( d_Sin, h_Sin, sMemSize );

	int numResult = cuda_hj( d_Rin, rLen, d_Sin, sLen, &d_Rout);

	//*h_Rout = (Record*)malloc( sizeof(Record)*numResult );
	CPUMALLOC( (void**)h_Rout, sizeof(Record)*numResult );

	FROMGPU( *h_Rout, d_Rout, sizeof(Record)*numResult );

	GPUFREE( d_Rin );
	GPUFREE( d_Sin );
	GPUFREE( d_Rout );

	return numResult;
}


#endif
