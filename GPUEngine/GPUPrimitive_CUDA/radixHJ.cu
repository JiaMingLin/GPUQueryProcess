
#include "stdio.h"
#include "stdlib.h"
#include "scan.cu"
#include "radixHJ_kernel.cu"

#include "cutil.h"

/////////////////////////////////////////////////////local func
#include "scan.cu"

#define SAFE_FREE(p) {if(p) {free(p); (p)=NULL;} };

int myLog2(int a)	//a should be 2^x. return x
{
	int x = 0;
	while (a = (a>>1))
		++x;
	return x;
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
//nPartR: num of parts in R
//output:
//nPasses: num of passes to go through
//Pns: Pn of each pass, should be 2^
void RadixPlan(int* Pns, int& nPass, int Pn)
{
	const int Pn0 = _bDebug? 2: 32; //SM 2k / tn 32. num of parts each pass could handle at most, due to SM size limitation
	nPass = 0;
	while(Pn > 1)
	{
		Pns[nPass ++] = min(Pn, Pn0);
		Pn /= Pn0;
	}
}

//partition and reorder d_R[nR] into d_R1, with Pn partitions starting from d_RDir.
void Partition(Record** d_R1, int* d_RDir, Record* d_R, const int nR, const int Bn, const int Tn, const int Pn, const int shift)
{
	//histoMat
	dim3  Dg(Bn, 1, 1);
	dim3  Db(Tn, 1, 1);
	int* d_HisM;	//histomat: bn*tn rows, Pn cols
	GPUMALLOC((void**) &d_HisM, sizeof(int) * Pn * Bn * Tn);
	int Ns = Tn * Pn * sizeof(int);	//each thread has a int[pn1]
	assert(Ns <= 4096); 
	if(Bn * Tn > nR)
		printf("Bn * Tn > nR, waste threads\n");
	Histo<<<Dg, Db, Ns>>> (d_HisM, d_R, nR, Pn, shift);
	CUT_CHECK_ERROR("Histo");
#if _bDebug
int* h_test = (int*)malloc( sizeof(int) * Bn * Tn * Pn);
CUDA_SAFE_CALL(cudaMemcpy(h_test, d_HisM, sizeof(int) * Bn * Tn * Pn, cudaMemcpyDeviceToHost));
#endif	
	//presum
	int* d_Loc = NULL;
    GPUMALLOC( (void**) &d_Loc, sizeof(int) * Pn * Bn * Tn);
	//saven_initialPrefixSum(Pn * Bn * Tn);
	//prescanArray(d_Loc, d_HisM, 16);
	//prescanArray(d_Loc, d_HisM, Pn * Bn * Tn);
	scanImpl(d_HisM, Pn * Bn * Tn, d_Loc);
	//deallocBlockSums();
	CUT_CHECK_ERROR("kernel error");
	CUDA_SAFE_CALL(cudaFree(d_HisM));
#if _bDebug
int* h_test2 = (int*)malloc( sizeof(int) * Bn * Tn * Pn);
CUDA_SAFE_CALL(cudaMemcpy(h_test2, d_Loc, sizeof(int) * Bn * Tn * Pn, cudaMemcpyDeviceToHost));
#endif
	//reorder
	GPUMALLOC((void**) d_R1, sizeof(Record) * nR);
	Reorder<<<Dg, Db, Ns>>> (*d_R1, d_RDir, d_R, d_Loc, nR, Pn, shift);
#if _bDebug
Record* h_test5 = (Record*)malloc(sizeof(Record) * nR);
CUDA_SAFE_CALL(cudaMemcpy(h_test5, *d_R1, sizeof(Record) * nR, cudaMemcpyDeviceToHost));

int* h_test3 = (int*)malloc( sizeof(int) * (Pn + 1));
CUDA_SAFE_CALL(cudaMemcpy(h_test3, d_RDir, sizeof(int) * (Pn + 1), cudaMemcpyDeviceToHost));
#endif
	CUDA_SAFE_CALL(cudaFree(d_Loc));

}

//partition unitNum units, d_R[RDir[pId]] into d_R1 and dRDir1[Pn].
void Partition2(Record* d_R1, int* d_RDir1, Record* d_R, const int nR, const int Bn, const int Tn, const int unitNum, const int Pn, const int shift, int* const d_RDir)
{
	dim3  Dg(Bn, 1, 1);
	dim3  Db(Tn, 1, 1);
	
	//histoMat
	int* d_HisM;	//histomat: bn*tn rows, Pn cols
	GPUMALLOC((void**) &d_HisM, sizeof(int) * Pn * Bn * Tn);
	int Ns = Tn * Pn * sizeof(int);	//each thread has a int[pn1]
	assert(Ns <= 4096); 
	if(Bn * Tn > nR)
		printf("Bn * Tn > nR, waste threads\n");

	//presum
	int* d_Loc = NULL;
	GPUMALLOC( (void**) &d_Loc, sizeof(int) * Pn * Bn * Tn);
	//saven_initialPrefixSum(Pn * Bn * Tn);
	//prescanArray(d_Loc, d_HisM, 16);

	for(int unitId = 0; unitId < unitNum; ++unitId)
	{
		//hisM
		Histo2<<<Dg, Db, Ns>>> (d_HisM, d_R, d_RDir, unitId, Pn, shift);
		CUT_CHECK_ERROR("kernel error");

#if _bDebug
int* h_test = (int*)malloc( sizeof(int) * Bn * Tn * Pn);
CUDA_SAFE_CALL(cudaMemcpy(h_test, d_HisM, sizeof(int) * Bn * Tn * Pn, cudaMemcpyDeviceToHost));
#endif	
		//presum
		//prescanArray(d_Loc, d_HisM, Pn * Bn * Tn);
		scanImpl(d_HisM, Pn * Bn * Tn, d_Loc);
		CUT_CHECK_ERROR("kernel error");

#if _bDebug
int* h_test2 = (int*)malloc( sizeof(int) * Bn * Tn * Pn);
CUDA_SAFE_CALL(cudaMemcpy(h_test2, d_Loc, sizeof(int) * Bn * Tn * Pn, cudaMemcpyDeviceToHost));
#endif
		//reorder
		Reorder2<<<Dg, Db, Ns>>> (d_R1, d_RDir1 + unitId * Pn, d_R, d_RDir, unitId, d_Loc, Pn, shift);
		CUT_CHECK_ERROR("kernel error");

#if _bDebug
Record* h_test5 = (Record*)malloc(sizeof(Record) * nR);
CUDA_SAFE_CALL(cudaMemcpy(h_test5, d_R1, sizeof(Record) * nR, cudaMemcpyDeviceToHost));

int* h_test3 = (int*)malloc( sizeof(int) * (unitNum * Pn + 1));
CUDA_SAFE_CALL(cudaMemcpy(h_test3, d_RDir1, sizeof(int) * (unitNum * Pn + 1), cudaMemcpyDeviceToHost));
#endif

#if _bDebug
	SAFE_FREE(h_test);
	SAFE_FREE(h_test2);
	SAFE_FREE(h_test3);
	SAFE_FREE(h_test5);
#endif 
	}
	//deallocBlockSums();
	CUT_CHECK_ERROR("kernel error");
	CUDA_SAFE_CALL(cudaFree(d_HisM));
	CUDA_SAFE_CALL(cudaFree(d_Loc));
}

//build hash table for h_R[nR]. store clustered&rearranged R in d_R1, each bucket's starting offset in it in d_RDir
void BuildHashTable_inner(Record** d_R1, int** d_RDir, Record* h_R, const int nR, int* const Pns, const int nPass)
{
	Record* d_R = NULL;
	unsigned int memsizeR = sizeof(Record) * nR;
	GPUMALLOC((void**) &d_R, memsizeR);
	CUDA_SAFE_CALL(cudaMemcpy(d_R, h_R, memsizeR, cudaMemcpyHostToDevice));
	const int Tn = 32;
	const int Bn = min(256, (int)ceilf(float(nR) / Tn) );	//tune
	
	//pass 1
	int* d_RDir1 = NULL;
	GPUMALLOC((void**) &d_RDir1, sizeof(int) * (Pns[0] + 1));
	Partition(d_R1, d_RDir1, d_R, nR, Bn, Tn, Pns[0], 0);
#if  _bDebug
Record* h_test6 = (Record*)malloc(sizeof(Record) * nR);
CUDA_SAFE_CALL(cudaMemcpy(h_test6, *d_R1, sizeof(Record) * nR, cudaMemcpyDeviceToHost));

int* h_test8 = (int*)malloc( sizeof(int) * (Pns[0] * Pns[1] + 1));
CUDA_SAFE_CALL(cudaMemcpy(h_test8, d_RDir1, sizeof(int) * (Pns[0] + 1), cudaMemcpyDeviceToHost));
#endif
	if(nPass == 1)
	{
		CUDA_SAFE_CALL(cudaFree(d_R));
		*d_RDir = d_RDir1;
		return;
	}

	//pass 2: further-partition each partition
	int* d_RDir2 = NULL;
	GPUMALLOC(void**) &d_RDir2, sizeof(int) * (Pns[0] * Pns[1] + 1));
	Partition2(d_R, d_RDir2, *d_R1, nR, Bn, Tn, Pns[0], Pns[1], myLog2(Pns[0]), d_RDir1);	//pingpong between d_R1 and d_R
#if  _bDebug
Record* h_test5 = (Record*)malloc(sizeof(Record) * nR);
CUDA_SAFE_CALL(cudaMemcpy(h_test5, d_R, sizeof(Record) * nR, cudaMemcpyDeviceToHost));

int* h_test3 = (int*)malloc( sizeof(int) * (Pns[0] * Pns[1] + 1));
CUDA_SAFE_CALL(cudaMemcpy(h_test3, d_RDir2, sizeof(int) * (Pns[0] * Pns[1] + 1), cudaMemcpyDeviceToHost));
#endif
	if(nPass == 2)
	{
		CUDA_SAFE_CALL(cudaFree(*d_R1));
		*d_R1 = d_R;
		CUDA_SAFE_CALL(cudaFree(d_RDir1));
		*d_RDir = d_RDir2;
		return;
	}

	//pass3: further-partition each partition
	int* d_RDir3 = NULL;
	GPUMALLOC((void**) &d_RDir3, sizeof(int) * (Pns[0] * Pns[1] * Pns[2] + 1));
	Partition2(*d_R1, d_RDir3, d_R, nR, Bn, Tn, Pns[0] * Pns[1], Pns[2], myLog2(Pns[0] * Pns[1]), d_RDir2);	//pingpong between d_R1 and d_R

#if _bDebug
Record* h_test2 = (Record*)malloc(sizeof(Record) * nR);
CUDA_SAFE_CALL(cudaMemcpy(h_test2, *d_R1, sizeof(Record) * nR, cudaMemcpyDeviceToHost));

int* h_test4 = (int*)malloc( sizeof(int) * (Pns[0] * Pns[1] * Pns[2] + 1));
CUDA_SAFE_CALL(cudaMemcpy(h_test4, d_RDir3, sizeof(int) * (Pns[0] * Pns[1] * Pns[2] + 1), cudaMemcpyDeviceToHost));
#endif
#if _bDebug
	SAFE_FREE(h_test2);
	SAFE_FREE(h_test3);
	SAFE_FREE(h_test4);
	SAFE_FREE(h_test5);
	SAFE_FREE(h_test6);
	SAFE_FREE(h_test8);
#endif 

	CUDA_SAFE_CALL(cudaFree(d_R));
	CUDA_SAFE_CALL(cudaFree(d_RDir1));
	CUDA_SAFE_CALL(cudaFree(d_RDir2));
	*d_RDir = d_RDir3;
}

int ProbeHJ(Record** h_RS, Record* const d_R1, int* const d_RDir, const int nR, Record* const d_S1, int* const d_SDir, const int nS, const int Pn)
{
	const int Bn = 64;	//tune
	const int Tn = 256;

	//Cnt
	dim3  DgJoinCnt(Bn, 1, 1);
	dim3  DbJoinCnt(Tn, 1, 1);	
	int* d_ThreadCnts;
	GPUMALLOC((void**)&d_ThreadCnts, Bn * Tn * sizeof(int));
	
	int Ns = _JoinPartMax * sizeof(Record);
	int* d_skewPid;
	GPUMALLOC((void**)&d_skewPid, sizeof(int) * Pn);
	CUDA_SAFE_CALL(cudaMemset(d_skewPid, 0, sizeof(int) * Pn));

	Joining_Cnt<<<DgJoinCnt, DbJoinCnt, Ns>>>(d_ThreadCnts, d_skewPid, d_R1, d_RDir, d_S1, d_SDir, Pn);
	
	CUT_CHECK_ERROR("Kernel failed");
#if _bDebug
int* h_cnts = (int*) malloc(sizeof(int) * Bn * Tn);
CUDA_SAFE_CALL(cudaMemcpy(h_cnts, d_ThreadCnts, Bn * Tn * sizeof(int), cudaMemcpyDeviceToHost));
#endif

	//presum
    int* d_ThreadCntsPresum = NULL;
    GPUMALLOC( (void**) &d_ThreadCntsPresum, sizeof(int) * Bn * Tn);
    //saven_initialPrefixSum(Tn * Tn);
	//prescanArray(d_ThreadCntsPresum, d_ThreadCnts, 16);
	//prescanArray(d_ThreadCntsPresum, d_ThreadCnts, Bn * Tn);
	scanImpl(d_ThreadCnts, Bn * Tn, d_ThreadCntsPresum);
	//deallocBlockSums();	
	
	//nRS
	int h_SumButLast = 0, h_LastCnt = 0;
	CUDA_SAFE_CALL(cudaMemcpy(&h_SumButLast, d_ThreadCntsPresum + Bn * Tn - 1, sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy((void*)&h_LastCnt, d_ThreadCnts + Bn * Tn - 1, sizeof(int), cudaMemcpyDeviceToHost));
	int nRS = h_SumButLast + h_LastCnt;
	//printf("\nNon-overflow result part: %d\n", nRS);
#if _bDebug
int* h_ThreadCntsPresum = (int*)malloc(sizeof(int) * Bn * Tn);
CUDA_SAFE_CALL(cudaMemcpy(h_ThreadCntsPresum, d_ThreadCntsPresum, sizeof(int) * Bn * Tn, cudaMemcpyDeviceToHost));
#endif

	//write
	dim3  DgJoinWrite(Bn, 1, 1);
	dim3  DbJoinWrite(Tn, 1, 1);
	Record* d_RS;
	GPUMALLOC( (void**) &d_RS, sizeof(Record) * nRS);
	Join_Write<<<DgJoinWrite, DbJoinWrite, Ns>>>(d_RS, d_ThreadCntsPresum, d_R1, d_RDir, d_S1, d_SDir, Pn);
	CUT_CHECK_ERROR("Kernel failed");
	//printf("Join_Write: \t%f (ms)\n", cutGetTimerValue(timer));

	//handle overflowed buckets
	LinkedList *ll=(LinkedList*)malloc(sizeof(LinkedList)); 
	ll->init();	
	Record tmpRS;
	int nSkew = 0;
	int* h_skewPid = (int*) malloc(sizeof(int) * Pn);
	CUDA_SAFE_CALL(cudaMemcpy((void*)h_skewPid, d_skewPid, sizeof(int) * Pn, cudaMemcpyDeviceToHost));
	//saven_initialPrefixSum(Tn * Tn);
	int nSkewBuck = 0;
	for(int i = 0; i < Pn; ++i)
	{
		if(h_skewPid[i] != 0)
		{
			printf("overflow, write not handled yet. \t");
			++nSkewBuck;
			Joining_CntOverflow<<<DgJoinCnt, DbJoinCnt, Ns>>>(d_ThreadCnts, h_skewPid[i], d_R1, d_RDir, d_S1, d_SDir);
			
//int* h_cnts = (int*) malloc(sizeof(int) * Bn * Tn);
//CUDA_SAFE_CALL(cudaMemcpy(h_cnts, d_ThreadCnts, Bn * Tn * sizeof(int), cudaMemcpyDeviceToHost));
		
			//prescanArray(d_ThreadCntsPresum, d_ThreadCnts, Bn * Tn);
			scanImpl(d_ThreadCnts, Bn * Tn, d_ThreadCntsPresum);
			h_SumButLast = 0;
			h_LastCnt = 0;
			CUDA_SAFE_CALL(cudaMemcpy(&h_SumButLast, d_ThreadCntsPresum + Bn * Tn - 1, sizeof(int), cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL(cudaMemcpy((void*)&h_LastCnt, d_ThreadCnts + Bn * Tn - 1, sizeof(int), cudaMemcpyDeviceToHost));
			nSkew += h_SumButLast + h_LastCnt;

			//write overflow:
			//..todo. currently assume uniformly randomized tuples and no severely overflow (skewRatio < 2).
		}
	}
	SAFE_FREE(h_skewPid);
	//deallocBlockSums();	
	//printf("Overflow result part: %d, num of skewed bucket: %d\n", nSkew, nSkewBuck);

	*h_RS = (Record*)malloc( sizeof(Record) * (nRS + nSkew));
	CUDA_SAFE_CALL(cudaMemcpy(*h_RS, d_RS, sizeof(Record) * nRS, cudaMemcpyDeviceToHost));
	
	//copy h_RS <- ll
	ll->copyToArray((*h_RS) + nRS);
	ll->destroy();
	SAFE_FREE(ll);

    //free all
	CUDA_SAFE_CALL(cudaFree(d_RS));
	CUDA_SAFE_CALL(cudaFree(d_ThreadCnts));
	CUDA_SAFE_CALL(cudaFree(d_ThreadCntsPresum));
	CUDA_SAFE_CALL(cudaFree(d_skewPid));
	CUDA_SAFE_CALL(cudaFree(d_RDir));
	CUDA_SAFE_CALL(cudaFree(d_SDir));
	return (nRS + nSkew);	
}


///////////////////////////////////////RadixClusteredHashJoin
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
int RadixClusteredHashJoin(Record* h_R, const int nR, Record* h_S, const int nS, PREDICATE_NODE * pRoot, Record** h_RS)
{
	int PnTmp = (int)ceilf((float)min(nR, nS) / _JoinPartMax * 2);	//tune
	int Pn = UpNearestPowOf2(PnTmp);
	int Pns[256] = {0};	//Pn in at most 3 passes
	int nPass = 0;
	RadixPlan(Pns, nPass, Pn);

	unsigned int timer = 0;
	//CUT_SAFE_CALL(cutCreateTimer(&timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

	Record* d_R1 = NULL;	//table
	int* d_RDir = NULL;
	BuildHashTable_inner(&d_R1, &d_RDir, h_R, nR, Pns, nPass);

	//CUT_SAFE_CALL(cutStopTimer(timer));
	//printf("build table for R: %f (ms)\n", cutGetTimerValue(timer));
	//CUT_SAFE_CALL(cutResetTimer(timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

#if _bDebug
Record* h_test5 = (Record*)malloc(sizeof(Record) * nR);
CUDA_SAFE_CALL(cudaMemcpy(h_test5, d_R1, sizeof(Record) * nR, cudaMemcpyDeviceToHost));

int* h_test3 = (int*)malloc( sizeof(int) * (Pn + 1));
CUDA_SAFE_CALL(cudaMemcpy(h_test3, d_RDir, sizeof(int) * (Pn + 1), cudaMemcpyDeviceToHost));
#endif

	Record* d_S1 = NULL;	//table
	int* d_SDir = NULL;
	BuildHashTable_inner(&d_S1, &d_SDir, h_S, nS, Pns, nPass);

	//CUT_SAFE_CALL(cutStopTimer(timer));
	//printf("build table for S: %f (ms)\n", cutGetTimerValue(timer));
	//CUT_SAFE_CALL(cutResetTimer(timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

	int nRS = ProbeHJ(h_RS, d_R1, d_RDir, nR, d_S1, d_SDir, nS, Pn);
	
	//CUT_SAFE_CALL(cutStopTimer(timer));
	//printf("probe: %f (ms)\n", cutGetTimerValue(timer));

	CUDA_SAFE_CALL(cudaFree(d_R1));
	CUDA_SAFE_CALL(cudaFree(d_RDir));
	CUDA_SAFE_CALL(cudaFree(d_S1));
	CUDA_SAFE_CALL(cudaFree(d_SDir));
	//CUT_SAFE_CALL(cutDeleteTimer(timer));
	return nRS;
}

int cuda_hj_inner(Record *h_R, int nR, Record *h_S, int nS, PREDICATE_NODE * pRoot, Record** Rout)
{
	unsigned int timer = 0;
	//CUT_SAFE_CALL(cutCreateTimer(&timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));

	//radix-clustered hash join
 	int nRS = RadixClusteredHashJoin(h_R, nR, h_S, nS, pRoot, Rout);

	//CUT_SAFE_CALL(cutStopTimer(timer));
	//printf("\nCUDA nRS: %d, total processing: %f (ms)\n", nRS, cutGetTimerValue(timer));
	//CUT_SAFE_CALL(cutDeleteTimer(timer));
	return nRS;
}
