#include "stdafx.h"
#include "CPU_dll.h"
#include "common.h"
#include "Joins.h"

//selection
int CPU_PointSelection(Record* Rin, int rLen, int matchingKeyValue, Record **Rout, int numThread)
{
	return filter_openmp(Rin,rLen,matchingKeyValue,matchingKeyValue,Rout,numThread);
}

int CPU_RangeSelection(Record* Rin, int rLen, int rangeSmallKey, int rangeLargeKey, Record **Rout,int numThread)
{
	return filter_openmp(Rin,rLen,rangeSmallKey,rangeLargeKey,Rout,numThread);
}

//constructing hash tables, or trees.
void CPU_BuildHashTable(Record* h_R, int rLen, int intBits, Bound *h_bound)
{
	buildHashTable(h_R,rLen,intBits,h_bound);
}

void CPU_BuildTreeIndex(Record* R, int rLen, CC_CSSTree** tree)
{
	*tree=new CC_CSSTree(R, rLen, CSS_TREE_FANOUT);
}

//access methods
int CPU_HashSearch(Record* R, int rLen, Bound *h_bound, int intBits, Record *S, int sLen, Record** Rout, int numThread)
{
	return HashSearch_omp(R, rLen,h_bound,intBits,S, sLen,Rout,numThread);
}

int CPU_TreeSearch(Record *R, int rLen, CC_CSSTree *tree, Record *S, int sLen, Record** Rout, int numThread)
{
	return inlj_omp(R,rLen,tree, S, sLen,Rout,numThread);
}

//aggregation
int CPU_AggAvg(Record *R, int rLen, int numThread)
{
	int sum= reduce_openmp(R,rLen,numThread,REDUCE_SUM);
	return (sum/rLen);
}

int CPU_AggSum(Record *R, int rLen, int numThread)
{
	return reduce_openmp(R,rLen,numThread,REDUCE_SUM);
}

int CPU_AggMax(Record *R, int rLen, int numThread)
{
	return reduce_openmp(R,rLen,numThread,REDUCE_MAX);
}

int CPU_AggMin(Record *R, int rLen, int numThread)
{
	return reduce_openmp(R,rLen,numThread,REDUCE_MIN);
}

//group by
void partGroupBy(Record *R, int startS, int endS, LinkedList *ll, int cpuid, int numThread)
{
	set_thread_affinity(cpuid,numThread);
	int i=0;
	Record r;
	for(i=startS;i<endS;i++)
	{
		if(i==0)
		{
			r.value=0;
			ll->fill(&r);
		}
		else
		{
			if(R[i].value!=R[i-1].value)
			{
				r.value=i;
				ll->fill(&r);
			}
		}			
	}
}

int CPU_GroupBy(Record*R, int rLen, Record* Rout, int** d_startPos, int numThread)
{
	sort(R,rLen,compare,Rout,numThread);
	int* start=new int[numThread];
	int* end=new int[numThread];
	int i=0;
	int chunkSize=rLen/numThread;
	for(i=0;i<numThread;i++)
	{
		start[i]=i*chunkSize;
		if(i==(numThread-1))
			end[i]=rLen;
		else
			end[i]=(i+1)*chunkSize;
	}
	LinkedList **llList=(LinkedList **)malloc(sizeof(LinkedList*)*numThread);
	initMLL(llList, numThread);
	//omp_set_num_threads(numThread);
	#pragma omp parallel for
	for(i=0;i<numThread;i++)
	{
		partGroupBy(Rout, start[i],end[i],llList[i],i, numThread);
	}
	Record **RTempOut=(Record**)malloc(sizeof(Record*));
	int numResult=dumpMLLtoArray(llList,numThread,RTempOut);
	freeMLL(llList,numThread);
	*d_startPos=new int[numResult];
	for(i=0;i<numResult;i++)
		(*d_startPos)[i]=(*RTempOut)[i].value;
	free(*RTempOut);
	free(RTempOut);
	delete start;
	delete end;
	return numResult;
}


/*
aggregation after group by.
with the known number of groups, we can allocate the output for advance: d_aggResults.
*/
void CPU_agg_max_afterGroupBy(Record *Rin, int rLen, int* d_startPos, int numGroups, 
							  Record * RinAggOrig, int* d_aggResults, int numThread)
{
	int i=0;
	int *tempLoc=(int*)malloc(sizeof(int)*rLen);
	for(i=0;i<rLen;i++)
	{
		tempLoc[i]=Rin[i].rid;
	}
	Record* RAggTemp=new Record[rLen];
	gather(RinAggOrig,tempLoc,RAggTemp,rLen,numThread);
//	#pragma omp parallel for
	for(i=0;i<numGroups;i++)
	{
		int start=d_startPos[i];
		int end=0;
		if((i+1)==numGroups)
			end=rLen;
		else
			end=d_startPos[i+1];
		int j=start;
		int curMax=RAggTemp[j].value;
		for(j=start+1;j<end;j++)
			if(curMax<RAggTemp[j].value)
				curMax=RAggTemp[j].value;
		d_aggResults[i]=curMax;
	}



	delete RAggTemp;
	free(tempLoc);
}

void CPU_agg_min_afterGroupBy(Record *Rin, int rLen, int* d_startPos, int numGroups, 
							  Record * RinAggOrig, int* d_aggResults, int numThread)
{
	int i=0;
	int *tempLoc=(int*)malloc(sizeof(int)*rLen);
	for(i=0;i<rLen;i++)
	{
		tempLoc[i]=Rin[i].rid;
	}
	Record* RAggTemp=new Record[rLen];
	gather(RinAggOrig,tempLoc,RAggTemp,rLen,numThread);
	//#pragma omp parallel for
	for(i=0;i<numGroups;i++)
	{
		int start=d_startPos[i];
		int end=0;
		if((i+1)==numGroups)
			end=rLen;
		else
			end=d_startPos[i+1];
		int j=start;
		int curMin=RAggTemp[j].value;
		for(j=start+1;j<end;j++)
			if(curMin>RAggTemp[j].value)
				curMin=RAggTemp[j].value;
		d_aggResults[i]=curMin;
	}
	delete RAggTemp;
	free(tempLoc);
}


void CPU_agg_sum_afterGroupBy(Record *Rin, int rLen, int* d_startPos, int numGroups, 
							  Record * RinAggOrig, int* d_aggResults, int numThread)
{
	int i=0;
	int *tempLoc=(int*)malloc(sizeof(int)*rLen);
	for(i=0;i<rLen;i++)
	{
		tempLoc[i]=Rin[i].rid;
	}
	Record* RAggTemp=new Record[rLen];
	gather(RinAggOrig,tempLoc,RAggTemp,rLen,numThread);
	//#pragma omp parallel for
	for(i=0;i<numGroups;i++)
	{
		int start=d_startPos[i];
		int end=0;
		if((i+1)==numGroups)
			end=rLen;
		else
			end=d_startPos[i+1];
		int j=start;
		int curSum=RAggTemp[j].value;
		for(j=start+1;j<end;j++)
			curSum+=RAggTemp[j].value;
		d_aggResults[i]=curSum;
	}
	delete RAggTemp;
	free(tempLoc);
}


void CPU_agg_avg_afterGroupBy(Record *Rin, int rLen, int* d_startPos, int numGroups, 
							  Record * RinAggOrig, int* d_aggResults, int numThread)
{
	int i=0;
	int *tempLoc=(int*)malloc(sizeof(int)*rLen);
	for(i=0;i<rLen;i++)
	{
		tempLoc[i]=Rin[i].rid;
	}
	Record* RAggTemp=new Record[rLen];
	gather(RinAggOrig,tempLoc,RAggTemp,rLen,numThread);
	//#pragma omp parallel for
	for(i=0;i<numGroups;i++)
	{
		int start=d_startPos[i];
		int end=0;
		if((i+1)==numGroups)
			end=rLen;
		else
			end=d_startPos[i+1];
		int j=start;
		int curSum=RAggTemp[j].value;
		for(j=start+1;j<end;j++)
			curSum+=RAggTemp[j].value;
		d_aggResults[i]=curSum/(end-start);
	}
	delete RAggTemp;
	free(tempLoc);
}

//sort

void CPU_Sort(Record* Rin, int rLen, Record* Rout, int numThread)
{
	sort(Rin,rLen,compare,Rout,numThread);
}

//join

int CPU_ninlj(Record *R, int rLen, Record *S, int sLen, Record** Rout, int numThread)
{
	return ninlj_omp(R,rLen,S,sLen,Rout,numThread);
}
int CPU_inlj(Record *R, int rLen, CC_CSSTree *tree, Record *S, int sLen, 
			 Record** Rout, int numThread)
{
	return inlj_omp(R,rLen,tree,S,sLen,Rout,numThread);
}
int CPU_smj(Record *R, int rLen, Record *S, int sLen, Record** Rout, int numThread)
{
	return smj_omp(R,rLen,S,sLen,Rout,numThread);
}

int CPU_hj(Record *R, int rLen, Record *S, int sLen, Record** Rout, int numThread)
{
	return hj_omp(R,rLen,S,sLen,Rout,numThread);
}


//for testing only
int CPU_Sum(int a, int b)
{
	return (a+b);
}
//testing GroupBY
void testGroupBy(int rLen,  int numThread)
{
	int result=0;
	int numGroup=64;
	Record *R=new Record[rLen];
	generateRand(R, numGroup,rLen,0);	
	Record *RinAggOrig=new Record[rLen];
	generateRand(RinAggOrig, numGroup,rLen,1);	
	Record *Rout=(Record*)malloc(sizeof(Record)*rLen);
	int **d_startPos=(int**)malloc(sizeof(int*));
	*d_startPos=NULL;
	startTime();
	int resultGroups=CPU_GroupBy(R,rLen,Rout,d_startPos,numThread);
	double sec=endTime("group by");
	int* d_aggResults=new int[resultGroups];
	CPU_agg_max_afterGroupBy(Rout,rLen,*d_startPos,resultGroups,RinAggOrig,d_aggResults,numThread);
	cout<<", rLen, "<<rLen<<", numThreads, "<<numThread<<", sum, "<<result<<", "<<endl;
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	cout<<"\nGroup by bandwidth, "<< bandwidth<<" MB/sec"<<endl;
	delete R;
	delete RinAggOrig;
	if(*d_startPos!=NULL)
		free(*d_startPos);
	free(Rout);
	delete d_aggResults;
}
// RS Hash Function
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
void getRSHashID(void *Rin, void* para, void *Rout)
{
	Record *r=(Record*)Rin;
	int *o=(int*)Rout;
	int *mask=((int*)para);
	*o=RSHash(r->value,*mask);
}

void CPU_Partition(Record *R, int rLen, int numPart, Record* Rout, int* startHist, int numThread)
{
	int mask=numPart-1;
	split(R,rLen,mask+1,Rout,startHist,getRSHashID,(void*)(&mask),numThread);
}

int CPU_MergeJoin(Record *R, int rLen, Record *S, int sLen, Record** Rout, int numThread)
{
	return MergeJoinSortedRelation_omp(R,rLen,S,sLen,Rout,numThread);
}

void testDLL(int argc, char **argv)
{
	int i=0;
	for(i=0;i<argc;i++)
	{
		if(strcmp(argv[i],"-GroupBY")==0)
		{
			int rLen=8*1024*1024;
			int numThread=4;
			if(argc==(i+3))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				numThread=atoi(argv[i+2]);
			}
			testGroupBy(rLen,numThread);
		}
	}
}

void CPU_Projection(Record* baseTable, int rLen, Record* projTable, int pLen, int numThread)
{
	int* loc=new int[pLen];
	int i=0;
	for(i=0;i<pLen;i++)
	{
		loc[i]=projTable[i].rid;
	}
	gather(baseTable,loc,projTable,pLen,numThread);
	delete loc;
}