#include "stdafx.h"
#include "hashTable.h"
#include "LinkedList.h"
#include "common.h"
#include "CPU_Dll.h"

#define HASH(VALUE, RLEN) ( ((VALUE)>>1) &( ((RLEN)>>1)-1))
void buildHashTable(Record* h_R, int rLen, int intBits, Bound *h_bound)
{
	//qsort(h_R, rLen, sizeof(int2), compareInt2);
	int curL=0;
	int start=0;
	int end=0;
	int curValue=0;
	//int index=0;
	for(curL=0;curL<rLen;curL++)
	{
		start=curL;
		curValue=HASH(h_R[curL].value,rLen);
		curL++;
		end=curL;
		for(;curL<rLen;curL++)
		if(HASH(h_R[curL].value,rLen)!=curValue)//one row.
		{
			end=curL;
			curL--;
			break;
		}
		h_bound[curValue].start=start;
		h_bound[curValue].end=end;
		/*printf("curValue, %d, [%d, %d], ",curValue, start, end);
		index++;
		if(index%10==9) printf("\n");*/
	}		
}

int partHashSearch(Record* h_R, int rLen, Bound *h_bound, int intBits, Record *S, int startS, int endS, LinkedList *ll, int cpuid, int numThread)
{
	set_thread_affinity(cpuid,numThread);
	int i=0;
	int curBucket=-1;
	int start=-1, end=-1;
	int j=0;
	int result=0;
	Record r;
	for(i=startS;i<endS;i++)
	{
		curBucket=HASH(S[i].value,rLen);
		start=h_bound[curBucket].start;
		end=h_bound[curBucket].end;
		for(j=start;j<end;j++)
		{
			if(h_R[j].value==S[i].value)
			{
				r.rid=h_R[j].rid;
				r.value=S[i].rid;
				ll->fill(&r);
				result++;
			}
		}
	}
	return result;
}


int HashSearch_omp(Record* R, int rLen, Bound *h_bound, int intBits, Record *S, int sLen, Record** Rout, int numThread)
{
	int numResult = 0;
	int i=0;
	int j=0;
	int k=0;
	int m=0;
	int startR=0;
	int endR=0;
	LinkedList **llList=(LinkedList **)malloc(sizeof(LinkedList*)*numThread);
	initMLL(llList, numThread);
	int *startS=new int[numThread];
	int *endS=new int[numThread];
	int chunkSize=sLen/numThread;
	for(i=0;i<numThread;i++)
	{
		startS[i]=i*chunkSize;
		if(i==(numThread-1))
			endS[i]=sLen;
		else
			endS[i]=(i+1)*chunkSize;
		cout<<"T"<<i<<", "<<endS[i]-startS[i]<<"; ";
	}
	cout<<endl;
	//omp_set_num_threads(numThread);
	#pragma omp parallel for
	for(j=0;j<numThread;j++)
	{
		partHashSearch(R, rLen, h_bound,intBits,S, startS[j],endS[j],llList[j],j,numThread);
	}
	
	numResult=dumpMLLtoArray(llList,numThread,Rout);
	freeMLL(llList,numThread);
	delete startS;
	delete endS;
	return numResult;
}

