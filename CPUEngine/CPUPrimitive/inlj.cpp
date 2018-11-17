#include "stdafx.h"
#include "Joins.h"
#include "LinkedList.h"
#include "common.h"
#include "CPU_Dll.h"


//the indexed relation is R!!
int partInlj(Record *R, int rLen, CC_CSSTree *tree, Record *S, int startS, int endS, LinkedList *ll, int cpuid,int numThread)
{
	set_thread_affinity(cpuid,numThread);
	int result=0;
	int i=0;
	int k=0;
	int curIndex=0;
	int keyForSearch;
	Record r;
	int data[2];
	for(k=startS; k<endS; k++)
	{
		keyForSearch=S[k].value;
		curIndex=tree->search(keyForSearch);
		data[1]=keyForSearch;
		for(i=curIndex-1;i>0;i--)
		{
			data[0]=R[i].value;
			//if(R[i].value==keyForSearch)
			if(PRED_EQUAL2(data))
			{
				r.rid=R[i].rid;
				r.value=S[k].rid;
				ll->fill(&r);
				result++;
			}
			else
				if(R[i].value<keyForSearch)
				break;
		}
		for(i=curIndex;i<rLen;i++)
		{	
			data[0]=R[i].value;
			//if(R[i].value==keyForSearch)
			if(PRED_EQUAL2(data))
			{
				r.rid=R[i].rid;
				r.value=S[k].rid;
				ll->fill(&r);
				result++;
			}
			else
				if(R[i].value>keyForSearch)
				break;
		}
	}
	return result;
}


int inlj_omp(Record *R, int rLen, CC_CSSTree *tree, Record *S, int sLen, Record** Rout, int numThread)
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
		partInlj(R, rLen, tree,S, startS[j],endS[j],llList[j],i,numThread);
	}
	
	numResult=dumpMLLtoArray(llList,numThread,Rout);
	freeMLL(llList,numThread);
	delete startS;
	delete endS;
	return numResult;
}