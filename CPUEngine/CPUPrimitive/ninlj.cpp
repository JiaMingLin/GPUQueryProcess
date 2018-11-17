#include "stdafx.h"
#include "Joins.h"
#include "LinkedList.h"
#include "common.h"
#include "CPU_Dll.h"

/*
the blocked nested-loop join.
The blocking is on the relation R. Thus, we apply multithreading in access S. 
*/
void partNinlj(Record *R, int startR, int endR, Record *S, int startS, int endS, LinkedList *ll, int cpuid, int numThread)
{
	set_thread_affinity(cpuid,numThread);
	int numResult = 0;
	int i=0;
	int j=0;
	int k=0;
	int m=0;
	int startIndex=0;
	int endIndex=0;
	Record r;
	int data[2];
	for(j=startS;j<endS;j++)
	{
		data[1]=S[j].value;
		for(k=startR;k<endR;k++)
		{
			//if (R[k].value==S[j].value) 
			data[0]=R[k].value;
			if(PRED_EQUAL2(data))
			{	
				r.rid=R[k].rid;
				r.value=S[j].rid;
				ll->fill(&r);
				numResult++;				
			}
		}
	}
} 


int ninlj_omp(Record *R, int rLen, Record *S, int sLen, Record** Rout, int numThread)
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
	int numBlock=rLen/NLJ_BLOCK_SIZE;
	if (rLen%NLJ_BLOCK_SIZE!=0) 
		numBlock=numBlock+1;
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
	
	for(i=0;i<(numBlock-1);i++)
	{
		startR=i*NLJ_BLOCK_SIZE;
		endR=(i+1)*NLJ_BLOCK_SIZE;
		omp_set_num_threads(numThread);
		#pragma omp parallel for
		for(j=0;j<numThread;j++)
		{
			partNinlj(R, startR,endR,S, startS[j],endS[j],llList[j],j,numThread);
		}
	}
	//for the last block.
	startR=i*NLJ_BLOCK_SIZE;
	endR=rLen;
	omp_set_num_threads(numThread);
	#pragma omp parallel for
	for(j=0;j<numThread;j++)
	{
		partNinlj(R, startR,endR,S, startS[j],endS[j],llList[j],j,numThread);
	}
	
	numResult=dumpMLLtoArray(llList,numThread,Rout);
	freeMLL(llList,numThread);
	delete startS;
	delete endS;
	return numResult;
}
