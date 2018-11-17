#include "stdafx.h"
#include "Joins.h"
#include "LinkedList.h"
#include "common.h"
#include "Primitive.h"
#include "time.h"
#include "CPU_Dll.h"

int partMergejoin(Record *R, int startR, int endR, Record *S, int startS, int endS, LinkedList *ll, int cpuid, int numThread)
{
	set_thread_affinity(cpuid,numThread);
	//__DEBUG__("mergeJoin");
	int result=0;
	int pR=startR;
	int pS=startS;
	int pS2=startS;
	Record r;
	while(pR<endR && pS<endS)
	{
		if(R[pR].value==S[pS].value)
		{
			pS2=pS;
			r.rid=R[pR].rid;
			r.value=S[pS].rid;
			ll->fill(&r);
			result++;
			pS2++;
			while (pS2<endS) {
				if(R[pR].value==S[pS2].value)
				{
					r.rid=R[pR].rid;
					r.value=S[pS2].rid;
					ll->fill(&r);
					result++;
					pS2++;				
				}
				else
				break;
			}
			pR++;
		}
		else if(R[pR].value<S[pS].value)
			pR++;
		else
			pS++;
	}
	return result;
}


int smj_omp(Record *R, int rLen, Record *S, int sLen, Record** Joinout, int numThread)
{
	int result=0;
	//sort the two relation.
	clock_t timer=0;
	//startTimer(timer);
	Record *Rout=new Record[rLen];
	sort(R,rLen,compare,Rout,numThread);
	Record *Sout=new Record[sLen];
	sort(S,sLen,compare,Sout,numThread);
	//endTimer("sorting the two relations", timer);
	int *quanPosR=new int[numThread];
	int *quanPosS=new int[numThread];
	findQuantile(Rout,0,rLen,numThread,quanPosR);
	int i=0;
	for(i=0;i<numThread;i++)
		cout<<"R"<<i<<","<<quanPosR[i]<<"; ";
	cout<<endl;
	
	//cout<<"set the number of threads from "<<numThread;
	//numThread=1;
	//cout<<" to "<<numThread<<endl;
	for(i=0;i<numThread;i++)
		quanPosS[i]=findLargestSmaller(Sout,0,sLen,Rout[quanPosR[i]].value);
	for(i=0;i<numThread;i++)
		cout<<"S"<<i<<","<<quanPosS[i]<<"; ";
	cout<<endl;

	//use only the last few.
	int *startR=new int[numThread];
	int *endR=new int[numThread];
	int *startS=new int[numThread];
	int *endS=new int[numThread];

	for(i=0;i<(numThread-1);i++)
	{
		startR[i]=quanPosR[i];
		endR[i]=quanPosR[i+1];
		startS[i]=quanPosS[i];
		endS[i]=quanPosS[i+1];
	}
	startR[numThread-1]=quanPosR[numThread-1];
	startS[numThread-1]=quanPosS[numThread-1];
	endR[numThread-1]=rLen;
	endS[numThread-1]=sLen;
	for(i=0;i<numThread;i++)
		cout<<"pair,"<<i<<": "<<startR[i]<<","<<endR[i]<<", "<<startS[i]<<","<<endS[i]<<"\t";//<<"value: "<<Rout[startR[i]].value<<", "<<Sout[startS[i]].value<<endl;
	cout<<endl;
	startTimer(timer);
	LinkedList **llList=(LinkedList **)malloc(sizeof(LinkedList*)*numThread);
	initMLL(llList, numThread);
	omp_set_num_threads(numThread);
	#pragma omp parallel for
	for(int j=0;j<numThread;j++)
	{
		partMergejoin(Rout, startR[j],endR[j],Sout, startS[j],endS[j],llList[j],j,numThread);
	}
	endTimer("merge join", timer);
	result=dumpMLLtoArray(llList,numThread,Joinout);
	freeMLL(llList,numThread);


	delete startR;
	delete endR;
	delete startS;
	delete endS;
	delete quanPosR;
	delete Rout;
	delete Sout;
	return result;
}


int MergeJoinSortedRelation_omp(Record *Rout, int rLen, Record *Sout, int sLen, Record** Joinout, int numThread)
{
	int result=0;
	//sort the two relation.
	clock_t timer=0;
	//startTimer(timer);
	//endTimer("sorting the two relations", timer);
	int *quanPosR=new int[numThread];
	int *quanPosS=new int[numThread];
	findQuantile(Rout,0,rLen,numThread,quanPosR);
	int i=0;
	for(i=0;i<numThread;i++)
		cout<<"R"<<i<<","<<quanPosR[i]<<"; ";
	cout<<endl;
	
	//cout<<"set the number of threads from "<<numThread;
	//numThread=1;
	//cout<<" to "<<numThread<<endl;
	for(i=0;i<numThread;i++)
		quanPosS[i]=findLargestSmaller(Sout,0,sLen,Rout[quanPosR[i]].value);
	for(i=0;i<numThread;i++)
		cout<<"S"<<i<<","<<quanPosS[i]<<"; ";
	cout<<endl;

	//use only the last few.
	int *startR=new int[numThread];
	int *endR=new int[numThread];
	int *startS=new int[numThread];
	int *endS=new int[numThread];

	for(i=0;i<(numThread-1);i++)
	{
		startR[i]=quanPosR[i];
		endR[i]=quanPosR[i+1];
		startS[i]=quanPosS[i];
		endS[i]=quanPosS[i+1];
	}
	startR[numThread-1]=quanPosR[numThread-1];
	startS[numThread-1]=quanPosS[numThread-1];
	endR[numThread-1]=rLen;
	endS[numThread-1]=sLen;
	for(i=0;i<numThread;i++)
		cout<<"pair,"<<i<<": "<<startR[i]<<","<<endR[i]<<", "<<startS[i]<<","<<endS[i]<<"\t";//<<"value: "<<Rout[startR[i]].value<<", "<<Sout[startS[i]].value<<endl;
	cout<<endl;
	startTimer(timer);
	LinkedList **llList=(LinkedList **)malloc(sizeof(LinkedList*)*numThread);
	initMLL(llList, numThread);
	omp_set_num_threads(numThread);
	#pragma omp parallel for
	for(int j=0;j<numThread;j++)
	{
		partMergejoin(Rout, startR[j],endR[j],Sout, startS[j],endS[j],llList[j],j,numThread);
	}
	endTimer("merge join", timer);
	result=dumpMLLtoArray(llList,numThread,Joinout);
	freeMLL(llList,numThread);


	delete startR;
	delete endR;
	delete startS;
	delete endS;
	delete quanPosR;
	return result;
}