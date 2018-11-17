#include "stdafx.h"
#include "Joins.h"
#include "LinkedList.h"
#include "common.h"
#include "Primitive.h"
#include "CPU_Dll.h"

struct radixBit{
	int mask;
	int moveRightBits;
};

void getRadixPartID(void *Rin, void* para, void *Rout)
{
	Record *r=(Record*)Rin;
	int *o=(int*)Rout;
	radixBit *rb=((radixBit*)para);
	int mask=rb->mask;
	int moveRightBits=rb->moveRightBits;
	*o=(r->value>>moveRightBits)&mask;
}

void radixPart(Record *R, int rLen, int moveRightBits, int mask, Record *Rout, int* startHist, int numThread)
{
	radixBit *rb=new radixBit();
	rb->mask=mask;
	rb->moveRightBits=moveRightBits;
	split(R,rLen,mask+1,Rout,startHist,getRadixPartID,(void*)rb,numThread);
	delete rb;
}

void radixJoin(Record* R, int *startHistR, int* endHistR, 
			  Record* S, int *startHistS, int* endHistS, int start, int end,
			  LinkedList* ll, int cpuid, int numThread)
{
	set_thread_affinity(cpuid,numThread);
	int i=0;
	for(i=start;i<end;i++)
	{
		partNinlj(R,startHistR[i],endHistR[i],S,startHistS[i],endHistS[i],ll,i,1);//we do not need further parallel
	}
}


void radixPartJoin(Record* R, int *g_startHistR, int* g_endHistR, 
			  Record* S, int *g_startHistS, int* g_endHistS, int start, int end,
			  int moveRightBits, int mask, LinkedList* ll, int cpuid, int numThread)
{
	set_thread_affinity(cpuid,numThread);
	int i=0;
	int numPart=mask+1;
	int* l_startHistR=new int[numPart];
	int* l_endHistR=new int[numPart];
	int* l_startHistS=new int[numPart];
	int* l_endHistS=new int[numPart];
	int j=0;
	for(i=start;i<end;i++)
	{
		//partNinlj(R,startHistR[i],endHistR[i],S,startHistS[i],endHistS[i],ll);
		int rLen=g_endHistR[i]-g_startHistR[i];
		Record *Rin=R+g_startHistR[i];
		Record *Rout=new Record[rLen];
		radixPart(Rin,rLen,moveRightBits,mask,Rout,l_startHistR,1);

		int sLen=g_endHistS[i]-g_startHistS[i];
		Record *Sin=S+g_startHistS[i];
		Record *Sout=new Record[sLen];
		radixPart(Sin,sLen,moveRightBits,mask,Sout,l_startHistS,1);
		for(j=0;j<numPart-1;j++)
		{
			l_endHistR[j]=l_startHistR[j+1];
			l_endHistS[j]=l_startHistS[j+1];
		}
		l_endHistR[j]=rLen;
		l_endHistS[j]=sLen;
		for(j=0;j<numPart;j++)
			partNinlj(Rout,l_startHistR[j],l_endHistR[j],Sout,l_startHistS[j], l_endHistS[j],ll,i,1);
		delete Rout;
		delete Sout;
	}
	delete l_startHistR;
	delete l_endHistR;
	delete l_startHistS;
	delete l_endHistS;
}


int hj_omp(Record *R, int rLen, Record *S, int sLen, Record** Joinout, int numThread)
{
	int result=0;
	int totalBitsUsed=log2Ceil(rLen)-BASE_BITS;
	int totalNumPass=0;
	if(totalBitsUsed>=10)
	{
		totalNumPass=2;
	}
	else
	{
		totalNumPass=1;
	}
	int bitPerPass=totalBitsUsed/totalNumPass;
	int *bitUsedPerPass=new int[totalNumPass];
	int curPass=0;
	for(curPass=0;curPass<totalNumPass-1;curPass++)
		bitUsedPerPass[curPass]=bitPerPass;
	//the last pass
	bitUsedPerPass[curPass]=totalBitsUsed-bitPerPass*(totalNumPass-1);
	for(curPass=0;curPass<totalNumPass;curPass++)
		printf("P%d, %d; ", curPass, bitUsedPerPass[curPass]);
	cout<<endl;
	//the first pass of partitioning
	clock_t timer=0;
	//startTimer(timer);
	int moveRightBits=totalBitsUsed-bitUsedPerPass[0];
	int mask=(1<<bitUsedPerPass[0])-1;
	int numPart=1<<bitUsedPerPass[0];
	Record *Rout=new Record[rLen];
	int* startHistR=new int[numPart];
	int* endHistR=new int[numPart];
	radixPart(R,rLen,moveRightBits,mask,Rout,startHistR,numThread);
	

	Record *Sout=new Record[sLen];
	int* startHistS=new int[numPart];
	int* endHistS=new int[numPart];
	radixPart(S,sLen,moveRightBits,mask,Sout,startHistS,numThread);
	//endTimer("first pass in radixPart", timer);

	int i=0;
	for(i=0;i<numPart-1;i++)
	{
		endHistR[i]=startHistR[i+1];
		endHistS[i]=startHistS[i+1];
	}
	endHistR[i]=rLen;
	endHistS[i]=sLen;


	LinkedList **llList=(LinkedList **)malloc(sizeof(LinkedList*)*numThread);
	initMLL(llList, numThread);

	int* start=new int[numThread];
	int* end=new int[numThread];
	int chunkSize=numPart/numThread;
	for(i=0;i<numThread;i++)
	{
		start[i]=i*chunkSize;
		if((i+1)==numThread)
			end[i]=numPart;
		else
			end[i]=(i+1)*chunkSize;
	}
#ifdef DEBUG_SAVEN
	for(i=0;i<numThread;i++)
		cout<<"J"<<i<<": "<<start[i]<<", "<<end[i]<<endl;
#endif
	
	if(totalNumPass==1)
	{
		//join
		//startTimer(timer);
		#pragma omp parallel for
		for(i=0;i<numThread;i++)
		{
			radixJoin(Rout,startHistR,endHistR,Sout,startHistS,endHistS,start[i],end[i],llList[i],i, numThread);
		}
		//endTimer("radixJoin", timer);
	}
	else
	{
		//startTimer(timer);
		moveRightBits=moveRightBits-bitUsedPerPass[1];
		mask=(1<<bitUsedPerPass[1])-1;
		//partition and join.
		//omp_set_num_threads(numThread);
		#pragma omp parallel for
		for(i=0;i<numThread;i++)
		{
			radixPartJoin(Rout,startHistR,endHistR,Sout,startHistS,endHistS,start[i],end[i],moveRightBits, mask,llList[i], i, numThread);
		}
		//endTimer("radixPartJoin", timer);
	}
	
	result=dumpMLLtoArray(llList,numThread,Joinout);
	delete start;
	delete end;
	delete startHistR;
	delete endHistR;
	delete startHistS;
	delete endHistS;
	
	delete Rout;
	delete Sout;
	freeMLL(llList,numThread);
	return result;
}