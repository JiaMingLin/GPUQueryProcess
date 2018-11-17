#include "stdafx.h"
#include "MyThreadPool.h"
#include "Primitive.h"
#include "omp.h"
#include "CPU_Dll.h"



DWORD WINAPI tp_scatter( LPVOID lpParam ) 
{ 
    ws_scatter* pData;
	pData = (ws_scatter*)lpParam;
	Record *R=pData->R;
	Record *S=pData->S;
	int *loc=pData->loc;
	int startID=pData->startID;
	int endID=pData->endID;
	int i=0;
	int sum=0;
	int j=0;
	int targetPos=0;
	for(j=startID;j<endID;j++)
	{
		targetPos=loc[j];
		S[targetPos]=R[j];
	}
	//pData->sum=sum;
	return sum;
} 

//compute the sum.
void scatter_thread(Record *Rin, int *loc, Record *S, int rLen, int numThread)
{
	MyThreadPool *pool=new MyThreadPool();
	pool->init(numThread, tp_scatter);
	int i=0;
	ws_scatter** pData=(ws_scatter**)malloc(sizeof(ws_scatter*)*numThread);
	int chunkSize=rLen/numThread;
	if(rLen%numThread!=0)
		chunkSize++;
	for( i=0; i<numThread; i++ )
	{
		// Allocate memory for thread data.
		pData[i] = (ws_scatter*) HeapAlloc(GetProcessHeap(),
				HEAP_ZERO_MEMORY, sizeof(ws_scatter));

		if( pData[i]  == NULL )
			ExitProcess(2);

		// Generate unique data for each thread.
		pData[i]->R=Rin;
		pData[i]->S=S;
		pData[i]->loc=loc;
		pData[i]->startID=i*chunkSize;
		pData[i]->endID=(i+1)*chunkSize;
		if(pData[i]->endID>rLen)
			pData[i]->endID=rLen;
		pool->assignParameter(i, pData[i]);
	}
	pool->run();
	delete pool;
	for(i=0;i<numThread;i++)
		HeapFree(GetProcessHeap(),0, pData[i]);
	free(pData);
}


//for gather
DWORD WINAPI tp_gather( LPVOID lpParam ) 
{ 
    ws_scatter* pData;
	pData = (ws_scatter*)lpParam;
	Record *R=pData->R;
	Record *S=pData->S;
	int *loc=pData->loc;
	int startID=pData->startID;
	int endID=pData->endID;
	int i=0;
	int sum=0;
	int j=0;
	int targetPos=0;
	for(j=startID;j<endID;j++)
	{
		targetPos=loc[j];
		S[j]=R[targetPos];
	}
	//pData->sum=sum;
	return sum;
} 

//compute the sum.
void gather_thread(Record *Rin, int *loc, Record *S, int rLen, int numThread)
{
	MyThreadPool *pool=new MyThreadPool();
	pool->init(numThread, tp_gather);
	int i=0;
	ws_scatter** pData=(ws_scatter**)malloc(sizeof(ws_scatter*)*numThread);
	int chunkSize=rLen/numThread;
	if(rLen%numThread!=0)
		chunkSize++;
	for( i=0; i<numThread; i++ )
	{
		// Allocate memory for thread data.
		pData[i] = (ws_scatter*) HeapAlloc(GetProcessHeap(),
				HEAP_ZERO_MEMORY, sizeof(ws_scatter));

		if( pData[i]  == NULL )
			ExitProcess(2);

		// Generate unique data for each thread.
		pData[i]->R=Rin;
		pData[i]->S=S;
		pData[i]->loc=loc;
		pData[i]->startID=i*chunkSize;
		pData[i]->endID=(i+1)*chunkSize;
		if(pData[i]->endID>rLen)
			pData[i]->endID=rLen;
		pool->assignParameter(i, pData[i]);
	}
	pool->run();
	delete pool;
	for(i=0;i<numThread;i++)
		HeapFree(GetProcessHeap(),0, pData[i]);
	free(pData);
}

//open mp versions

void scatter_openmp(Record *R, int *loc, Record *S, int rLen)
{
	//set_CPU_affinity(rand()%3,3);//we reserve the core 4 as the GPU scheduler.
	int j=0;
	int targetPos=0;
	for(j=0;j<rLen;j++)
	{
		S[loc[j]]=R[j];
	}

}

void gather_openmp(Record *R, int *loc, Record *S, int rLen)
{
	//set_CPU_affinity(rand()%3,3);//we reserve the core 4 as the GPU scheduler.
	int j=0;
	int targetPos=0;
	for(j=0;j<rLen;j++)
	{
		S[j]=R[loc[j]];
	}

}