#include "stdafx.h"
#include "MyThreadPool.h"
#include "Primitive.h"
#include "omp.h"
#include "mapImpl.h"
#include "common.h"


DWORD WINAPI tp_split_hist( LPVOID lpParam ) 
{ 
    ws_split* pData;
	pData = (ws_split*)lpParam;
	int *pidArray=pData->pidArray;
	int startID=pData->startID;
	int endID=pData->endID;
	int *hist=pData->hist;
	int sum=0;
	computeHist(pidArray, startID, endID, hist);
	return sum;
} 

DWORD WINAPI tp_split_loc( LPVOID lpParam ) 
{ 
    ws_split* pData;
	pData = (ws_split*)lpParam;
	int *pidArray=pData->pidArray;
	int startID=pData->startID;
	int endID=pData->endID;
	int *hist=pData->hist;
	int *loc=pData->loc;
	int sum=0;
	outputLocInSplit(pidArray, startID, endID, hist,loc);
	return sum;
} 



void split_thread(Record *Rin, int rLen, int numPart, Record* Rout, int* startHist,mapper_t splitFunc, void*para, int numThread)
{
	//compute the partition ID
	int *pidArray=new int[rLen];
	mapImpl<int>(Rin,rLen,splitFunc,para, pidArray, numThread);	

	//hist
	int **hist=(int**)malloc(sizeof(int*)*numThread);
	int *loc=new int[rLen];
	int i=0;
	int j=0;
	for(i=0;i<numThread;i++)
	{
		hist[i]=new int[numPart];
		for(j=0;j<numPart;j++)
			hist[i][j]=0;
	}
	MyThreadPool *pool=new MyThreadPool();
	pool->init(numThread, tp_split_hist);
	ws_split** pData=(ws_split**)malloc(sizeof(ws_split*)*numThread);
	int chunkSize=rLen/numThread;
	if(rLen%numThread!=0)
		chunkSize++;
	for( i=0; i<numThread; i++ )
	{
		// Allocate memory for thread data.
		pData[i] = (ws_split*) HeapAlloc(GetProcessHeap(),
				HEAP_ZERO_MEMORY, sizeof(ws_split));

		if( pData[i]  == NULL )
			ExitProcess(2);

		// Generate unique data for each thread.
		pData[i]->pidArray=pidArray;
		pData[i]->hist=hist[i];
		pData[i]->startID=i*chunkSize;
		pData[i]->endID=(i+1)*chunkSize;
		pData[i]->loc=loc;
		if(pData[i]->endID>rLen)
			pData[i]->endID=rLen;
		pool->assignParameter(i, pData[i]);
	}
	pool->run();
	for(i=0;i<numThread;i++)
		HeapFree(GetProcessHeap(),0, pData[i]);
	delete pool;
	//loc
	histToOffset(startHist, hist, numPart, numThread);
	ws_split** locData=(ws_split**)malloc(sizeof(ws_split*)*numThread);
	MyThreadPool *locPool=new MyThreadPool();
	locPool->init(numThread, tp_split_loc);
	for( i=0; i<numThread; i++ )
	{
		// Allocate memory for thread data.
		locData[i] = (ws_split*) HeapAlloc(GetProcessHeap(),
				HEAP_ZERO_MEMORY, sizeof(ws_split));

		if( locData[i]  == NULL )
			ExitProcess(2);

		// Generate unique data for each thread.
		locData[i]->pidArray=pidArray;
		locData[i]->hist=hist[i];
		locData[i]->startID=i*chunkSize;
		locData[i]->endID=(i+1)*chunkSize;
		locData[i]->loc=loc;
		if(locData[i]->endID>rLen)
			locData[i]->endID=rLen;
		locPool->assignParameter(i, locData[i]);
	}
	locPool->run();

	
	for(i=0;i<numThread;i++)
		HeapFree(GetProcessHeap(),0, locData[i]);
	free(locData);
	delete locPool;
	//pool->assignTask(tp_split_loc);
	//pool->run();

	

	scatter(Rin,loc,Rout,rLen,numThread);
	for(i=0;i<numThread;i++)
		delete hist[i];
	free(hist);
	delete loc;
	delete pidArray;
}

void split_openmp(Record *Rin, int rLen, int numPart, Record* Rout, int* startHist,mapper_t splitFunc, void *para, int numThread)
{
	int *pidArray=new int[rLen];
	mapImpl<int>(Rin,rLen,splitFunc,para,pidArray,numThread);
	int **hist=(int**)malloc(sizeof(int*)*numThread);
	int i=0;
	int j=0;
	for(i=0;i<numThread;i++)
	{
		hist[i]=new int[numPart];
		for(j=0;j<numPart;j++)
			hist[i][j]=0;
	}
	int* start=new int[numThread];
	int* end=new int[numThread];
	int chunkSize=rLen/numThread;
	if(rLen%numThread!=0)
		chunkSize++;
	//omp_set_num_threads(1);
	for( i=0; i<numThread; i++ )
	{
		start[i]=i*chunkSize;
		end[i]=(i+1)*chunkSize;
		if(end[i]>rLen)
			end[i]=rLen;
	}
	#pragma omp parallel for
	for( i=0; i<numThread; i++ )
	{
		computeHist(pidArray, start[i], end[i],hist[i],i,numThread);
	}
	histToOffset(startHist, hist, numPart, numThread);
	//output according to hist.
	int *loc=new int[rLen];
	#pragma omp parallel for
	for( i=0; i<numThread; i++ )
	{
		outputLocInSplit(pidArray, start[i], end[i],hist[i],loc,i, numThread);
	}
	delete start;
	delete end;

	scatter(Rin,loc,Rout,rLen,numThread);
	for(i=0;i<numThread;i++)
		delete hist[i];
	free(hist);
	delete loc;
	delete pidArray;
}

