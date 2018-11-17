#include "stdafx.h"
#include "MyThreadPool.h"
#include "Primitive.h"
#include "omp.h"

#define SEG_PER_THREAD_MODE 
#ifdef SEG_PER_THREAD_MODE
DWORD WINAPI tp_scan( LPVOID lpParam ) 
{ 
    ws_scan* pData;
	pData = (ws_scan*)lpParam;
	Record *data=pData->data;
	int startID=pData->startID;
	int endID=pData->endID;
	int i=0;
	int sum=0;
	for(i=startID;i<endID;i++)
	{
		sum+=data[i].value;
	}
	pData->sum=sum;
	return sum;
} 

//compute the sum.
int scan_thread(Record *Rin, int rLen, int numThread)
{
	int result=0;
	MyThreadPool *pool=new MyThreadPool();
	pool->init(numThread, tp_scan);
	int i=0;
	ws_scan** pData=(ws_scan**)malloc(sizeof(ws_scan*)*numThread);
	int chunkSize=rLen/numThread;
	if(rLen%numThread!=0)
		chunkSize++;
	for( i=0; i<numThread; i++ )
	{
		// Allocate memory for thread data.
		pData[i] = (ws_scan*) HeapAlloc(GetProcessHeap(),
				HEAP_ZERO_MEMORY, sizeof(ws_scan));

		if( pData[i]  == NULL )
			ExitProcess(2);

		// Generate unique data for each thread.
		pData[i]->data=Rin;
		pData[i]->startID=i*chunkSize;
		pData[i]->endID=(i+1)*chunkSize;
		if(pData[i]->endID>rLen)
			pData[i]->endID=rLen;
		pool->assignParameter(i, pData[i]);
	}
	pool->run();
	for(i=0;i<numThread;i++)
		result+=pData[i]->sum;
	delete pool;
	for(i=0;i<numThread;i++)
		HeapFree(GetProcessHeap(),0, pData[i]);
	free(pData);

	return result;
}
int scan_sum(Record *Rin, int start, int end, int *pS)
{
	int result=0;
	int i=0;
	for(i=start;i<end;i++)
	{
		pS[i]=result;
		result+=Rin[i].value;
	}
	return result;
}

void scan_prefixSum(Record *Rin, int start, int end, int baseValue, int *pS)
{
	int i=0;
	for(i=start;i<end;i++)
	{
		pS[i]=pS[i]+baseValue;
	}
}
int scan_openmp(Record *Rin, int rLen, int numThread, int *pS)
{
	int result=0;
	int* partialSum=new int[numThread];
	int* start=new int[numThread];
	int* end=new int[numThread];
	int chunkSize=rLen/numThread;
	if(rLen%numThread!=0)
		chunkSize++;
	int i=0;

	// when numThread = 1 it's the best
	omp_set_num_threads(numThread);
	#pragma omp parallel for
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
		partialSum[i]=scan_sum(Rin, start[i], end[i], pS);
	}
	result=0;
	int tempResult=0;
	for(i=0;i<numThread;i++)
	{
		result+=partialSum[i];
		partialSum[i]=tempResult;
		tempResult=result;
	}
	#pragma omp parallel for
	for( i=0; i<numThread; i++ )
	{
		scan_prefixSum(Rin, start[i], end[i], partialSum[i], pS);
	}
	delete partialSum;
	delete start;
	delete end;
	return result;
}
#else//no SEG_PER_THREAD_MODE.

DWORD WINAPI tp_scan( LPVOID lpParam ) 
{ 
    ws_scan* pData;
	pData = (ws_scan*)lpParam;
	Record *data=pData->data;
	int startID=pData->startID;
	int endID=pData->endID;
	int delta=pData->delta;
	int i=0;
	int sum=0;
	for(i=startID;i<endID;i=i+delta)
	{
		sum+=data[i].value;
	}
	pData->sum=sum;
	return sum;
} 

//compute the sum.
int scan(Record *Rin, int rLen, int numThread)
{
	int result=0;
	MyThreadPool *pool=new MyThreadPool();
	pool->init(numThread, tp_scan);
	int i=0;
	ws_scan** pData=(ws_scan**)malloc(sizeof(ws_scan*)*numThread);
	for( i=0; i<numThread; i++ )
	{
		// Allocate memory for thread data.
		pData[i] = (ws_scan*) HeapAlloc(GetProcessHeap(),
				HEAP_ZERO_MEMORY, sizeof(ws_scan));

		if( pData[i]  == NULL )
			ExitProcess(2);

		// Generate unique data for each thread.
		pData[i]->data=Rin;
		pData[i]->startID=i;
		pData[i]->endID=rLen;
		pData[i]->delta=numThread;
		pool->assignParameter(i, pData[i]);
	}
	pool->run();
	for(i=0;i<numThread;i++)
		result+=pData[i]->sum;
	delete pool;
	for(i=0;i<numThread;i++)
		HeapFree(GetProcessHeap(),0, pData[i]);
	free(pData);

	return result;
}

#endif