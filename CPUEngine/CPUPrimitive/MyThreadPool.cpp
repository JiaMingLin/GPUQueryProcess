#include "stdafx.h"
#include "stdlib.h"
#include "MyThreadPool.h"



MyThreadPool::MyThreadPool(void): 
numThread(0), 
masks(NULL), 
dwThreadId(NULL), 
hThread(NULL), 
tParam (NULL)
{
}

MyThreadPool::~MyThreadPool(void)
{
	if(dwThreadId!=NULL)
		delete dwThreadId;
	if(hThread!=NULL)
		delete hThread;
	if(masks!=NULL)
		delete masks;
	if(tParam!=NULL)
		free(tParam);
}

int MyThreadPool::setMask(int tid, int maskValue)
{
	//SetThreadAffinityMask(hThread[tid],maskValue);
	masks[tid]=maskValue;
	return 0;
}

int MyThreadPool::assignTask(LPTHREAD_START_ROUTINE  pfn)
{
	this->pfnThreadProc=pfn;
	return 0;
}

int MyThreadPool::assignParameter(int tid, void* pvParam)
{
	tParam[tid]=pvParam;
	return 0;
}

int MyThreadPool::run(void)
{
	int i=0;
	for( i=0; i<numThread; i++ )
	{
		hThread[i] = CreateThread( 
			NULL,              // default security attributes
			0,                 // use default stack size  
			pfnThreadProc,        // thread function 
			tParam[i],             // argument to thread function 
			0,                 // use default creation flags 
			&dwThreadId[i]);   // returns the thread identifier 
 
		// Check the return value for success. 
		SetThreadAffinityMask(hThread[i],masks[i]);
 
		if (hThread[i] == NULL) 
		{
			ExitProcess(i);
		}
	}
	// Wait until all threads have terminated.

	WaitForMultipleObjects(numThread, hThread, TRUE, INFINITE);
	// Close all thread handles upon completion.
	for(i=0; i<numThread; i++)
	{
//		CloseHandle(hThread[i]);
	}
	return 0;
}

int MyThreadPool::init(int nThead, LPTHREAD_START_ROUTINE  pfn)
{
	numThread=nThead;
	dwThreadId=new DWORD[numThread];
	hThread=new HANDLE[numThread];
	masks=new int[numThread];
	int i=0;
	for(i=0;i<numThread;i++)
		masks[i]=(1<<i)%MAX_RUN;
	this->tParam=(void**)malloc(sizeof(void*)*numThread);
	this->pfnThreadProc=pfn;
	return 0;
}
