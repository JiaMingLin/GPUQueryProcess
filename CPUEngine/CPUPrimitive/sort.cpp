#include "stdafx.h"
#include "MyThreadPool.h"
#include "Primitive.h"
#include "omp.h"
#include "common.h"
#include "assert.h"
#include "CPU_Dll.h"

struct pivotStruct
{
	Record* sampleRin;
	int level;
	int baseOffset;//the offset to the leaf node.
};

void qsort_pidfunc(void *Rin, void* para, void *Rout)
{
	Record *in=(Record*)Rin;
	int *o=(int*)Rout;
	pivotStruct *ps=(pivotStruct*)para;
	Record *sample=ps->sampleRin;
	int baseOffset=ps->baseOffset;
	int level=ps->level;
	int loc=0;
	int pid=0;
	int curLevel=0;
	while(curLevel<level)
	{
		if(in->value<sample[loc].value)
		{
			pid=(pid<<1)+1;
			loc=(loc<<1)+1;
		}
		else
		{
			pid=(pid<<1)+2;
			loc=(loc<<1)+2;
		}
		curLevel++;
	}
	*o=pid-baseOffset;
}
//for sorting the strings.

char *g_stringData=NULL;
void qsortstring_pidfunc(void *Rin, void* para, void *Rout)
{
	Record *in=(Record*)Rin;
	int *o=(int*)Rout;
	pivotStruct *ps=(pivotStruct*)para;
	Record *sample=ps->sampleRin;
	int baseOffset=ps->baseOffset;
	int level=ps->level;
	int loc=0;
	int pid=0;
	int curLevel=0;
	while(curLevel<level)
	{
		//if(in->value<sample[loc].value)
		if(compareStringCPU(in,sample+loc)<0)
		{
			pid=(pid<<1)+1;
			loc=(loc<<1)+1;
		}
		else
		{
			pid=(pid<<1)+2;
			loc=(loc<<1)+2;
		}
		curLevel++;
	}
	*o=pid-baseOffset;
}


int compareStringCPU(const void *d_a, const void *d_b)
{
	Record* r_a=(Record*)d_a;
	Record* r_b=(Record*)d_b;
	char *str_a=g_stringData+r_a->value;
	char *str_b=g_stringData+r_b->value;
	int i=0;
	int result=0;
	while(str_a[i]!='\0' && str_b[i]!='\0' && result==0)
	{
		if(str_a[i]==str_b[i])
			i++;
		else
			if(str_a[i]>str_b[i])
			{
				result=1;
			}
			else
			{
				result=-1;
			}
	}
	if(result==0)
	{
		if(str_a[i]=='\0' && str_b[i]=='\0')
			result=0;
		else
			if(str_a[i]=='\0')
				result=-1;
			else
				result=1;
	}
	return result;
}


DWORD WINAPI tp_sort( LPVOID lpParam ) 
{ 
    ws_sort* pData;
	pData = (ws_sort*)lpParam;
	Record *Rout=pData->Rout;
	cmp_func fcn=pData->fcn;
	int startID=pData->startID;
	int endID=pData->endID;
	int i=0;
	int sum=0;
	qsort(Rout+startID, (endID-startID), sizeof(Record),fcn);	
	return sum;
} 

//compute the sum.
void sort_thread(Record* Rin, int rLen, cmp_func fcn, Record* Rout, int numThread)
{
	int* start=new int[numThread];
	int* end=new int[numThread];
	Record **sampleRout=(Record**)malloc(sizeof(Record*));
	int sampleLevel=log2(numThread);
	getPivot(Rin,rLen,sampleLevel,fcn,sampleRout);
	pivotStruct* ps=new pivotStruct;
	ps->level=sampleLevel;
	ps->sampleRin=*sampleRout;
	ps->baseOffset=(1<<sampleLevel)-1;
	int numPart=(1<<sampleLevel);
	int *startHist=new int[numPart+1];//the last element is rLen
	startHist[numPart]=rLen;
	assert(numPart==numThread);
	split(Rin,rLen,numPart,Rout,startHist,qsort_pidfunc,(void*) ps, numThread);

	MyThreadPool *pool=new MyThreadPool();
	pool->init(numThread, tp_sort);
	int i=0;
	ws_sort** pData=(ws_sort**)malloc(sizeof(ws_sort*)*numThread);
	for( i=0; i<numThread; i++ )
	{
		// Allocate memory for thread data.
		pData[i] = (ws_sort*) HeapAlloc(GetProcessHeap(),
				HEAP_ZERO_MEMORY, sizeof(ws_sort));

		if( pData[i]  == NULL )
			ExitProcess(2);

		// Generate unique data for each thread.
		pData[i]->Rout=Rout;
		pData[i]->fcn=fcn;
		pData[i]->startID=startHist[i];
		pData[i]->endID=startHist[i+1];
		pool->assignParameter(i, pData[i]);
	}
	pool->run();
	delete pool;
	for(i=0;i<numThread;i++)
		HeapFree(GetProcessHeap(),0, pData[i]);
	free(pData);
	delete start;
	delete end;
	delete ps;
	delete startHist;
}

void qsort_openmp_thread(Record* Rin, int rLen, int unitSize, cmp_func fcn, int cpuid, int numThread)
{
	set_thread_affinity(cpuid,numThread);
	qsort(Rin, rLen, unitSize,fcn);
}

void sort_openmp(Record* Rin, int rLen, cmp_func fcn, mapper_t qsort_pidfunc, Record* Rout, int numThread)
{
	//get the pivots
	int* start=new int[numThread];
	int* end=new int[numThread];
	Record **sampleRout=(Record**)malloc(sizeof(Record*));
	int sampleLevel=log2(numThread)+2;
	getPivot(Rin,rLen,sampleLevel,fcn,sampleRout);
	pivotStruct* ps=new pivotStruct;
	ps->level=sampleLevel;
	ps->sampleRin=*sampleRout;
	ps->baseOffset=(1<<sampleLevel)-1;
	int numPart=(1<<sampleLevel);
	int *startHist=new int[numPart+1];//the last element is rLen
	startHist[numPart]=rLen;
	//partition by split.
	split(Rin,rLen,numPart,Rout,startHist,qsort_pidfunc,(void*) ps, 1);//no nested parallelism
	int i=0;
	//sort.
	//omp_set_num_threads(numThread);
	#pragma omp parallel for
	for(i=0;i<numPart;i++)
	{
		printf("i: %d, %d\t",i, startHist[i+1]-startHist[i]);
		qsort_openmp_thread(Rout+startHist[i], (startHist[i+1]-startHist[i]),sizeof(Record),fcn,i,numThread);		
	}
	printf("\n");
	delete start;
	delete end;
	delete ps;
	delete startHist;
}


//sort the string. 
void sortString_openmp(char* stringData, Record* Rin, int rLen, Record* Rout, int numThread)
{
	g_stringData=stringData;
	sort_openmp(Rin,rLen,compareStringCPU,qsortstring_pidfunc,Rout,numThread);
}




