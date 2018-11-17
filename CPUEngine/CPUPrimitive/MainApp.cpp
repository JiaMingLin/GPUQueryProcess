// CPUPrimitive.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include "Test.h"
#include "common.h"
#include "AccessMethods.h"
#include "CPU_Dll.h"
#include <iostream>
using namespace std;



void set_thread_affinity_rnd (int numThread) 
{
    #pragma omp parallel default(shared)
    {
    DWORD_PTR mask = (1 << (rand()%numThread))%((1<<numThread));
	cout<<mask<<endl;
    SetThreadAffinityMask( GetCurrentThread(), mask );
    }
}




void sortRecord(Record* Rin, int rLen, Record *Rout, int cpuid, int numThread)
{
	set_thread_affinity(cpuid, numThread);
	//set_thread_affinity_rnd(numThread);
	sort(Rin,rLen,compare,Rout,numThread);
}

void testOpenMPThread(int numThread)
{
	int numPart=4;
	Record** Rin=(Record**)malloc(sizeof(Record*)*numPart);
	Record** Rout=(Record**)malloc(sizeof(Record*)*numPart);
	int i=0;
	int rLen=16*1024*1024;
	for(i=0;i<numPart;i++)
	{
		Rin[i]=new Record[rLen];
		generateRand(Rin[i],TEST_MAX,rLen, i);
		Rout[i]=new Record[rLen];
	}
	cout<<"start sorting: "<<numThread<<endl;
	//omp_set_num_threads(numThread);
	//#pragma omp parallel for
	for(i=0;i<1;i++)
		sortRecord(Rin[i],rLen,Rout[i], i%numThread, numThread);

	
	for(i=0;i<numPart;i++)
	{
		delete Rin[i];
		delete Rout[i];
	}
	free(Rin);
	free(Rout);
}

int main(int argc, char **argv)
{
	//testOpenMPThread(atoi(argv[1]));
#ifdef SAVEN_OPen_MP
	cout<<"openmp, ";
#else
	cout<<"home-made threading, ";
#endif
	int i=0;
	for(i=0;i<argc;i++)
		cout<<argv[i]<<" ";
	cout<<endl;
	testPrimitive(argc,argv);
	testJoin(argc,argv);
	testAccessMethods(argc,argv);
	testDLL(argc,argv);
	return 0;
}

//test reduce
/*void main()
{
	int rLen = 1024*1024*16;
	int numThread = 4;
	int OPERATOR = REDUCE_MAX;

	testReduce( rLen, numThread, OPERATOR);
}

//test filter
void main()
{
	int rLen = 1024*1024*16;
	int numThread = 4;
	
	testFilter( rLen, numThread );
}*/

