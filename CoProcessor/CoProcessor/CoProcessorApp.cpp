#include "CPU_Dll.h"
#include "SingularThreadOp.h"
#include "CoProcessorTest.h"
#include <iostream>
using namespace std;

int main(int argc, char **argv)
{
	set_numCoreForCPU(NUM_CORE_FOR_CPU_PROCESSING);
#ifdef FIXED_CORE_TO_GPU
	cout<<"FIXED_CORE_TO_GPU"<<endl;
#else
	cout<<"FIXED_CORE_TO_GPU disabled!"<<endl;
#endif
	int i=1;
	for(i=0;i<argc;i++)
		cout<<argv[i]<<" ";
	cout<<endl;
	for(i=1;i<argc;i++)
	{
		if(strcmp(argv[i], "-cop")==0)
		{
			int testCase=1;//1, mix, 2, simple, 3, complicated.
			int modeDelta=0;
			int numWorkerThread=2;
			int numQueries=20;
			int queryDelta=0;
			if(argc==(i+3))
			{
				testCase=atoi(argv[i+1]);
				modeDelta=atoi(argv[i+2]);
			}
			if(argc==(i+4))
			{
				testCase=atoi(argv[i+1]);
				modeDelta=atoi(argv[i+2]);
				numWorkerThread=atoi(argv[i+3]);
			}
			if(argc==(i+5))
			{
				testCase=atoi(argv[i+1]);
				modeDelta=atoi(argv[i+2]);
				numWorkerThread=atoi(argv[i+3]);
				numQueries=atoi(argv[i+4]);
			}
			if(argc==(i+6))
			{
				testCase=atoi(argv[i+1]);
				modeDelta=atoi(argv[i+2]);
				numWorkerThread=atoi(argv[i+3]);
				numQueries=atoi(argv[i+4]);
				queryDelta=atoi(argv[i+5]);
			}
			EXEC_MODE eM=(EXEC_MODE)(EXEC_CPU+modeDelta);
			QUERY_TYPE qT=(QUERY_TYPE)(Q_POINT_SELECTION+queryDelta);
			if(testCase==1)
				testQueryProcessor(Q_POINT_SELECTION,Q_INLJ,numQueries,eM,numWorkerThread);
			if(testCase==2)
				testQueryProcessor(Q_POINT_SELECTION,Q_AGG,numQueries,eM,numWorkerThread);
			if(testCase==3)
				testQueryProcessor(Q_ORDERBY,Q_INLJ,10,eM,numWorkerThread);
			if(testCase==4)
				testQueryProcessor(Q_POINT_SELECTION,Q_NINLJ,numQueries,eM,numWorkerThread,true);
			if(testCase==5)
				testQueryProcessor(qT,qT,numQueries,eM,numWorkerThread,true);
			
				//testQueryProcessor(Q_NINLJ,Q_NINLJ,5,eM,numWorkerThread);
		}
		if(strcmp(argv[i], "-copSingle")==0)
		{
			int queryDelta=0;
			int modeDelta=0;
			if(argc==(i+3))
			{
				queryDelta=atoi(argv[i+1]);
				modeDelta=atoi(argv[i+2]);
			}
			EXEC_MODE eM=(EXEC_MODE)(EXEC_CPU+modeDelta);
			QUERY_TYPE qT=(QUERY_TYPE)(Q_POINT_SELECTION+queryDelta);
			testQueryProcessor(qT,qT,20,eM);
		}

		if(strcmp(argv[i], "-micro")==0)
		{
			int scale=1;//1, mix, 2, simple, 3, complicated.
			int modeDelta=0;
			int numWorkerThread=2;
			if(argc==(i+4))
			{
				scale=atoi(argv[i+1]);
				modeDelta=atoi(argv[i+2]);
				numWorkerThread=atoi(argv[i+3]);
			}
			if(argc==(i+3))
			{
				scale=atoi(argv[i+1]);
				modeDelta=atoi(argv[i+2]);
			}
			EXEC_MODE eM=(EXEC_MODE)(EXEC_CPU+modeDelta);
			testMbench(20,eM,numWorkerThread,scale);
		}
		if(strcmp(argv[i], "-single")==0)
		{
			int queryDelta=0;
			int isGPUONLY_QP=0;
			int isAdaptive=0;
			int execMode=0;
			if(argc==(i+5))
			{
				queryDelta=atoi(argv[i+1]);
				isGPUONLY_QP=atoi(argv[i+2]);
				isAdaptive=atoi(argv[i+3]);
				execMode=atoi(argv[i+4]);
			}
			else
				cout<<"-single <querID> <isGPUONLY_QP> <isAdaptive> <execMode>"<<endl;
			test_plan(queryDelta,isGPUONLY_QP, isAdaptive,execMode);			
		}

		if(strcmp(argv[i], "-microSingle")==0)
		{
			int queryDelta=0;
			int isGPUONLY_QP=0;
			int isAdaptive=0;
			int execMode=0;
			int scale=2;
			if(argc==(i+6))
			{
				scale=atoi(argv[i+1]);
				queryDelta=atoi(argv[i+2]);
				isGPUONLY_QP=atoi(argv[i+3]);
				isAdaptive=atoi(argv[i+4]);
				execMode=atoi(argv[i+5]);
			}
			else
				cout<<"-microSingle <scale> <querID> <isGPUONLY_QP> <isAdaptive> <execMode>"<<endl;
			testMicroBenchmark(scale, queryDelta,isGPUONLY_QP, isAdaptive,execMode);			
		}

		
	}
	
	return 0;
}