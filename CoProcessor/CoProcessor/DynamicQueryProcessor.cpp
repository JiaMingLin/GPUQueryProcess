#include "CoProcessor.h"
#include "MyThreadPoolCop.h"
#include "CoProcessorTest.h"
#include "QueryPlanTree.h"
#include "Database.h"

#define CPUCORE_FORGPU 4

#define SMART_PROCESSOR 1
#define WORK_STEALING 1

#define STEALING_CHANCE 3

HANDLE gpuInUsedMutex=NULL;
bool isGPUavailable=true;
int evalautedQuery=0;

int pickQueryRandom(Query_stat *gQstat, int numQuery)
{
	__DEBUG__("pickQueryRandom");
	int i=0;
	int result=-1;
	for(i=0;i<numQuery;i++)
		if(gQstat[i].isAssigned==false)
		{
			result=i;
			gQstat[i].isAssigned=true;
			break;
		}
	return result;
}
#ifdef WORK_STEALING
int pickQuerySmart(int threadid, Query_stat *gQstat, int numQuery, EXEC_MODE toBeEM, float thresholdForGPUApp, float thresholdForCPUApp, EXEC_MODE& preferMode)
{
	__DEBUG__("pickQuerySmart");
	int result=-1;
	if(toBeEM==EXEC_CPU)
	{
		//pick from the end and find the one with the minimum speedup.
		int i=0;
		float minSpeedup=65536;//the maximum possible speedup:)
		for(i=numQuery-1;i>=0;i--)
		{
			if(gQstat[i].isAssigned==false)
			{
				if(gQstat[i].speedupGPUoverCPU<=thresholdForGPUApp)//the job can be run on the CPU
				{
					if(minSpeedup>=gQstat[i].speedupGPUoverCPU)
					{
						result=i;//keep as the minimum
						minSpeedup=gQstat[i].speedupGPUoverCPU;
					}
				}
			}
		}
	}
	else if(toBeEM==EXEC_GPU || toBeEM==EXEC_GPU_ONLY)
	{
		int i=0;
		float maxSpeedup=0;//the maximum possible speedup:)
		for(i=0;i<numQuery;i++)
		{
			if(gQstat[i].isAssigned==false)
			{
				if(gQstat[i].speedupGPUoverCPU>=thresholdForCPUApp)//the job can be run on the GPU.
				{
					if(maxSpeedup<=gQstat[i].speedupGPUoverCPU)
					{
						result=i;
						maxSpeedup=gQstat[i].speedupGPUoverCPU;
					}
				}
			}
		}
	}
	else if(toBeEM==EXEC_ADAPTIVE)
	{
		/*int i=0;
		for(i=0;i<numQuery;i++)
		{
			if(gQstat[i].isAssigned==false)
			{
				result=i;
				if(gQstat[i].needCop==true)
					break;
			}
		}*/
		if(threadid==0)
		{
			int i=0;
			float maxSpeedup=0;//the maximum possible speedup:)
			for(i=0;i<numQuery;i++)
			{
				if(gQstat[i].isAssigned==false)
				{
					if(gQstat[i].speedupGPUoverCPU>=thresholdForCPUApp)//the job can be run on the GPU.
					{
						if(maxSpeedup<=gQstat[i].speedupGPUoverCPU)
						{
							result=i;
							maxSpeedup=gQstat[i].speedupGPUoverCPU;
						}
					}
				}
			}
		}
		else
		{
			int i=0;
			float minSpeedup=65536;//the maximum possible speedup:)
			for(i=numQuery-1;i>=0;i--)
			{
				if(gQstat[i].isAssigned==false)
				{
					if(gQstat[i].speedupGPUoverCPU<=thresholdForGPUApp)//the job can be run on the CPU
					{
						if(minSpeedup>=gQstat[i].speedupGPUoverCPU)
						{
							result=i;//keep as the minimum
							minSpeedup=gQstat[i].speedupGPUoverCPU;
						}
					}
				}
			}
		}
	}
	if(result!=-1)
	{
		gQstat[result].isAssigned=true;					
		if(toBeEM==EXEC_ADAPTIVE)
		{
			//a simple estimation for worthwhile,
			//need to replace it with a more realistic one.
			preferMode=toBeEM;
			if(threadid==0)
			{
				//stealing it on the GPU.
				if(result%STEALING_CHANCE!=0)
					preferMode=EXEC_CPU;
				else
					preferMode=EXEC_GPU_ONLY;
			}
			else
			{
				preferMode=EXEC_CPU;
			}
			if(gQstat[result].needCop==true)
				preferMode=EXEC_ADAPTIVE;
		}
		else if(toBeEM==EXEC_CPU)
		{
			if(gQstat[result].speedupGPUoverCPU>1)
			{
				if(result%STEALING_CHANCE!=0)
					preferMode=EXEC_GPU_ONLY;
				else
					preferMode=EXEC_CPU;//steal
			}
		}
		else if(toBeEM==EXEC_GPU || toBeEM==EXEC_GPU_ONLY)
		{
			if(gQstat[result].speedupGPUoverCPU<1)
			{
				if(result%STEALING_CHANCE!=0)
					preferMode=EXEC_CPU;
				else
					preferMode=EXEC_GPU_ONLY;//steal
			}
		}
		
	}
	return result;
}
#else//no work stealing...
int pickQuerySmart(int threadid, Query_stat *gQstat, int numQuery, EXEC_MODE toBeEM, float thresholdForGPUApp, float thresholdForCPUApp, EXEC_MODE& preferMode)
{
	__DEBUG__("pickQuerySmart");
	int result=-1;
	if(toBeEM==EXEC_CPU)
	{
		//pick from the end and find the one with the minimum speedup.
		int i=0;
		float minSpeedup=65536;//the maximum possible speedup:)
		for(i=numQuery-1;i>=0;i--)
		{
			if(gQstat[i].isAssigned==false)
			{
				if(gQstat[i].speedupGPUoverCPU<=1)//the job can be run on the CPU
				{
					if(minSpeedup>=gQstat[i].speedupGPUoverCPU)
					{
						result=i;//keep as the minimum
						minSpeedup=gQstat[i].speedupGPUoverCPU;
					}

				}
			}
		}
	}
	else if(toBeEM==EXEC_GPU || toBeEM==EXEC_GPU_ONLY)
	{
		int i=0;
		float maxSpeedup=0;//the maximum possible speedup:)
		for(i=0;i<numQuery;i++)
		{
			if(gQstat[i].isAssigned==false)
			{
				if(gQstat[i].speedupGPUoverCPU>=1)//the job can be run on the GPU.
				{
					if(maxSpeedup<=gQstat[i].speedupGPUoverCPU)
					{
						result=i;
						maxSpeedup=gQstat[i].speedupGPUoverCPU;
					}
				}
			}
		}
	}
	else if(toBeEM==EXEC_ADAPTIVE)
	{
		int i=0;
		for(i=0;i<numQuery;i++)
		{
			if(gQstat[i].isAssigned==false)
			{
				result=i;
				if(gQstat[i].needCop==true)
					break;
			}
		}
	}
	if(result!=-1)
	{
		gQstat[result].isAssigned=true;					
		if(toBeEM==EXEC_ADAPTIVE)
		{
			//a simple estimation for worthwhile,
			//need to replace it with a more realistic one.
			preferMode=toBeEM;
			if(gQstat[result].needCop==true)
				preferMode=EXEC_ADAPTIVE;
		}
		
	}
	return result;
}
#endif

DWORD WINAPI tp_QueryThread(LPVOID lpParam)
{
	tp_singleQuery* pData;
	pData = (tp_singleQuery*)lpParam;
	char* query=pData->query;
	EXEC_MODE eM=pData->eM;
	int id=pData->id;
	int tid=pData->postThreadID;
	if(eM==EXEC_ADAPTIVE)
	{
		setHighPriority();
		set_selfCPUID(3);
		cout<<"set the GPU thread ID to core 4"<<endl;
		QueryPlanTree tree(false);//totally on the GPU
		tree.buildTree(query);
		clock_t l_startTime = clock();
		tree.execute(eM);
		clock_t l_endTime = clock();
		int len=tree.planStatus->numResultColumn*tree.planStatus->numResultRow;
		free(tree.planStatus->finalResult);
		printf("ExecuteQuery, %s, %d on 1time, ",query,len);
		printf("T%d, Adaptive, %d , %.3f ms, \n",tid, id, (l_endTime-l_startTime)*1000/ (double)CLOCKS_PER_SEC);
		//if(tree.hasLock)
		//{
		//	isGPUavailable=true;
			//ReleaseMutex( gpuInUsedMutex );
			//ExitThread(0);
			//cout<<"*******release the lock"<<endl;
		//}
	}
	else if(eM==EXEC_CPU)
	{
		/*cout<<"testing on selections"<<endl;
		query=new char[512];
		strcpy(query, "PRO;R;$;R.a00,;$;:SEL;R;$;R.a00,;=,R.a00,$,$,773919665,$,$,;:$:$:$:");
		//strcpy(query, "PRO;R;$;R.a00,;$;:JOIN;R;S;R.a00,S.a00,;<,R.a00,$,$,S.a00,$,$,;:SEL;R;$;R.a00,;AND,>,R.a00,$,$,90574754,$,$,<,R.a00,$,$,110574754,$,$,;:$:$:SEL;S;$;S.a00,;AND,>,S.a00,$,$,704169827,$,$,<,S.a00,$,$,724169827,$,$,;:$:$:$:");*/
		setHighPriority();
		QueryPlanTree tree(false);
		tree.buildTree(query);
		clock_t l_startTime = clock();
		tree.execute(eM);
		clock_t l_endTime = clock();
		int len=tree.planStatus->numResultColumn*tree.planStatus->numResultRow;
		free(tree.planStatus->finalResult);
		printf("ExecuteQuery, %s, %d on 1time, ",query,len);
		printf("T%d, CPU, %d , %.3f ms, \n",tid, id, (l_endTime-l_startTime)*1000/ (double)CLOCKS_PER_SEC);
	}
	else 
	{
#ifdef FIXED_CORE_TO_GPU
		set_selfCPUID(3);
		cout<<"set the GPU thread ID to core 4"<<endl;
#endif		
		//for testing.
		/*cout<<"testing on selections"<<endl;
		query=new char[512];
		strcpy(query, "PRO;R;$;R.a00,;$;:JOIN;R;S;R.a00,S.a00,;<,R.a00,$,$,S.a00,$,$,;:SEL;R;$;R.a00,;AND,>,R.a00,$,$,90574754,$,$,<,R.a00,$,$,110574754,$,$,;:$:$:SEL;S;$;S.a00,;AND,>,S.a00,$,$,704169827,$,$,<,S.a00,$,$,724169827,$,$,;:$:$:$:");
		//strcpy(query, "PRO;R;$;R.a00,;$;:SEL;R;$;R.a00,;=,R.a00,$,$,773919665,$,$,;:$:$:$:");*/
		setHighPriority();
		set_selfCPUID(3);
		cout<<"set the GPU thread ID to core 4"<<endl;
		QueryPlanTree tree(true);//totally on the GPU
		tree.buildTree(query);
		clock_t l_startTime = clock();
		tree.execute();
		clock_t l_endTime = clock();
		int len=tree.planStatus->numResultColumn*tree.planStatus->numResultRow;
		free(tree.planStatus->finalResult);
		printf("ExecuteQuery, %s, %d on 1time, ",query,len);
		printf("T%d, GPU, %d , %.3f ms, \n",tid, id, (l_endTime-l_startTime)*1000/ (double)CLOCKS_PER_SEC);
	}
	return 0;
}



//the naive query processor, just pick the query random one by one.
DWORD WINAPI tp_naiveQP( LPVOID lpParam ) 
{ 
	tp_batchQuery* pData;
	pData = (tp_batchQuery*)lpParam;
	HANDLE dispatchMutex=pData->dispatchMutex;
	EXEC_MODE eM=pData->eM;
	EXEC_MODE preferMode=pData->eM;
	EXEC_MODE toRunMode=pData->eM;
	int totalQuery=pData->totalQuery;
	char** sqlQuery=pData->sqlQuery;
	int *numActiveThread=pData->numThreadActive;
	Query_stat *gQstat=pData->stat;
	int threadid=pData->threadid;
	int curQuery;
	char* query=NULL;
	float thresholdForGPUApp=3;
	float thresholdForCPUApp=0.5;
	cout<<"threshold: "<<thresholdForGPUApp<<", "<<thresholdForCPUApp<<endl;
	setHighPriority();
	while(1)
	{
		//get the tuples to sort
		WaitForSingleObject( dispatchMutex, INFINITE );
#ifdef SMART_PROCESSOR
		curQuery=pickQuerySmart(threadid,gQstat,totalQuery,eM,thresholdForGPUApp,thresholdForCPUApp,preferMode);
		cout<<"smart picker:)"<<endl;
#else
		curQuery=pickQueryRandom(gQstat,totalQuery);
		cout<<"random picker"<<endl;
#endif
		if(curQuery==-1) 
		{
			ReleaseMutex( dispatchMutex );
			(*numActiveThread)--;
			break;
		}
		//cout<<"did not set the threadign!!! static testing"<<endl;
#ifdef WORK_STEALING
		if(eM!=preferMode)
		{
		//	cout<<"**work stealing*** :"<<eM<<", prefer, "<<preferMode<<endl;
			toRunMode=preferMode;
		}
		else
		{
		//	cout<<"** no work stealing **"<<endl;
			toRunMode=eM;
		}
#endif
		evalautedQuery++;
		ReleaseMutex( dispatchMutex );
		query=sqlQuery[curQuery];
#ifdef ADAPTIVE_DEBUG
		cout<<"T"<<threadid<<", "<<curQuery<<": query "<<gQstat[curQuery].qT<<" ->"<<query<<endl;
#else
		cout<<"T"<<threadid<<", "<<curQuery<<endl;
#endif

		//evaluate it.
		MyThreadPoolCop *pool=(MyThreadPoolCop*)malloc(sizeof(MyThreadPoolCop));
		pool->create(1);
		tp_singleQuery* pData = (tp_singleQuery*) HeapAlloc(GetProcessHeap(),
				HEAP_ZERO_MEMORY, sizeof(tp_singleQuery));

		if( pData  == NULL )
			ExitProcess(2);
		pData->init(toRunMode,query,curQuery,threadid);
		pool->assignParameter(0, pData);
		pool->assignTask(0, tp_QueryThread);
		pool->run();
		HeapFree(GetProcessHeap(),0, pData);
		pool->destory();
		//resetGPU();
	}
	return 0;
} 




void testQueryProcessor(QUERY_TYPE fromType, QUERY_TYPE toType, int numQuery, EXEC_MODE eM, int numThread,bool isEx)
{
#ifdef WORK_STEALING
	cout<<"work stealing"<<endl;
#else
	cout<<"NO work stealing"<<endl;
#endif
	initDB2("RS.conf");
	
	HANDLE dispatchMutex = CreateMutex( NULL, FALSE, NULL );  
	gpuInUsedMutex= CreateMutex( NULL, FALSE, NULL );  
	MyThreadPoolCop *pool=(MyThreadPoolCop*)malloc(sizeof(MyThreadPoolCop));
	EXEC_MODE mode1, mode2;
	if(eM==EXEC_CPU)
	{
		mode1=EXEC_CPU;
		numThread=1;
		cout<<"EXEC_CPU: "<<numThread<<endl;
		
	}
	else if(eM==EXEC_GPU || eM==EXEC_GPU_ONLY)
	{
		mode1=EXEC_GPU_ONLY;
		numThread=1;
		cout<<"eM==EXEC_GPU || eM==EXEC_GPU_ONLY: "<<numThread<<endl;		
	}
	else if(eM==EXEC_CPU_GPU)
	{
		mode1=EXEC_GPU_ONLY;
		mode2=EXEC_CPU;
		numThread=2;
		cout<<"EXEC_CPU_GPU: "<<numThread<<endl;		
	}
	else if(eM==EXEC_ADAPTIVE)
	{
		cout<<"EXEC_ADAPTIVE: "<<numThread<<endl;
		mode1=EXEC_ADAPTIVE;
		mode2=EXEC_ADAPTIVE;		
	}
	pool->create(numThread);
	int i=0;
	tp_batchQuery** pData=(tp_batchQuery**)malloc(sizeof(tp_batchQuery*)*numThread);
	char** sqlQuery=(char**)malloc(sizeof(char*)*numQuery);
	//QUERY_TYPE* queryType=(QUERY_TYPE*)malloc(sizeof(QUERY_TYPE)*numQuery);
	Query_stat* gQStat=new Query_stat[numQuery];
	for(i=0;i<numQuery;i++)
	{
		sqlQuery[i]=new char[512];
		if(isEx==false)
			gQStat[i].qT=makeRandomQuery(fromType,toType,sqlQuery[i]);
		else
		{
			//int randomValue=rand()%20;
			if(i!=0)
			{
				makeTestQuery(fromType,sqlQuery[i]);
				gQStat[i].qT=fromType;
			}
			else
			{
				makeTestQuery(toType,sqlQuery[i]);
				gQStat[i].qT=toType;
			}
		}
		gQStat[i].speedupGPUoverCPU=getSpeedUP(gQStat[i].qT);
		gQStat[i].needCop=getNeedCop(gQStat[i].qT);
		cout<<gQStat[i].speedupGPUoverCPU<<endl;
		gQStat[i].isAssigned=false;
	}
//testing
	/*cout<<"for testing: i set the first four to be ninljs"<<endl;
	for(i=0;i<4;i++)
	{
		sqlQuery[i]=new char[512];
		queryType[i]=makeRandomQuery(Q_NINLJ,Q_NINLJ,sqlQuery[i]);
	}*/
	int curID=0;
	int numActiveThread=numThread;
	for( i=0; i<numThread; i++ )
	{
		// Allocate memory for thread data.
		pData[i] = (tp_batchQuery*) HeapAlloc(GetProcessHeap(),
				HEAP_ZERO_MEMORY, sizeof(tp_batchQuery));

		if( pData[i]  == NULL )
			ExitProcess(2);

		if(i==0)//GPU first.
		{
			pData[i]->init(dispatchMutex,&curID,mode1,numQuery,sqlQuery,gQStat,&numActiveThread,i);
#ifdef FIXED_CORE_TO_GPU
			if(mode1==EXEC_GPU_ONLY || mode1==EXEC_ADAPTIVE)
			{
				pool->setCPUID(i, CPUCORE_FORGPU);
				cout<<"set the GPU thread ID to core 4"<<endl;
			}
#endif
		}
		else
		{
			pData[i]->init(dispatchMutex,&curID,mode2,numQuery,sqlQuery,gQStat,&numActiveThread,i);		
		}

		pool->assignParameter(i, pData[i]);
		pool->assignTask(i, tp_naiveQP);
	}
	clock_t tt;
	startTimer(tt);
	pool->run();
	endTimer("total execution", tt);
	for(i=0;i<numThread;i++)
	{
		HeapFree(GetProcessHeap(),0, pData[i]);
	}
	free(pData);
	pool->destory();
	CloseHandle( dispatchMutex );
}



void testMbench(int numQuery, EXEC_MODE eM, int numThread, int scale)
{
	easedb=new Database();
	easedb->dbmbench(scale);
	easedb->sortTable("T1", "T1.a1");
	easedb->createTreeIndex("T1", "T1.a1");
	
	HANDLE dispatchMutex = CreateMutex( NULL, FALSE, NULL );  
	gpuInUsedMutex= CreateMutex( NULL, FALSE, NULL );  
	MyThreadPoolCop *pool=(MyThreadPoolCop*)malloc(sizeof(MyThreadPoolCop));
	EXEC_MODE mode1, mode2;
	if(eM==EXEC_CPU)
	{
		mode1=EXEC_CPU;
		numThread=1;
		cout<<"EXEC_CPU: "<<numThread<<endl;
		
	}
	else if(eM==EXEC_GPU || eM==EXEC_GPU_ONLY)
	{
		mode1=EXEC_GPU_ONLY;
		numThread=1;
		cout<<"eM==EXEC_GPU || eM==EXEC_GPU_ONLY: "<<numThread<<endl;		
	}
	else if(eM==EXEC_CPU_GPU)
	{
		mode1=EXEC_GPU_ONLY;
		mode2=EXEC_CPU;
		numThread=2;
		cout<<"EXEC_CPU_GPU: "<<numThread<<endl;
		
	}
	else if(eM==EXEC_ADAPTIVE)
	{
		cout<<"EXEC_ADAPTIVE: "<<numThread<<endl;
		mode1=EXEC_ADAPTIVE;
		mode2=EXEC_ADAPTIVE;
		
	}
	pool->create(numThread);
	int i=0;
	tp_batchQuery** pData=(tp_batchQuery**)malloc(sizeof(tp_batchQuery*)*numThread);
	char** sqlQuery=(char**)malloc(sizeof(char*)*numQuery);
	Query_stat* gQStat=new Query_stat[numQuery];
	for(i=0;i<numQuery;i++)
	{
		sqlQuery[i]=new char[512];
		gQStat[i].qT=makeRandomQuery(Q_DBMBENCH1,Q_DBMBENCH3,sqlQuery[i]);
		//gQStat[i].qT=makeRandomQuery(Q_DBMBENCH2,Q_DBMBENCH2,sqlQuery[i]);
		gQStat[i].speedupGPUoverCPU=getSpeedUP(gQStat[i].qT);
		gQStat[i].needCop=getNeedCop(gQStat[i].qT);
		cout<<gQStat[i].speedupGPUoverCPU<<endl;
		gQStat[i].isAssigned=false;
	}
	int curID=0;
	int numActiveThread=numThread;
	for( i=0; i<numThread; i++ )
	{
		// Allocate memory for thread data.
		pData[i] = (tp_batchQuery*) HeapAlloc(GetProcessHeap(),
				HEAP_ZERO_MEMORY, sizeof(tp_batchQuery));

		if( pData[i]  == NULL )
			ExitProcess(2);

		if(i==0)//GPU first.
		{
			pData[i]->init(dispatchMutex,&curID,mode1,numQuery,sqlQuery,gQStat,&numActiveThread,i);
#ifdef FIXED_CORE_TO_GPU
			if(mode1==EXEC_GPU_ONLY || mode1==EXEC_ADAPTIVE)
			{
				pool->setCPUID(i, CPUCORE_FORGPU);
				cout<<"set the GPU thread ID to core 4"<<endl;
			}
#endif
		}
		else
		{
			pData[i]->init(dispatchMutex,&curID,mode2,numQuery,sqlQuery,gQStat,&numActiveThread,i);		
		}

		pool->assignParameter(i, pData[i]);
		pool->assignTask(i, tp_naiveQP);
	}
	clock_t tt;
	startTimer(tt);
	pool->run();
	endTimer("total execution", tt);
	for(i=0;i<numThread;i++)
	{
		HeapFree(GetProcessHeap(),0, pData[i]);
	}
	free(pData);
	pool->destory();
	CloseHandle( dispatchMutex );
}