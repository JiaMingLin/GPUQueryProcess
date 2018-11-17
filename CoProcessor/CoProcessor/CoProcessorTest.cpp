#include "CoProcessorTest.h"
#include "QueryPlanTree.h"
#include "Database.h"

Database *easedb;

float SpeedupGPUOverCPU[12]={
0.125333,
0.62,//0.422604,
0.06015,
2.080115,
2.217028,
12.10571,
4.926108,
3.456805,
3.538269,
0.124,
13.3,
0.66666666666666666666666666666667
};

bool needCoProcesssing[12]={
false, //0.125333,
false,//0.422604,
false,
false,
false,
true,
false,
false,
false,
false,
true,
false
};

float getSpeedUP(int qid)
{
	return SpeedupGPUOverCPU[qid];
}

bool getNeedCop(int qid)
{
	return needCoProcesssing[qid];
}


void initDB2(char *confFile)
{
	easedb=new Database();
	easedb->loadDB(confFile);
	int i=0;
	int id=easedb->getTableID("R.b00");
	if(id!=-1)
	{
		Record* bb=easedb->tables[id];
		int rLen=easedb->tPro[id].rLen;
		for(i=0;i<rLen;i++)
			bb[i].value=bb[i].value%64;
	}
}



void execQuery(char *query, bool isGPUONLY_QP, bool isAdaptive, EXEC_MODE eM)
{
	if(isAdaptive)
		cout<<"Adaptive query processing"<<endl;
	else
		cout<<"static query processing"<<endl;
	if(isGPUONLY_QP)
		cout<<"GPUonly"<<endl;
	else
		cout<<"not GPU only"<<endl;
	QueryPlanTree tree(isGPUONLY_QP);
	tree.buildTree(query);
	clock_t l_startTime = clock();
	if(isAdaptive)
	{
		eM=EXEC_ADAPTIVE;
	}
	tree.execute(eM);
	clock_t l_endTime = clock();
	int len=tree.planStatus->numResultColumn*tree.planStatus->numResultRow;
	free(tree.planStatus->finalResult);
	printf("ExecuteQuery, %s, %d",query,len);
	printf("1 time, %.3f ms, \n",(l_endTime-l_startTime)*1000/ (double)CLOCKS_PER_SEC);
}

/*typedef enum{
	Q_POINT_SELECTION,
	Q_RANGE_SELECTION,
	Q_AGG,
	Q_ORDERBY,
	Q_AGG_GROUPBY,
	Q_NINLJ,
	Q_INLJ,
	Q_SMJ,
	Q_HJ,//9
	Q_DBMBENCH1,
	Q_DBMBENCH2,
	Q_DBMBENCH3,
}QUERY_TYPE;*/

void test_plan(int delta, int isGPUONLY_QP, int isAdaptive, int execMode)
{
	
	initDB2("RS.conf");
	char* query=new char[512];
	QUERY_TYPE qID=Q_POINT_SELECTION;
	qID=(QUERY_TYPE)(delta+Q_POINT_SELECTION);
	EXEC_MODE eM=(EXEC_MODE)(execMode+EXEC_CPU);
	if(qID==Q_INLJ)
	{			
		easedb->sortTable("R", "R.a00");
		easedb->createTreeIndex("R", "R.a00");
	}
	else if(qID==Q_AGG_GROUPBY)
	{
		int i=0;
		int id=easedb->getTableID("R.b00");
		Record* bb=easedb->tables[id];
		int rLen=easedb->tPro[id].rLen;
		for(i=0;i<rLen;i++)
			bb[i].value=bb[i].value%64;

	}
	makeTestQuery(qID,query);
	execQuery(query,isGPUONLY_QP,isAdaptive,eM);
	if(qID==Q_INLJ)
		easedb->removeTreeIndex("R", "R.a00");
}




//here we start to test the engine.
void testMicroBenchmark(int scale,int delta, int isGPUONLY_QP, int isAdaptive, int execMode)
{
	easedb=new Database();
	easedb->dbmbench(scale);
	easedb->sortTable("T1", "T1.a1");
	easedb->createTreeIndex("T1", "T1.a1");
	QUERY_TYPE qID=Q_DBMBENCH1;
	qID=(QUERY_TYPE)(delta+Q_DBMBENCH1);
	EXEC_MODE eM=(EXEC_MODE)(execMode+EXEC_CPU);
	char* query=new char[512];
	makeTestQuery(qID,query);
	execQuery(query,isGPUONLY_QP,isAdaptive,eM);
}


QUERY_TYPE makeRandomQuery(QUERY_TYPE fromType, QUERY_TYPE toType, char *query)
{
	QUERY_TYPE result=fromType;
	int offset=(int)(toType-fromType+1);
	int randomValue=RAND(offset);
	result=(QUERY_TYPE)(fromType+randomValue);
	makeTestQuery(result,query);
	return result;
}

void makeTestQuery(QUERY_TYPE qt, char *query)
{
	switch(qt)
	{
	case Q_POINT_SELECTION:
		{
			int a=RAND(TEST_MAX);
			cout<<"point: "<<a<<endl;
			sprintf(query, "PRO;R;$;R.a00,;$;:SEL;R;$;R.a00,;=,R.a00,$,$,%d,$,$,;:$:$:$:", a);
		}break;
	case Q_RANGE_SELECTION:
		{
			int a=RAND(TEST_MAX);
			int b=RAND(TEST_MAX);
			int minmin=min(a,b);
			int maxmax=minmin+20000000;//max(a,b);
			//result: 155943, 10000000
			//result: 311916, 20000000
			cout<<"range: ["<<minmin<<", "<<maxmax<<"]"<<endl;
			sprintf(query, "PRO;R;$;R.a00,;$;:SEL;R;$;R.a00,;AND,>,R.a00,$,$,%d,$,$,<,R.a00,$,$,%d,$,$,;:$:$:$:", minmin, maxmax);
		}break;
	case Q_AGG:
		{
			sprintf(query, "AGG;R;MAX;R.a00,;$;:$:$:");
		}break;
	case Q_ORDERBY:
		{
			sprintf(query, "AGG;R;MAX;R.a00,;$;:ORD;R;$;R.a00,;$;:$:$:$:");
			//sprintf(query, "PRO;$;$;R.b00,;$;:ORD;R;$;R.a00,;$;:$:$:$:");
			//sprintf(query, "PRO;R;$;R.b00,;$;:ORD;R;$;R.a00,;$;:SEL;R;$;R.b00,;<,R.b00,$,$,8,$,$,;:$:$:$:$:");
		}break;
	case Q_AGG_GROUPBY:
		{
			sprintf(query, "AGG;R;MAX;R.a00,;$;:GRP;R;$;R.b00,;$;:$:$:$:");
		}break;
	case Q_NINLJ:
		{
			//sprintf(query, "JOIN;R;S;R.a00,S.a00,;<,R.a00,$,$,S.a00,$,$,;:$:$:");
			/*int a1=RAND(TEST_MAX);
			int a2=RAND(TEST_MAX);
			int minmin1=a1;
			int maxmax1=minmin1+20000000;//max(a,b);
			int minmin2=a2;
			int maxmax2=minmin2+20000000;//max(a,b);
			sprintf(query, "PRO;R;$;R.a00,;$;:JOIN;R;S;R.a00,S.a00,;<,R.a00,$,$,S.a00,$,$,;:SEL;R;$;R.a00,;AND,>,R.a00,$,$,%d,$,$,<,R.a00,$,$,%d,$,$,;:$:$:SEL;S;$;S.a00,;AND,>,S.a00,$,$,%d,$,$,<,S.a00,$,$,%d,$,$,;:$:$:$:",minmin1,maxmax1,minmin2,maxmax2);*/
			cout<<"I have changed the NLJ for testing"<<endl;
			int a1=RAND(TEST_MAX);
			int a2=RAND(TEST_MAX);
			int delta=(int)(20*1000000.0);
			int minmin1=a1;
			int maxmax1=minmin1+delta;//max(a,b);
			int minmin2=a2;
			int maxmax2=minmin2+delta;//max(a,b);
			sprintf(query, "PRO;R;$;R.a00,;$;:JOIN;R;S;R.a00,S.a00,;<,R.a00,$,$,S.a00,$,$,;:SEL;R;$;R.a00,;AND,>,R.a00,$,$,%d,$,$,<,R.a00,$,$,%d,$,$,;:$:$:SEL;S;$;S.a00,;AND,>,S.a00,$,$,%d,$,$,<,S.a00,$,$,%d,$,$,;:$:$:$:",minmin1,maxmax1,minmin2,maxmax2);
		}break;
	case Q_INLJ:
		{
			sprintf(query,"PRO;R;$;R.a00,;$;:JOIN;R;S;R.a00,S.a00,;=,R.a00,$,$,S.a00,$,$,;:$:$:$:");
		}break;
	case Q_SMJ:
		{
			sprintf(query,"PRO;R;$;R.a00,;$;:JOIN;R;S;R.a00,S.a00,;=,R.a00,$,$,S.a00,$,$,;:$:$:$:");
		}break;
	case Q_HJ:
		{
			sprintf(query,"PRO;R;$;R.a00,;$;:JOIN;R;S;R.a00,S.a00,;=,R.a00,$,$,S.a00,$,$,;:$:$:$:");
		}break;
	case Q_DBMBENCH1:
		{
			int a=RAND(20000);
			int b=(a-2000);
			if(b<0) b=a+2000;
			int minmin=min(a,b);
			int maxmax=max(a,b);
			sprintf(query,"PRO;$;$;T1.a3,;$;:SEL;$;T1;T1.a2,;AND,<,%d,$,$,T1.a2,$,$,<,T1.a2,$,$,%d,$,$,;:$:$:$:",minmin,maxmax);
		}break;
	case Q_DBMBENCH2:
		{
			int a=RAND(20000);
			int b=(a-4000);
			if(b<0) b=a+4000;
			int minmin=min(a,b);
			int maxmax=max(a,b);
			sprintf(query,"AGG;T1;AVG;T1.a3,;$;:JOIN;T1;T2;T1.a1,T2.a1,;=,T1.a1,$,$,T2.a1,$,$,;:SEL;$;T1;T1.a2,;AND,<,%d,$,$,T1.a2,$,$,<,T1.a2,$,$,%d,$,$,;:$:$:$:$:",minmin,maxmax);
			//sprintf(query,"PRO;T1;$;T1.a3,;$;:JOIN;T1;T2;T1.a1,T2.a1,;=,T1.a1,$,$,T2.a1,$,$,;:$:$:$:");
		}break;
	case Q_DBMBENCH3:
		{
			int a=RAND(20000);
			int b=(a-100);
			if(b<0) b=a+100;
			int minmin=min(a,b);
			int maxmax=max(a,b);
			sprintf(query,"AGG;T1;AVG;T1.a3,;$;:SEL;$;T1;T1.a2,;AND,<,%d,$,$,T1.a2,$,$,<,T1.a2,$,$,%d,$,$,;:$:$:$:",minmin,maxmax);
		}break;
	}
	__DEBUG__(query);
}


void GPUDEBUG_Int(int* d_RIDList, int RIDLen, bool GPUONLY_QP)
{
#ifdef DEBUG
if( GPUONLY_QP)
{
	int tempSize=sizeof(int)*RIDLen;
	int* temp=NULL;
	CPUAllocateByCUDA((void**)&temp,tempSize);
	CopyGPUToCPU(temp,d_RIDList,tempSize);
	int i=0;
	cout<<"GPUDEBUG_Int"<<endl;
	for(i=0;i<RIDLen;i++)
		cout<<temp[i]<<"; ";
	cout<<endl;
	CPUFreeByCUDA(temp);
}
else
{
	cout<<"GPUDEBUG_Int"<<endl;
	int i=0;
	for(i=0;i<RIDLen;i++)
		cout<<d_RIDList[i]<<"; ";
	cout<<endl;
}
#endif
}
void GPUDEBUG_Record(Record* d_RIDList, int RIDLen, bool GPUONLY_QP)
{
#ifdef DEBUG
if( GPUONLY_QP)
{
	int tempSize=sizeof(Record)*RIDLen;
	Record* temp=NULL;
	CPUAllocateByCUDA((void**)&temp,tempSize);
	CopyGPUToCPU(temp,d_RIDList,tempSize);
	int i=0;
	cout<<"GPUDEBUG_Record: "<<RIDLen<<endl;
	for(i=0;i<RIDLen;i++)
		cout<<temp[i].rid<<","<<temp[i].value<<"; ";
	cout<<endl;
	CPUFreeByCUDA(temp);
}
else
{
	cout<<"GPUDEBUG_Record"<<endl;
	int i=0;
	for(i=0;i<RIDLen;i++)
		cout<<d_RIDList[i].rid<<","<<d_RIDList[i].value<<"; ";
	cout<<endl;
}
#endif
}


char* OpToString(OP_MODE op,EXEC_MODE eM)
{
	switch(eM)
	{
	case EXEC_CPU:
		{
			switch(op)
			{
			case SELECTION:
				return "SELECTION, on the CPU";
			case TYPE_AGGREGATION:
				return "TYPE_AGGREGATION, on the CPU";
			case AGG_SUM:
				return "AGG_SUM, on the CPU";
			case AGG_MAX:
				return "AGG_MAX, on the CPU";
			case AGG_MIN:
				return "AGG_MIN, on the CPU";
			case AGG_AVG:
				return "AGG_AVG, on the CPU";
			case AGG_COUNT:
				return "AGG_COUNT, on the CPU";
			case GROUP_BY:
				return "GROUP_BY, on the CPU";
			case AGG_SUM_AFTER_GROUP_BY:
				return "AGG_SUM_AFTER_GROUP_BY, on the CPU";
			case AGG_MAX_AFTER_GROUP_BY:
				return "AGG_MAX_AFTER_GROUP_BY, on the CPU";
			case AGG_AVG_AFTER_GROUP_BY:
				return "AGG_AVG_AFTER_GROUP_BY, on the CPU";
			case AGG_COUNT_AFTER_GROUP_BY:
				return "AGG_COUNT_AFTER_GROUP_BY, on the CPU";
			case ORDER_BY:
				return "ORDER_BY, on the CPU";
			case SORT:
				return "SORT, on the CPU";
			case PROJECTION:
				return "PROJECTION, on the CPU";
			case TYPE_JOIN:
				return "TYPE_JOIN, on the CPU";
			case JOIN_NINLJ:
				return "JOIN_NINLJ, on the CPU";
			case JOIN_INLJ:
				return "JOIN_INLJ, on the CPU";
			case JOIN_SMJ:
				return "JOIN_SMJ, on the CPU";
			case JOIN_HJ:
				return "JOIN_HJ, on the CPU";
			case DISTINCT:
				return "DISTINCT, on the CPU";
			default:
				return "TYPE_UNKNOWN, on the CPU";
			}
		}break;
	case EXEC_GPU:
		{
			switch(op)
			{
			case SELECTION:
				return "SELECTION, on the GPU";
			case TYPE_AGGREGATION:
				return "TYPE_AGGREGATION, on the GPU";
			case AGG_SUM:
				return "AGG_SUM, on the GPU";
			case AGG_MAX:
				return "AGG_MAX, on the GPU";
			case AGG_MIN:
				return "AGG_MIN, on the GPU";
			case AGG_AVG:
				return "AGG_AVG, on the GPU";
			case AGG_COUNT:
				return "AGG_COUNT, on the GPU";
			case GROUP_BY:
				return "GROUP_BY, on the GPU";
			case AGG_SUM_AFTER_GROUP_BY:
				return "AGG_SUM_AFTER_GROUP_BY, on the GPU";
			case AGG_MAX_AFTER_GROUP_BY:
				return "AGG_MAX_AFTER_GROUP_BY, on the GPU";
			case AGG_AVG_AFTER_GROUP_BY:
				return "AGG_AVG_AFTER_GROUP_BY, on the GPU";
			case AGG_COUNT_AFTER_GROUP_BY:
				return "AGG_COUNT_AFTER_GROUP_BY, on the GPU";
			case ORDER_BY:
				return "ORDER_BY, on the GPU";
			case SORT:
				return "SORT, on the GPU";
			case PROJECTION:
				return "PROJECTION, on the GPU";
			case TYPE_JOIN:
				return "TYPE_JOIN, on the GPU";
			case JOIN_NINLJ:
				return "JOIN_NINLJ, on the GPU";
			case JOIN_INLJ:
				return "JOIN_INLJ, on the GPU";
			case JOIN_SMJ:
				return "JOIN_SMJ, on the GPU";
			case JOIN_HJ:
				return "JOIN_HJ, on the GPU";
			case DISTINCT:
				return "DISTINCT, on the GPU";
			default:
				return "TYPE_UNKNOWN, on the GPU";
			}

		}break;
	case EXEC_CPU_GPU:
		{
			switch(op)
			{
			case SELECTION:
				return "SELECTION, on the CPU AND GPU";
			case TYPE_AGGREGATION:
				return "TYPE_AGGREGATION, on the CPU AND GPU";
			case AGG_SUM:
				return "AGG_SUM, on the CPU AND GPU";
			case AGG_MAX:
				return "AGG_MAX, on the CPU AND GPU";
			case AGG_MIN:
				return "AGG_MIN, on the CPU AND GPU";
			case AGG_AVG:
				return "AGG_AVG, on the CPU AND GPU";
			case AGG_COUNT:
				return "AGG_COUNT, on the CPU AND GPU";
			case GROUP_BY:
				return "GROUP_BY, on the CPU AND GPU";
			case AGG_SUM_AFTER_GROUP_BY:
				return "AGG_SUM_AFTER_GROUP_BY, on the CPU AND GPU";
			case AGG_MAX_AFTER_GROUP_BY:
				return "AGG_MAX_AFTER_GROUP_BY, on the CPU AND GPU";
			case AGG_AVG_AFTER_GROUP_BY:
				return "AGG_AVG_AFTER_GROUP_BY, on the CPU AND GPU";
			case AGG_COUNT_AFTER_GROUP_BY:
				return "AGG_COUNT_AFTER_GROUP_BY, on the CPU AND GPU";
			case ORDER_BY:
				return "ORDER_BY, on the CPU AND GPU";
			case SORT:
				return "SORT, on the CPU AND GPU";
			case PROJECTION:
				return "PROJECTION, on the CPU AND GPU";
			case TYPE_JOIN:
				return "TYPE_JOIN, on the CPU AND GPU";
			case JOIN_NINLJ:
				return "JOIN_NINLJ, on the CPU AND GPU";
			case JOIN_INLJ:
				return "JOIN_INLJ, on the CPU AND GPU";
			case JOIN_SMJ:
				return "JOIN_SMJ, on the CPU AND GPU";
			case JOIN_HJ:
				return "JOIN_HJ, on the CPU AND GPU";
			case DISTINCT:
				return "DISTINCT, on the CPU AND GPU";
			default:
				return "TYPE_UNKNOWN, on the CPU AND GPU";
			}
		}break;
	case EXEC_GPU_ONLY:
		{
			switch(op)
			{
			case SELECTION:
				return "SELECTION, GPU only";
			case TYPE_AGGREGATION:
				return "TYPE_AGGREGATION, GPU only";
			case AGG_SUM:
				return "AGG_SUM, GPU only";
			case AGG_MAX:
				return "AGG_MAX, GPU only";
			case AGG_MIN:
				return "AGG_MIN, GPU only";
			case AGG_AVG:
				return "AGG_AVG, GPU only";
			case AGG_COUNT:
				return "AGG_COUNT, GPU only";
			case GROUP_BY:
				return "GROUP_BY, GPU only";
			case AGG_SUM_AFTER_GROUP_BY:
				return "AGG_SUM_AFTER_GROUP_BY, GPU only";
			case AGG_MAX_AFTER_GROUP_BY:
				return "AGG_MAX_AFTER_GROUP_BY, GPU only";
			case AGG_AVG_AFTER_GROUP_BY:
				return "AGG_AVG_AFTER_GROUP_BY, GPU only";
			case AGG_COUNT_AFTER_GROUP_BY:
				return "AGG_COUNT_AFTER_GROUP_BY, GPU only";
			case ORDER_BY:
				return "ORDER_BY, GPU only";
			case SORT:
				return "SORT, GPU only";
			case PROJECTION:
				return "PROJECTION, GPU only";
			case TYPE_JOIN:
				return "TYPE_JOIN, GPU only";
			case JOIN_NINLJ:
				return "JOIN_NINLJ, GPU only";
			case JOIN_INLJ:
				return "JOIN_INLJ, GPU only";
			case JOIN_SMJ:
				return "JOIN_SMJ, GPU only";
			case JOIN_HJ:
				return "JOIN_HJ, GPU only";
			case DISTINCT:
				return "DISTINCT, GPU only";
			default:
				return "TYPE_UNKNOWN, GPU only";
			}

		}break;
	case EXEC_ADAPTIVE:
		{
			switch(op)
			{
			case SELECTION:
				return "SELECTION, Adaptive";
			case TYPE_AGGREGATION:
				return "TYPE_AGGREGATION, Adaptive";
			case AGG_SUM:
				return "AGG_SUM, Adaptive";
			case AGG_MAX:
				return "AGG_MAX, Adaptive";
			case AGG_MIN:
				return "AGG_MIN, Adaptive";
			case AGG_AVG:
				return "AGG_AVG, Adaptive";
			case AGG_COUNT:
				return "AGG_COUNT, Adaptive";
			case GROUP_BY:
				return "GROUP_BY, Adaptive";
			case AGG_SUM_AFTER_GROUP_BY:
				return "AGG_SUM_AFTER_GROUP_BY, Adaptive";
			case AGG_MAX_AFTER_GROUP_BY:
				return "AGG_MAX_AFTER_GROUP_BY, Adaptive";
			case AGG_AVG_AFTER_GROUP_BY:
				return "AGG_AVG_AFTER_GROUP_BY, Adaptive";
			case AGG_COUNT_AFTER_GROUP_BY:
				return "AGG_COUNT_AFTER_GROUP_BY, Adaptive";
			case ORDER_BY:
				return "ORDER_BY, Adaptive";
			case SORT:
				return "SORT, Adaptive";
			case PROJECTION:
				return "PROJECTION, Adaptive";
			case TYPE_JOIN:
				return "TYPE_JOIN, Adaptive";
			case JOIN_NINLJ:
				return "JOIN_NINLJ, Adaptive";
			case JOIN_INLJ:
				return "JOIN_INLJ, Adaptive";
			case JOIN_SMJ:
				return "JOIN_SMJ, Adaptive";
			case JOIN_HJ:
				return "JOIN_HJ, Adaptive";
			case DISTINCT:
				return "DISTINCT, Adaptive";
			default:
				return "TYPE_UNKNOWN, Adaptive";
			}

		}break;
	
	}
}


void setHighPriority()
{
	SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_ABOVE_NORMAL);
	cout<<"setHighPriority"<<endl;
}