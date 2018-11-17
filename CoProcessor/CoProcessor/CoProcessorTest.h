#ifndef CO_PROCESSOR_TEST_H
#define CO_PROCESSOR_TEST_H

#include "windows.h"
#include "CoProcessor.h"
#include "CoProcessorTest.h"

typedef enum{
	Q_POINT_SELECTION,
	Q_RANGE_SELECTION,
	Q_AGG,
	Q_ORDERBY,
	Q_AGG_GROUPBY,
	Q_NINLJ,
	Q_INLJ,
	Q_SMJ,
	Q_HJ,
	Q_DBMBENCH1,
	Q_DBMBENCH2,
	Q_DBMBENCH3,
}QUERY_TYPE;

struct Query_stat{
	bool isAssigned;
	EXEC_MODE eM;
	float timeInSec;
	QUERY_TYPE qT;
	float speedupGPUoverCPU;
	bool needCop;
};



struct tp_batchQuery{
	HANDLE dispatchMutex;
	int* curQueryID;
	EXEC_MODE eM;
	int totalQuery;
	char** sqlQuery;
	Query_stat* stat;
	int* numThreadActive;
	int threadid;
	void init(HANDLE pMutex, int* pcurQueryID, EXEC_MODE peM, int ptotalQuery,
		char** psqlQuery, Query_stat* pstat, int *pnumThreadActive, int pthreadid)
	{
		dispatchMutex=pMutex;
		curQueryID=pcurQueryID;
		eM=peM;
		totalQuery=ptotalQuery;
		sqlQuery=psqlQuery;
		stat=pstat;
		numThreadActive=pnumThreadActive;
		threadid=pthreadid;
	}
};

struct tp_singleQuery{
	char* query;
	EXEC_MODE eM;
	int id;
	int postThreadID;
	void init(EXEC_MODE peM, char* pquery, int pid, int ppostThreadID)
	{
		eM=peM;
		query=pquery;
		id=pid;
		postThreadID=ppostThreadID;
	}
};

void test_plan(int delta, int isGPUONLY_QP, int isAdaptive, int execMode);
void initDB2(char *confFile);
void testMicroBenchmark(int scale, int delta, int isGPUONLY_QP, int isAdaptive, int execMode);
void makeTestQuery(QUERY_TYPE qt, char *query);
QUERY_TYPE makeRandomQuery(QUERY_TYPE fromType, QUERY_TYPE toType, char *query);
void execQuery(char *query, bool isGPUONLY_QP, bool isAdaptive, EXEC_MODE eM);


void testQueryProcessor(QUERY_TYPE fromType, QUERY_TYPE toType, int numQuery, EXEC_MODE eM, int numWorkerThread=2, bool isEx=false);

float getSpeedUP(int qid);
bool getNeedCop(int qid);


int pickQuerySmart(Query_stat *gQstat, int numQuery, EXEC_MODE toBeEM, float thresholdForGPUApp=2.0, float thresholdForCPUApp=0.5);
void testMbench(int numQuery, EXEC_MODE eM, int numThread, int scale);

#endif

