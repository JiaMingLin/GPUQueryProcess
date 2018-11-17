#ifndef QUERYPLANTREE_H
#define QUERYPLANTREE_H

#include "db.h"
#include "stdlib.h"
#include "QueryPlanNode.h"
#include "ExecStatus.h"
#include "ThreadOp.h"
#include <vector>
using namespace std;


class QueryPlanTree
{
protected:
	

public:
	void buildTree(char * str);
	ExecStatus* planStatus;
	void execute(EXEC_MODE eM=EXEC_CPU);
	QueryPlanTree(bool pGPUONLY_QP)
	{
		GPUONLY_QP=pGPUONLY_QP;
		root=NULL;
		planStatus=(ExecStatus*)malloc(sizeof(ExecStatus));
		planStatus->init(GPUONLY_QP);
		hasLock=false;
	}
	~QueryPlanTree();
	QueryPlanNode * QueryPlanTree::construct_plan_tree(char * str, int * index);
	QueryPlanNode * root;
	ThreadOp* getNextOp(EXEC_MODE eM);
	vector<QueryPlanNode *> nodeVec;
	int curActiveNode;//in the vector.
	int totalNumNode;
	void Marshup(QueryPlanNode * node);
	Record *q_Rout;
	int q_numResult;
	bool GPUONLY_QP;
	bool hasLock;

	
	

};

#endif

