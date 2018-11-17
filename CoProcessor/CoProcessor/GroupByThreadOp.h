#pragma once
#include "SortThreadOp.h"

class GroupByThreadOp: public SingularThreadOp
{
public:
	int numGroup;
	int* startPos;
	GroupByThreadOp(OP_MODE opt);
	void init(Record *p_R, int p_rLen, bool pGPUONLY_QP);
	~GroupByThreadOp(void);
	void execute(EXEC_MODE eM);
	ThreadOp* getNextOp(EXEC_MODE eM);
};


class AggAfterGroupByThreadOp: public SingularThreadOp
{
public:
	int numGroup;
	int* startPos;
	Record* RHavingGroupBy;
	int rLenHavingGroupBy;
	AggAfterGroupByThreadOp(OP_MODE opt);
	void init(Record *p_R, int p_rLen, Record* pRHavingGroupBy, int prLenHavingGroupBy,  int numG, int* pStartPos, bool pGPUONLY_QP);
	~AggAfterGroupByThreadOp(void);
	void execute(EXEC_MODE eM);
	ThreadOp* getNextOp(EXEC_MODE eM);
};
