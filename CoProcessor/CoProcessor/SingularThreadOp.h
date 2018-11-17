#pragma once
#include "threadop.h"

class SingularThreadOp :
	public ThreadOp
{
public:
	void init(Record *p_R, int p_rLen, bool pGPUONLY_QP);
	SingularThreadOp(OP_MODE opt);
	~SingularThreadOp(void);
	void execute(EXEC_MODE eM);
	ThreadOp* getNextOp(EXEC_MODE eM);
};

class SelectionOp: public SingularThreadOp
{
public:
	int lowerKey;
	int higherKey;
	void execute(EXEC_MODE eM);
	void init(Record *p_R, int p_rLen, int lowerKey, int higherKey,bool pGPUONLY_QP);
	SelectionOp(OP_MODE opt);
	ThreadOp* getNextOp(EXEC_MODE eM);
};

class ProjectionOp:public SingularThreadOp
{
public:
	int* RIDList;
	int RIDLen;
	void execute(EXEC_MODE eM);
	void init(Record *p_R, int p_rLen, int* RIDList, int RIDLen, bool pGPUONLY_QP);
	ProjectionOp(OP_MODE opt);
	ThreadOp* getNextOp(EXEC_MODE eM);
};
