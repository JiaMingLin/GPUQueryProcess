#ifndef THREAD_OP_H
#define THREAD_OP_H
#pragma once
#include "CoProcessor.h"
#include "common.h"
#include "db.h"
/*
we define the operator type in this file. Acutally, it corresponds to the 
queryPlanNode. The major difference here is: the queryPlanNode is for an entire operator, the ThreadOp
can be for a suboperator after the division. It is performed on the CPU or the GPU.
*/
class ThreadOp
{
public:
	Record* R;
	int rLen;
	Record *Rout;
	int numResult;
	EXEC_MODE execMode;
	OP_MODE optType;
	bool isFinished;
	ThreadOp();
	~ThreadOp();	
	virtual void execute(EXEC_MODE eM)=0;
	virtual ThreadOp* getNextOp(EXEC_MODE eM)=0;
	bool isDone() {return isFinished;}

	bool GPUONLY_QP;
};



#endif