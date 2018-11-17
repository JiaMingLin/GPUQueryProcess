#pragma once
#include "threadop.h"
#include "CPU_Dll.h"

class BinaryThreadOp :
	public ThreadOp
{
public:
	Record *S;
	int sLen;
	BinaryThreadOp(OP_MODE opt);
	void init(Record *p_R, int p_rLen, Record* p_S, int p_sLen, bool GPUONLY_QP);
	~BinaryThreadOp(void);
	void execute(EXEC_MODE eM);
	ThreadOp* getNextOp(EXEC_MODE eM);
};

class IndexJoinThreadOp: public BinaryThreadOp
{
public:
	CC_CSSTree *cpu_tree;
	CUDA_CSSTree *gpu_tree;
	IndexJoinThreadOp(OP_MODE opt);
	void init(Record *p_R, int p_rLen, CC_CSSTree *cT,  CUDA_CSSTree* gt, Record* p_S, int p_sLen,  bool GPUONLY_QP);
	~IndexJoinThreadOp(void);
	void execute(EXEC_MODE eM);
	ThreadOp* getNextOp(EXEC_MODE eM);

};
