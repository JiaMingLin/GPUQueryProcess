#include "BinaryThreadOp.h"
#include "CPU_Dll.h"
#include "GPU_Dll.h"
#include "CoProcessor.h"

BinaryThreadOp::BinaryThreadOp(OP_MODE opt)
{
	optType=opt;
}


void BinaryThreadOp::init(Record *p_R, int p_rLen, Record* p_S, int p_sLen, bool pGPUONLY_QP)
{
	R=p_R;
	rLen=p_rLen;
	S=p_S;
	sLen=p_sLen;
	GPUONLY_QP=pGPUONLY_QP;
}

BinaryThreadOp::~BinaryThreadOp(void)
{
}


void BinaryThreadOp::execute(EXEC_MODE eM)
{
if(GPUONLY_QP||eM==EXEC_GPU_ONLY)
{
	ON_GPUONLY("BinaryThreadOp::execute");
	switch(optType)
	{
		case JOIN_NINLJ:
			{
				//GPUDEBUG_Record(R,rLen);
				//GPUDEBUG_Record(S,sLen);
				numResult=GPUOnly_ninlj(R,rLen,S,sLen,&Rout);					
			}break;
		case JOIN_INLJ:
			{
				cout<<"you should not be here"<<endl;
			}break;
		case JOIN_SMJ:
			{
				numResult=GPUOnly_smj(R,rLen,S,sLen,&Rout);
			}break;
		case JOIN_HJ:
			{
				numResult=GPUOnly_hj(R,rLen,S,sLen,&Rout);
			}break;
	}
	ON_GPUONLY_DONE("BinaryThreadOp::execute");
}
else
{
	if(eM==EXEC_CPU)
	{
		ON_CPU("BinaryThreadOp::execute");
		switch(optType)
		{
			case JOIN_NINLJ:
				{
					numResult=CPU_ninlj(R,rLen,S,sLen,&Rout,OMP_JOIN_NUM_THREAD);					
				}break;
			case JOIN_INLJ:
				{
					cout<<"you should not be here"<<endl;
				}break;
			case JOIN_SMJ:
				{
					numResult=CPU_smj(R,rLen,S,sLen,&Rout,OMP_JOIN_NUM_THREAD);
				}break;
			case JOIN_HJ:
				{
					numResult=CPU_hj(R,rLen,S,sLen,&Rout,OMP_JOIN_NUM_THREAD);
				}break;
		}
		ON_CPU_DONE("BinaryThreadOp::execute");
	}
	else if(eM==EXEC_GPU)
	{
		ON_GPU("BinaryThreadOp::execute");
		switch(optType)
		{
			case JOIN_NINLJ:
				{
					numResult=GPUCopy_ninlj(R,rLen,S,sLen,&Rout);					
				}break;
			case JOIN_INLJ:
				{
					cout<<"you should not be here"<<endl;
				}break;
			case JOIN_SMJ:
				{
					numResult=GPUCopy_smj(R,rLen,S,sLen,&Rout);
				}break;
			case JOIN_HJ:
				{
					numResult=GPUCopy_hj(R,rLen,S,sLen,&Rout);
				}break;
		}
		ON_GPU_DONE("BinaryThreadOp::execute");
	}
	else//co-processing
	{
		ON_CPU_GPU("BinaryThreadOp::execute");
		switch(optType)
		{
			case JOIN_NINLJ:
				{
					numResult=CO_ninlj(R,rLen,S,sLen,&Rout);					
				}break;
			case JOIN_INLJ:
				{
					cout<<"you should not be here"<<endl;
				}break;
			case JOIN_SMJ:
				{
					numResult=CO_Smj(R,rLen,S,sLen,&Rout);
				}break;
			case JOIN_HJ:
				{
					//numResult=CO_Smj(R,rLen,S,sLen,&Rout);
					numResult=CO_hj(R,rLen,S,sLen,&Rout);
				}break;
		}
	}
}

}

ThreadOp* BinaryThreadOp::getNextOp(EXEC_MODE eM)
{
	this->execMode=eM;
	isFinished=true;
	return this;
}


/*
* for the indexed nlj.
*/
IndexJoinThreadOp::IndexJoinThreadOp(OP_MODE opt):BinaryThreadOp(opt)
{
	cpu_tree=NULL;
	gpu_tree=NULL;
}


void IndexJoinThreadOp::init(Record *p_R, int p_rLen,  CC_CSSTree *cT,  CUDA_CSSTree* gT, Record* p_S, int p_sLen, 
									 bool pGPUONLY_QP)
{
	R=p_R;
	rLen=p_rLen;
	S=p_S;
	sLen=p_sLen;
	GPUONLY_QP=pGPUONLY_QP;
	cpu_tree=cT;
	gpu_tree=gT;
}

IndexJoinThreadOp::~IndexJoinThreadOp(void)
{

}

void IndexJoinThreadOp::execute(EXEC_MODE eM)
{
if( GPUONLY_QP || eM==EXEC_GPU_ONLY)
{
	//ON_GPUONLY("GPUOnly_BuildTreeIndex");
	//GPUFree(gpu_tree);
	//GPUOnly_BuildTreeIndex(R,rLen,&(gpu_tree));
	//ON_GPUONLY_DONE("GPUOnly_BuildTreeIndex");
	ON_GPUONLY("IndexJoinThreadOp::execute");
	numResult=GPUOnly_inlj(R,rLen,gpu_tree,S,sLen,&Rout);
	ON_GPUONLY_DONE("IndexJoinThreadOp::execute");
}
else
{
	switch(eM)
	{
		case EXEC_CPU:
		{
			ON_CPU("IndexJoinThreadOp::execute");
			numResult=CPU_inlj(R,rLen,cpu_tree,S,sLen,&Rout,OMP_JOIN_NUM_THREAD);
			ON_CPU_DONE("IndexJoinThreadOp::execute");
		}break;
		case EXEC_GPU:
		{
			ON_GPU("IndexJoinThreadOp::execute");
			numResult=GPUCopy_inlj(R,rLen,gpu_tree,S,sLen,&Rout);
			ON_GPU_DONE("IndexJoinThreadOp::execute");
		}break;
		case EXEC_CPU_GPU:
		{
			ON_CPU_GPU("IndexJoinThreadOp::execute");
			numResult=CO_inlj(R,rLen,cpu_tree,gpu_tree,S,sLen,&Rout);
		}break;
	}
}
//cout<<"resultSize#,"<<numResult<<endl;
}


ThreadOp* IndexJoinThreadOp::getNextOp(EXEC_MODE eM)
{
	this->execMode=eM;
	isFinished=true;
	return this;
}
