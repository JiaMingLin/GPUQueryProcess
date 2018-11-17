#include "SingularThreadOp.h"
#include "CPU_Dll.h"
#include "GPU_Dll.h"

SingularThreadOp::SingularThreadOp(OP_MODE opt)
{
	optType=opt;
	R=NULL;
	rLen=-1;
	GPUONLY_QP=false;
}

void SingularThreadOp::init(Record *p_R, int p_rLen, bool pGPUONLY_QP)
{
	R=p_R;
	rLen=p_rLen;
	GPUONLY_QP=pGPUONLY_QP;
}

SingularThreadOp::~SingularThreadOp(void)
{
}


void SingularThreadOp::execute(EXEC_MODE eM)
{
if(GPUONLY_QP||eM==EXEC_GPU_ONLY)
{
	int result=0;
	ON_GPUONLY("SingularThreadOp::execute");
	switch(optType)
	{
		case AGG_SUM:
			{
				result=GPUOnly_AggSum(R,rLen,&Rout);					
			}break;
		case AGG_MAX:
			{
				result=GPUOnly_AggMax(R,rLen,&Rout);
			}break;
		case AGG_AVG:
			{
				if(rLen==0)
					result=0;
				else
					result=GPUOnly_AggAvg(R,rLen,&Rout);
			}break;
		case AGG_MIN:
			{
				result=GPUOnly_AggMin(R,rLen,&Rout);
			}break;
		case SELECTION:
			{
				SelectionOp *sop=(SelectionOp*)this;
				sop->execute(eM);
			}break;
	}
	numResult=1;
	ON_GPUONLY_DONE("SingularThreadOp::execute");
}
else
{
	if(eM==EXEC_CPU || eM==EXEC_CPU_GPU)
	{
		int result=0;
		ON_CPU("SingularThreadOp::execute");
		switch(optType)
		{
			case AGG_SUM:
				{
					result=CPU_AggSum(R,rLen,1);
					
				}break;
			case AGG_MAX:
				{
					result=CPU_AggMax(R,rLen,1);
				}break;
			case AGG_AVG:
				{
					if(rLen==0)
						result=0;
					else
						result=CPU_AggAvg(R,rLen,1);
				}break;
			case AGG_MIN:
				{
					result=CPU_AggMin(R,rLen,1);
				}break;
			case SELECTION:
				{
					SelectionOp *sop=(SelectionOp*)this;
					sop->execute(eM);
				}break;
		}
		Rout=new Record[1];
		(Rout)[0].value=result;
		(Rout)[0].rid=1;
		numResult=1;
		ON_CPU_DONE("SingularThreadOp::execute");
	}
	else if(eM==EXEC_GPU)
	{
		int result=0;
		ON_GPU("SingularThreadOp::execute");
		switch(optType)
		{
			case AGG_SUM:
				{
					result=GPUCopy_AggSum(R,rLen,&Rout);					
				}break;
			case AGG_MAX:
				{
					result=GPUCopy_AggMax(R,rLen,&Rout);
				}break;
			case AGG_AVG:
				{
					if(rLen==0)
						result=0;
					else
						result=GPUCopy_AggAvg(R,rLen,&Rout);
				}break;
			case AGG_MIN:
				{
					result=GPUCopy_AggMin(R,rLen,&Rout);
				}break;
			case SELECTION:
				{
					SelectionOp *sop=(SelectionOp*)this;
					sop->execute(eM);
				}break;
		}
		numResult=1;
		ON_GPU_DONE("SingularThreadOp::execute");
	}
}//else of GPUONLY_QP.
}

ThreadOp* SingularThreadOp::getNextOp(EXEC_MODE eM)
{
	this->execMode=eM;
	isFinished=true;
	return this;
}



SelectionOp::SelectionOp(OP_MODE opt):
SingularThreadOp(opt)
{
	
}

void SelectionOp::init(Record *p_R, int p_rLen, int p_lowerKey, int p_higherKey, bool pGPUONLY_QP)
{
	R=p_R;
	rLen=p_rLen;
	GPUONLY_QP=pGPUONLY_QP;
	lowerKey=p_lowerKey;
	higherKey=p_higherKey;
}

void SelectionOp::execute(EXEC_MODE eM)
{
if(GPUONLY_QP||eM==EXEC_GPU_ONLY)
{
	ON_GPUONLY("SelectionOp::execute");
	if(lowerKey==higherKey)
	{
		numResult=GPUOnly_PointSelection(R,rLen,lowerKey, &Rout);
	}
	else
	{
		numResult=GPUOnly_RangeSelection(R,rLen,lowerKey, higherKey,&Rout);
	}
	ON_GPUONLY_DONE("SelectionOp::execute");
}
else
{
	if(eM==EXEC_CPU || eM==EXEC_CPU_GPU)
	{
		ON_CPU("SelectionOp::execute");
		if(lowerKey==higherKey)
		{
			numResult=CPU_PointSelection(R,rLen,lowerKey, &Rout, 1);
		}
		else
		{
			numResult=CPU_RangeSelection(R,rLen,lowerKey, higherKey,&Rout, 1);
		}
		ON_CPU_DONE("SelectionOp::execute");
	}
	else
	{//execute on the GPU.
		ON_GPU("SelectionOp::execute");
		if(lowerKey==higherKey)
		{
			numResult=GPUCopy_PointSelection(R,rLen,lowerKey, &Rout);
		}
		else
		{
			numResult=GPUCopy_RangeSelection(R,rLen,lowerKey, higherKey,&Rout);
		}
		ON_GPU_DONE("SelectionOp::execute");
	}
}
	__DEBUGInt__(numResult);
}


ThreadOp* SelectionOp::getNextOp(EXEC_MODE eM)
{
	this->execMode=eM;
	isFinished=true;
	return this;
}

/*
* Projection operator
*/
void ProjectionOp::execute(EXEC_MODE eM)
{
if(GPUONLY_QP||eM==EXEC_GPU_ONLY)
{
	ON_GPUONLY("ProjectionOp::execute");
	numResult=RIDLen;
	if(RIDLen>0)
	{
		GPUAllocate((void**)&Rout, sizeof(Record)*RIDLen);
		GPUOnly_setRIDList(RIDList,RIDLen,Rout);
		GPUOnly_Projection(R,rLen,Rout,RIDLen);	
		//GPUDEBUG_Record(Rout, RIDLen);
	}
	ON_GPUONLY_DONE("ProjectionOp::execute");
}
else
{
	if(eM==EXEC_CPU || eM==EXEC_CPU_GPU)
	{
		ON_CPU("ProjectionOp::execute");
		this->Rout=new Record[RIDLen];
		this->numResult=RIDLen;
		int i=0;
		for(i=0;i<RIDLen;i++)
			Rout[i].rid=RIDList[i];
		CPU_Projection(R,rLen,Rout,RIDLen,1);
		numResult=RIDLen;
		ON_CPU_DONE("ProjectionOp::execute");
	}
	else
	{
		ON_GPU("ProjectionOp::execute");
		this->Rout=new Record[RIDLen];
		this->numResult=RIDLen;
		int i=0;
		for(i=0;i<RIDLen;i++)
			Rout[i].rid=RIDList[i];
		GPUCopy_Projection(R,rLen,Rout,RIDLen);
		numResult=RIDLen;
		ON_GPU_DONE("ProjectionOp::execute");
	}
}
}
ProjectionOp::ProjectionOp(OP_MODE opt):
SingularThreadOp(opt)
{
}

void ProjectionOp::init(Record *p_R, int p_rLen, int* pRIDList, int pRIDLen, bool pGPUONLY_QP)
{
	R=p_R;
	rLen=p_rLen;
	GPUONLY_QP=pGPUONLY_QP;
	RIDList=pRIDList;
	RIDLen=pRIDLen;
}
ThreadOp* ProjectionOp::getNextOp(EXEC_MODE eM)
{
	this->execMode=eM;
	isFinished=true;
	return this;
}