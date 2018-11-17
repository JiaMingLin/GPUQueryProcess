#include "GroupByThreadOp.h"
#include "CPU_Dll.h"
#include "GPU_Dll.h"

GroupByThreadOp::GroupByThreadOp(OP_MODE opt):
SingularThreadOp(opt)
{
	numGroup=-1;
	startPos=NULL;
}

void GroupByThreadOp::init(Record *p_R, int p_rLen, bool pGPUONLY_QP)
{
	R=p_R;
	rLen=p_rLen;
	GPUONLY_QP=pGPUONLY_QP;	
	numGroup=-1;
	startPos=NULL;
}

GroupByThreadOp::~GroupByThreadOp(void)
{
}


void GroupByThreadOp::execute(EXEC_MODE eM)
{
if( GPUONLY_QP || eM==EXEC_GPU_ONLY)
{
	ON_GPUONLY("GroupByThreadOp::execute");
	GPUAllocate((void**)&Rout,sizeof(Record)*rLen);
	//GPUDEBUG_Record(R,rLen,GPUONLY_QP);
	numGroup=GPUOnly_GroupBy(R,rLen,Rout,&startPos);
	//GPUDEBUG_Record(Rout,rLen,GPUONLY_QP);
	
	numResult=rLen;
	ON_GPUONLY_DONE("GroupByThreadOp::execute");
}
else
{
	if(eM==EXEC_CPU || eM==EXEC_CPU_GPU)
	{
		ON_CPU("GroupByThreadOp::execute");
		Rout=new Record[rLen];
		numGroup=CPU_GroupBy(R,rLen,Rout,&startPos,1);
		numResult=rLen;
		ON_CPU_DONE("GroupByThreadOp::execute");
	}
	else
	{
		ON_GPU("GroupByThreadOp::execute");
		Rout=new Record[rLen];
		numGroup=GPUCopy_GroupBy(R,rLen,Rout,&startPos);
		numResult=rLen;
		ON_GPU_DONE("GroupByThreadOp::execute");
	}
}
cout<<"numGroup: "<<numGroup<<endl;
}

ThreadOp* GroupByThreadOp::getNextOp(EXEC_MODE eM)
{
	this->execMode=eM;
	isFinished=true;
	return this;
}


/*
* aggregation after group by.
*/

AggAfterGroupByThreadOp::AggAfterGroupByThreadOp(OP_MODE opt):
SingularThreadOp(opt)
{
	
}


void AggAfterGroupByThreadOp::init(Record *p_R, int p_rLen, Record* pRHavingGroupBy, 
		  int prLenHavingGroupBy, int numG, int* pStartPos, bool pGPUONLY_QP)
{
	R=p_R;
	rLen=p_rLen;
	GPUONLY_QP=pGPUONLY_QP;	
	numGroup=numG;
	startPos=pStartPos;
	RHavingGroupBy=pRHavingGroupBy;
	rLenHavingGroupBy=prLenHavingGroupBy;
}

AggAfterGroupByThreadOp::~AggAfterGroupByThreadOp(void)
{
}


void AggAfterGroupByThreadOp::execute(EXEC_MODE eM)
{
if(GPUONLY_QP || eM==EXEC_GPU_ONLY)
{
	ON_GPUONLY("AggAfterGroupByThreadOp::execute");
	int* tempResult=NULL;
	GPUAllocate((void**)&tempResult,sizeof(int)*numGroup);
	switch(optType)
	{
		case AGG_SUM_AFTER_GROUP_BY:
			{
				GPUOnly_agg_sum_afterGroupBy(RHavingGroupBy,rLenHavingGroupBy,startPos,numGroup,R,tempResult,1);					
			}break;
		case AGG_MAX_AFTER_GROUP_BY:
			{
				//GPUDEBUG_Record(RHavingGroupBy,rLenHavingGroupBy,GPUONLY_QP);
				//GPUDEBUG_Int(startPos,numGroup,GPUONLY_QP);
				GPUOnly_agg_max_afterGroupBy(RHavingGroupBy,rLenHavingGroupBy,startPos,numGroup,R,tempResult,1);
			}break;
		case AGG_AVG_AFTER_GROUP_BY:
			{
				GPUOnly_agg_avg_afterGroupBy(RHavingGroupBy,rLenHavingGroupBy,startPos,numGroup,R,tempResult,1);
			}break;
		case AGG_MIN_AFTER_GROUP_BY:
			{
				GPUOnly_agg_min_afterGroupBy(RHavingGroupBy,rLenHavingGroupBy,startPos,numGroup,R,tempResult,1);
			}break;
	}
	GPUAllocate((void**)&Rout,sizeof(Record)*numGroup);
	GPUOnly_setValueList(tempResult,numGroup,Rout);
	numResult=numGroup;
	GPUFree(tempResult);
	ON_GPUONLY_DONE("AggAfterGroupByThreadOp::execute");
}
else
{
	if(eM==EXEC_CPU || eM==EXEC_CPU_GPU)
	{
		ON_CPU("AggAfterGroupByThreadOp::execute");
		int* tempResult=new int[numGroup];
		switch(optType)
		{
			case AGG_SUM_AFTER_GROUP_BY:
				{
					CPU_agg_sum_afterGroupBy(RHavingGroupBy,rLenHavingGroupBy,startPos,numGroup,R,tempResult,1);					
				}break;
			case AGG_MAX_AFTER_GROUP_BY:
				{
					CPU_agg_max_afterGroupBy(RHavingGroupBy,rLenHavingGroupBy,startPos,numGroup,R,tempResult,1);
				}break;
			case AGG_AVG_AFTER_GROUP_BY:
				{
					CPU_agg_avg_afterGroupBy(RHavingGroupBy,rLenHavingGroupBy,startPos,numGroup,R,tempResult,1);
				}break;
			case AGG_MIN_AFTER_GROUP_BY:
				{
					CPU_agg_min_afterGroupBy(RHavingGroupBy,rLenHavingGroupBy,startPos,numGroup,R,tempResult,1);
				}break;
		}
		Rout=new Record[numGroup];
		int i=0;
		for(i=0;i<numGroup;i++)
		{
			Rout[i].rid=i;
			Rout[i].value=tempResult[i];
		}
		numResult=numGroup;
		delete tempResult;
		ON_CPU_DONE("AggAfterGroupByThreadOp::execute");
		
	}
	else
	{
		ON_GPU("AggAfterGroupByThreadOp::execute");
		int* tempResult=new int[numGroup];
		switch(optType)
		{
			case AGG_SUM_AFTER_GROUP_BY:
				{
					GPUCopy_agg_sum_afterGroupBy(RHavingGroupBy,rLenHavingGroupBy,startPos,numGroup,R,tempResult,1);					
				}break;
			case AGG_MAX_AFTER_GROUP_BY:
				{
					GPUCopy_agg_max_afterGroupBy(RHavingGroupBy,rLenHavingGroupBy,startPos,numGroup,R,tempResult,1);
				}break;
			case AGG_AVG_AFTER_GROUP_BY:
				{
					GPUCopy_agg_avg_afterGroupBy(RHavingGroupBy,rLenHavingGroupBy,startPos,numGroup,R,tempResult,1);
				}break;
			case AGG_MIN_AFTER_GROUP_BY:
				{
					GPUCopy_agg_min_afterGroupBy(RHavingGroupBy,rLenHavingGroupBy,startPos,numGroup,R,tempResult,1);
				}break;
		}
		Rout=new Record[numGroup];
		int i=0;
		for(i=0;i<numGroup;i++)
		{
			Rout[i].rid=i;
			Rout[i].value=tempResult[i];
		}
		numResult=numGroup;
		delete tempResult;
		ON_GPU_DONE("AggAfterGroupByThreadOp::execute");
	}
}
}

ThreadOp* AggAfterGroupByThreadOp::getNextOp(EXEC_MODE eM)
{
	this->execMode=eM;
	isFinished=true;
	return this;
}