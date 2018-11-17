#include "SortThreadOp.h"
#include "CPU_Dll.h"
#include "GPU_Dll.h"
#include "CoProcessor.h"

SortThreadOp::SortThreadOp(OP_MODE opt)
:SingularThreadOp(opt)
{
	
}

void SortThreadOp::init(Record *p_R, int p_rLen, bool pGPUONLY_QP)
{
	R=p_R;
	rLen=p_rLen;
	GPUONLY_QP=pGPUONLY_QP;	
}

SortThreadOp::~SortThreadOp(void)
{
}


void SortThreadOp::execute(EXEC_MODE eM)
{
	clock_t tt=0;
	startTimer(tt);
if( GPUONLY_QP || eM==EXEC_GPU_ONLY)
{
	ON_GPUONLY("SortThreadOp::execute");
	GPUAllocate((void**)&Rout,sizeof(Record)*rLen);
	GPUOnly_bitonicSort(R,rLen,Rout);
	//GPUOnly_QuickSort(R,rLen,Rout);
	ON_GPUONLY_DONE("SortThreadOp::execute");
	numResult=rLen;
}
else
{
	Rout=new Record[rLen];
	if(eM==EXEC_CPU)
	{
		ON_CPU("SortThreadOp::execute");
		CPU_Sort(R,rLen,Rout,OMP_SORT_NUM_THREAD);
		ON_CPU_DONE("SortThreadOp::execute");
	}
	else if(eM==EXEC_GPU)
	{
		ON_GPU("SortThreadOp::execute");
		GPUCopy_bitonicSort(R,rLen,Rout);
		//GPUCopy_QuickSort(R,rLen,Rout);
		ON_GPU_DONE("SortThreadOp::execute");
	}	
	else 
	{
		ON_CPU_GPU("SortThreadOp::execute");		
		CO_Sort(R,rLen,Rout);		
		ON_CPU_GPU_DONE("SortThreadOp::execute");
	}
	numResult=rLen;	
}//endif GPUONLY_QP
	endTimer("total sort",tt);
}


ThreadOp* SortThreadOp::getNextOp(EXEC_MODE eM)
{
	this->execMode=eM;
	isFinished=true;
	return this;
}
