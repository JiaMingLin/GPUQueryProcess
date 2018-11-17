#include "ThreadOp.h"
#include "CPU_Dll.h"
#include <iostream>
using namespace std;

ThreadOp::ThreadOp()
{
	isFinished=false;
	numResult=rLen;
	Rout=NULL;
}

ThreadOp::~ThreadOp()
{
	if(Rout!=NULL)
	{
		if( GPUONLY_QP)
			GPUFree(Rout);
		else
			delete Rout;
	}
}


