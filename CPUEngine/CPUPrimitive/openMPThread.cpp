#include "stdafx.h"
#include "CPU_Dll.h"

int gNumCore=3;

void set_thread_affinity (int id, int numThread) 
{
    #pragma omp parallel default(shared)
    {
		if(numThread!=1)
		{
			DWORD_PTR mask = (1 << id)%((1<<gNumCore)-1);
			cout<<mask<<endl;
			SetThreadAffinityMask( GetCurrentThread(), mask );
		}
    }
}


void set_CPU_affinity (int id, int numThread) 
{
	if(numThread!=1)
	{
		DWORD_PTR mask = (1 << id)%((1<<numThread)-1);
		//cout<<mask<<endl;
		SetThreadAffinityMask( GetCurrentThread(), mask );
	}
}

void set_selfCPUID (int id)
{
	DWORD_PTR mask = (1 << id)%((1<<NUM_CORE_PER_CPU)-1);
	SetThreadAffinityMask( GetCurrentThread(), mask );
	//SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_ABOVE_NORMAL);
}


void set_numCoreForCPU (int numCore) 
{
	gNumCore=numCore;
	cout<<"set the number of core for the CPU processing: "<<gNumCore<<endl;
}