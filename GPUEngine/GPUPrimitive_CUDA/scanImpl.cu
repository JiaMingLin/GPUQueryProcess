#ifndef SCAN_IMPL_CU
#define SCAN_IMPL_CU

#include "scan.cu"

void scanImpl(int *d_input, int rLen, int *d_output)
{
	//saven_initialPrefixSum(rLen);
	preallocBlockSums(rLen);
	prescanArray(d_output, d_input, rLen);
	deallocBlockSums();
}


#endif

