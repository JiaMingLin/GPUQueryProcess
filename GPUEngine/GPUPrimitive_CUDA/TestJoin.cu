#ifndef TEST_JOIN_CU
#define TEST_JOIN_CU

#include "CUDA_ninlj.cu"
#include "CSSTree.cu"
#include "CUDA_inlj.cu"
#include "getMax.cu"
#include "CUDA_smj.cu"
#include "RadixClusteredHashJoin.cu"




void testNINLJ(int rLen, int sLen)
{
	int result=0;
	int memSizeR=sizeof(Record)*rLen;
	int memSizeS=sizeof(Record)*sLen;
	Record *h_R;
	CPUMALLOC((void**)&h_R, memSizeR);
	generateRand(h_R,TEST_MAX,rLen,0);
	Record *h_S;
	CPUMALLOC((void**)&h_S, memSizeS);
	generateRand(h_S, TEST_MAX,sLen,1);
	Record **h_Rout;
	CPUMALLOC((void**)&h_Rout,sizeof(Record*));
	
	startTime();
	unsigned int timer=0;
	startTimer(&timer);
	Record *d_R;
	GPUMALLOC((void**) & d_R, memSizeR);
	TOGPU(d_R, h_R,	memSizeR);
	Record *d_S;
	GPUMALLOC((void**) & d_S, memSizeS );
	TOGPU(d_S, h_S,	memSizeS);
	endTimer("copy to GPU",&timer);
	//ninlj
	startTimer(&timer);
	result=gpu_ninlj(d_R,rLen,d_S,sLen,h_Rout);
	double processingTime=endTimer("ninlj",&timer);

	
	double sec=endTime("ninlj");
	//gpuPrint(d_Rout, rLen, "d_Rout");
	printf("rLen, %d, sLen, %d, result, %d\n", rLen, sLen, result);	
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	double kernel_bandwidth=dataSize/processingTime;
	printf("\n dataSize, %f MB, bandwidth, %f, kernel_bandwidth, %f\n", dataSize, bandwidth, kernel_bandwidth);
	CPUFREE(h_Rout);
	CPUFREE(h_R);
	CPUFREE(h_S);
	GPUFREE(d_R);
	GPUFREE(d_S);
}


void testINLJ(int rLen, int sLen)
{
	int result=0;
	int memSizeR=sizeof(Record)*rLen;
	int memSizeS=sizeof(Record)*sLen;
	Record *h_R;
	CPUMALLOC((void**)&h_R, memSizeR);
	generateSort(h_R, TEST_MAX,rLen,0);
	CUDA_CSSTree* tree;
	
	unsigned int timer=0;
	Record *h_S;
	CPUMALLOC((void**)&h_S, memSizeS);
	generateRand(h_S, TEST_MAX,sLen,1);
	Record *d_Rout;
	
	startTime();
	startTimer(&timer);
	Record *d_R;
	GPUMALLOC((void**) & d_R, memSizeR);
	TOGPU(d_R, h_R,	memSizeR);
	endTimer("copy R to GPU",&timer);
	
	startTimer(&timer);
	//gpu_constructCSSTree(d_R, rLen, &tree);
	GPUOnly_BuildTreeIndex(d_R, rLen, &tree);
	endTimer("tree construction", &timer);

	startTimer(&timer);
	Record *d_S;
	GPUMALLOC((void**) & d_S, memSizeS );
	TOGPU(d_S, h_S,	memSizeS);
	endTimer("copy to GPU",&timer);

	//inlj
	startTimer(&timer);
	result=cuda_inlj(d_R,rLen,tree,d_S,sLen,&d_Rout);
	double processingTime=endTimer("inlj",&timer);

	Record* h_Rout = (Record*)malloc( sizeof(Record)*result );
	FROMGPU( h_Rout, d_Rout, sizeof(Record)*result );

	double sec=endTime("inlj");
	//gpuPrint(d_Rout, rLen, "d_Rout");
	printf("rLen, %d, sLen, %d, result, %d\n", rLen, sLen, result);	
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	double kernel_bandwidth=dataSize/processingTime;
	printf("\n dataSize, %f MB, bandwidth, %f, kernel_bandwidth, %f\n", dataSize, bandwidth, kernel_bandwidth);
	CPUFREE(h_Rout);
	CPUFREE(h_R);
	CPUFREE(h_S);
	GPUFREE(d_R);
	GPUFREE(d_S);
}

void testMax(int rLen)
{
	//int result=0;
	int memSizeR=sizeof(Record)*rLen;
	Record *h_R;
	CPUMALLOC((void**)&h_R, memSizeR);
	generateSort(h_R, rLen,rLen,0);
	unsigned int timer=0;
	startTimer(&timer);
	Record *d_R;
	GPUMALLOC((void**) & d_R, memSizeR);
	TOGPU(d_R, h_R,	memSizeR);
	printf("max, %d\n", getMax(d_R,rLen));
	endTimer("get max",&timer);
	CPUFREE(h_R);
	GPUFREE(d_R);
}


void testSMJ(int rLen, int sLen)
{
	int result=0;
	int memSizeR=sizeof(Record)*rLen;
	int memSizeS=sizeof(Record)*sLen;
	Record *h_R;
	CPUMALLOC((void**)&h_R, memSizeR);
	generateRand(h_R, TEST_MAX,rLen,0);
	Record *h_S;
	CPUMALLOC((void**)&h_S, memSizeS);
	generateRand(h_S, TEST_MAX,sLen,1);
	Record **h_Rout;
	CPUMALLOC((void**)&h_Rout,sizeof(Record*));
	
	startTime();
	unsigned int timer=0;
	startTimer(&timer);
	Record *d_R;
	GPUMALLOC((void**) & d_R, memSizeR);
	TOGPU(d_R, h_R,	memSizeR);
	Record *d_S;
	GPUMALLOC((void**) & d_S, memSizeS );
	TOGPU(d_S, h_S,	memSizeS);
	endTimer("copy to GPU",&timer);
	//ninlj
	startTimer(&timer);
	result=cuda_smj(d_R,rLen,d_S,sLen,h_Rout);
	double processingTime=endTimer("smj",&timer);

	startTimer(&timer);
	Record * Joinout;
	CPUMALLOC((void**)&Joinout, sizeof(Record)*result);
	FROMGPU(Joinout,*h_Rout,sizeof(Record)*result);
	endTimer("copy back",&timer);

	double sec=endTime("smj");
	//gpuPrint(d_Rout, rLen, "d_Rout");
	printf("rLen, %d, sLen, %d, result, %d\n", rLen, sLen, result);	
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	double kernel_bandwidth=dataSize/processingTime;
	printf("\n dataSize, %f MB, bandwidth, %f, kernel_bandwidth, %f\n", dataSize, bandwidth, kernel_bandwidth);
	CPUFREE(h_Rout);
	CPUFREE(h_R);
	CPUFREE(h_S);
	CPUFREE(Joinout);
/*#ifdef SHARED_MEM
	GPUFREE(d_R);
	GPUFREE(d_S);
#endif*/
}


void testMatch(int rLen, int sLen)
{
	int result=0;
	int memSizeR=sizeof(Record)*rLen;
	int memSizeS=sizeof(Record)*sLen;
	Record *h_R;
	CPUMALLOC((void**)&h_R, memSizeR);
	generateRand(h_R,TEST_MAX,rLen,0);
	Record *h_S;
	CPUMALLOC((void**)&h_S, memSizeS);
	generateRand(h_S, TEST_MAX,sLen,1);
	Record **h_Rout;
	CPUMALLOC((void**)&h_Rout,sizeof(Record*));
	
	startTime();
	unsigned int timer=0;
	startTimer(&timer);
	Record *d_R;
	GPUMALLOC((void**) & d_R, memSizeR);
	TOGPU(d_R, h_R,	memSizeR);
	Record *d_S;
	GPUMALLOC((void**) & d_S, memSizeS );
	TOGPU(d_S, h_S,	memSizeS);
	endTimer("copy to GPU",&timer);
	//ninlj
	startTimer(&timer);
	result=matchingBlocks(d_R,rLen,d_S,sLen,h_Rout);
	double processingTime=endTimer("match",&timer);

	
	double sec=endTime("match");
	//gpuPrint(d_Rout, rLen, "d_Rout");
	printf("rLen, %d, result, %d\n", rLen, result);	
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	double kernel_bandwidth=dataSize/processingTime;
	printf("\n dataSize, %f MB, bandwidth, %f, kernel_bandwidth, %f\n", dataSize, bandwidth, kernel_bandwidth);
	CPUFREE(h_Rout);
	CPUFREE(h_R);
	CPUFREE(h_S);
	GPUFREE(d_R);
	GPUFREE(d_S);
}


void testHJ(int rLen, int sLen)
{
#ifdef BINARY_SEARCH_HASH
	printf("Binary search on Hash join\n");
#else
	printf("No Binary search on Hash join\n");
#endif
	int result=0;
	int memSizeR=sizeof(Record)*rLen;
	int memSizeS=sizeof(Record)*sLen;
	Record *h_R;
	CPUMALLOC((void**)&h_R, memSizeR);
	generateRand(h_R, TEST_MAX,rLen,0);
	Record *h_S;
	CPUMALLOC((void**)&h_S, memSizeS);
	generateRand(h_S, TEST_MAX,sLen,1);
	Record **h_Rout;
	CPUMALLOC((void**)&h_Rout,sizeof(Record*));
	
	startTime();
	unsigned int timer=0;
	startTimer(&timer);
	Record *d_R;
	GPUMALLOC((void**) & d_R, memSizeR);
	TOGPU(d_R, h_R,	memSizeR);
	Record *d_S;
	GPUMALLOC((void**) & d_S, memSizeS );
	TOGPU(d_S, h_S,	memSizeS);
	endTimer("copy to GPU",&timer);
	//ninlj
	startTimer(&timer);

	/******************
	 * CUDA Hash Join *
	 ******************/
	result=cuda_hj(d_R,rLen,d_S,sLen,h_Rout);
	double processingTime=endTimer("hj",&timer);

	startTimer(&timer);
	Record * Joinout;
	CPUMALLOC((void**)&Joinout, sizeof(Record)*result);
	FROMGPU(Joinout,*h_Rout,sizeof(Record)*result);
	endTimer("copy back",&timer);

	double sec=endTime("hj");
	//gpuPrint(d_Rout, rLen, "d_Rout");
	printf("rLen, %d, result, %d\n", rLen, result);	
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	double kernel_bandwidth=dataSize/processingTime;
	printf("\n dataSize, %f MB, bandwidth, %f, kernel_bandwidth, %f\n", dataSize, bandwidth, kernel_bandwidth);
	CPUFREE(h_Rout);
	CPUFREE(h_R);
	CPUFREE(h_S);
//	GPUFREE(d_R);
//	GPUFREE(d_S);
}


void testNINLJ_2(Record* h_R, int rLen, Record* h_S, int sLen)
{
	int result=0;
	int memSizeR=sizeof(Record)*rLen;
	int memSizeS=sizeof(Record)*sLen;
	Record **h_Rout;
	CPUMALLOC((void**)&h_Rout,sizeof(Record*));
	
	startTime();
	unsigned int timer=0;
	startTimer(&timer);
	Record *d_R;
	GPUMALLOC((void**) & d_R, memSizeR);
	TOGPU(d_R, h_R,	memSizeR);
	Record *d_S;
	GPUMALLOC((void**) & d_S, memSizeS );
	TOGPU(d_S, h_S,	memSizeS);
	endTimer("copy to GPU",&timer);
	//ninlj
	startTimer(&timer);
	result=gpu_ninlj(d_R,rLen,d_S,sLen,h_Rout);
	double processingTime=endTimer("ninlj",&timer);

	
	double sec=endTime("ninlj2");
	//gpuPrint(d_Rout, rLen, "d_Rout");
	printf("rLen, %d, sLen, %d, result, %d\n", rLen, sLen, result);	
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	double kernel_bandwidth=dataSize/processingTime;
	printf("\n dataSize, %f MB, bandwidth, %f, kernel_bandwidth, %f\n", dataSize, bandwidth, kernel_bandwidth);
	CPUFREE(h_Rout);
	GPUFREE(d_R);
	GPUFREE(d_S);
}


void testINLJ_2(Record* h_R,int rLen, Record* h_S,  int sLen)
{
	int result=0;
	int memSizeR=sizeof(Record)*rLen;
	int memSizeS=sizeof(Record)*sLen;
	qsort(h_R,rLen,sizeof(Record),compare);
	CUDA_CSSTree* tree;
	
	unsigned int timer=0;
	Record **h_Rout;
	CPUMALLOC((void**)&h_Rout,sizeof(Record*));
	
	startTime();
	startTimer(&timer);
	Record *d_R;
	GPUMALLOC((void**) & d_R, memSizeR);
	TOGPU(d_R, h_R,	memSizeR);
	endTimer("copy R to GPU",&timer);
	
	startTimer(&timer);
	gpu_constructCSSTree(d_R, rLen, &tree);
	endTimer("tree construction", &timer);

	startTimer(&timer);
	Record *d_S;
	GPUMALLOC((void**) & d_S, memSizeS );
	TOGPU(d_S, h_S,	memSizeS);
	endTimer("copy to GPU",&timer);

	
	//ninlj
	startTimer(&timer);
	result=cuda_inlj(d_R,rLen,tree,d_S,sLen,h_Rout);
	double processingTime=endTimer("inlj",&timer);

	
	double sec=endTime("inlj2");
	//gpuPrint(d_Rout, rLen, "d_Rout");
	printf("rLen, %d, sLen, %d, result, %d\n", rLen, sLen, result);	
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	double kernel_bandwidth=dataSize/processingTime;
	printf("\n dataSize, %f MB, bandwidth, %f, kernel_bandwidth, %f\n", dataSize, bandwidth, kernel_bandwidth);
	CPUFREE(h_Rout);
	GPUFREE(d_R);
	GPUFREE(d_S);
}

void testSMJ_2(Record *h_R, int rLen, Record *h_S, int sLen)
{
	int result=0;
	int memSizeR=sizeof(Record)*rLen;
	int memSizeS=sizeof(Record)*sLen;
	Record **h_Rout;
	CPUMALLOC((void**)&h_Rout,sizeof(Record*));
	
	unsigned int timer=0;
	startTime();
	startTimer(&timer);
	Record *d_R;
	GPUMALLOC((void**) & d_R, memSizeR);
	TOGPU(d_R, h_R,	memSizeR);
	Record *d_S;
	GPUMALLOC((void**) & d_S, memSizeS );
	TOGPU(d_S, h_S,	memSizeS);
	endTimer("copy to GPU",&timer);
	//ninlj
	startTimer(&timer);
	result=cuda_smj(d_R,rLen,d_S,sLen,h_Rout);
	double processingTime=endTimer("smj",&timer);

	startTimer(&timer);
	Record * Joinout=NULL;
	if(result!=0)
	{
		CPUMALLOC((void**)&Joinout, sizeof(Record)*result);
		FROMGPU(Joinout,*h_Rout,sizeof(Record)*result);
	}
	endTimer("copy back",&timer);

	double sec=endTime("smj2");
	//gpuPrint(d_Rout, rLen, "d_Rout");
	printf("rLen, %d, sLen, %d, result, %d\n", rLen, sLen, result);	
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	double kernel_bandwidth=dataSize/processingTime;
	printf("\n dataSize, %f MB, bandwidth, %f, kernel_bandwidth, %f\n", dataSize, bandwidth, kernel_bandwidth);
	CPUFREE(h_Rout);
	CPUFREE(Joinout);
}


void testHJ_2(Record *h_R, int rLen, Record *h_S, int sLen)
{
#ifdef BINARY_SEARCH_HASH
	printf("Binary search on Hash join\n");
#else
	printf("No Binary search on Hash join\n");
#endif
	int result=0;
	int memSizeR=sizeof(Record)*rLen;
	int memSizeS=sizeof(Record)*sLen;
	Record **h_Rout;
	CPUMALLOC((void**)&h_Rout,sizeof(Record*));
	
	startTime();
	unsigned int timer=0;
	startTimer(&timer);
	Record *d_R;
	GPUMALLOC((void**) & d_R, memSizeR);
	TOGPU(d_R, h_R,	memSizeR);
	Record *d_S;
	GPUMALLOC((void**) & d_S, memSizeS );
	TOGPU(d_S, h_S,	memSizeS);
	endTimer("copy to GPU",&timer);
	//ninlj
	startTimer(&timer);
	result=cuda_hj(d_R,rLen,d_S,sLen,h_Rout);
	double processingTime=endTimer("hj",&timer);

	
	double sec=endTime("hj");
	//gpuPrint(d_Rout, rLen, "d_Rout");
	printf("rLen, %d, result, %d\n", rLen, result);	
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	double kernel_bandwidth=dataSize/processingTime;
	printf("\n dataSize, %f MB, bandwidth, %f, kernel_bandwidth, %f\n", dataSize, bandwidth, kernel_bandwidth);
	CPUFREE(h_Rout);
//	GPUFREE(d_R);
//	GPUFREE(d_S);
}

void testSkew(int rLen, int sLen, int mode, double oneRatio)
{
	//percentage of ones.
	//sort-merge join
	//int memSizeR=sizeof(Record)*rLen;
	//int memSizeS=sizeof(Record)*sLen;
	Record *R;
	CPUMALLOC((void**)&R, rLen*sizeof(Record));
	Record *S;
	CPUMALLOC((void**)&S, sLen*sizeof(Record));
	generateSkew(R,TEST_MAX,rLen,oneRatio,0);		
//	generateSkew(S,TEST_MAX,sLen,0.0000001,1);
	generateRand(S,TEST_MAX,sLen,1);
	if(mode==0)
	{
		printf("testNINLJ oneRatio, %f, S \n", oneRatio);
		testNINLJ_2(R,rLen,S,sLen);
	}
	if(mode==1)
	{
		printf("testINLJ oneRatio, %f, S \n", oneRatio);
		testINLJ_2(R,rLen,S,sLen);
	}
	if(mode==2)
	{
		printf("testSMJ oneRatio, %f, S \n", oneRatio);
		testSMJ_2(R,rLen,S,sLen);
	}
	if(mode==3)
	{
		printf("testSMJ oneRatio, %f, S \n", oneRatio);
		testHJ_2(R,rLen,S,sLen);
	}

	CPUFREE(R);
	CPUFREE(S);
	//GPUFREE(d_R);
	//GPUFREE(d_S);

}

void testSel(int rLen, int sLen, int mode, float joinSel)
{
	//int memSizeR=sizeof(Record)*rLen;
	//int memSizeS=sizeof(Record)*sLen;
	Record *R;
	CPUMALLOC((void**)&R, rLen*sizeof(Record));
	Record *S;
	CPUMALLOC((void**)&S, sLen*sizeof(Record));
	generateJoinSelectivity(R,rLen,S,sLen,TEST_MAX,joinSel,0);
	if(mode==0)
	{
		printf("testNINLJ joinSel, %f, S \n", joinSel);
		testNINLJ_2(R,rLen,S,sLen);
	}
	if(mode==1)
	{
		printf("testINLJ joinSel, %f, S \n", joinSel);
		testINLJ_2(R,rLen,S,sLen);
	}
	if(mode==2)
	{
		printf("testSMJ joinSel, %f, S \n", joinSel);
		testSMJ_2(R,rLen,S,sLen);
	}
	if(mode==3)
	{
		printf("testSMJ joinSel, %f, S \n", joinSel);
		testHJ_2(R,rLen,S,sLen);
	}
	CPUFREE(R);
	CPUFREE(S);
	//GPUFREE(d_R);
	//GPUFREE(d_S);

}


void testMicroJoin(int tL)
{
	if(tL<=512*1024)
	testNINLJ(tL, tL);
	testINLJ(tL,tL);
	testSMJ(tL,tL);
	testHJ(tL,tL);
}

int testAllJoin(int argc, char ** argv)
{
	int i=0;
	for(i=0;i<argc;i++)
		printf("%s ", argv[i]);
	printf("\n");
	//int tL=1024*1024*8;
	//testNINLJ(tL, tL);
	//testINLJ(tL,tL);
	//testMax(tL);
	//testSMJ(tL,tL);
	//testHJ(tL,tL);
	//testMatch(tL,tL);

	for(i=0;i<argc;i++)
	{
		if(strcmp(argv[i], "-microJoin")==0)
		{
			if(argc==(i+2))
			{
				int rLen=atoi(argv[i+1])*1024*1024;
				testMicroJoin(rLen);
			}
		}
		if(strcmp(argv[i], "-ninlj")==0)
		{
			int rLen=256*1024;
			int sLen=256*1024;
			if(argc==(i+3))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				sLen=atoi(argv[i+2])*1024*1024;
			}
			testNINLJ(rLen,sLen);
		}

		if(strcmp(argv[i], "-inlj")==0)
		{
			int rLen=256*1024;
			int sLen=256*1024;
			if(argc==(i+3))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				sLen=atoi(argv[i+2])*1024*1024;
			}
			testINLJ(rLen,sLen);
		}
		
		if(strcmp(argv[i], "-smj")==0)
		{
			int rLen=256*1024;
			int sLen=256*1024;
			if(argc==(i+3))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				sLen=atoi(argv[i+2])*1024*1024;
			}
			testSMJ(rLen,sLen);
		}

		if(strcmp(argv[i], "-hj")==0)
		{
			int rLen=256*1024;
			int sLen=256*1024;
			if(argc==(i+3))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				sLen=atoi(argv[i+2])*1024*1024;
			}
			testHJ(rLen,sLen);
		}

		if(strcmp(argv[i], "-sel")==0)
		{
			int rLen=256*1024;
			int sLen=256*1024;
			float joinSel=0;
			if(argc==(i+5))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				sLen=atoi(argv[i+2])*1024*1024;
				int mode=atoi(argv[i+3]);
				joinSel=atof(argv[i+4]);
				testSel(rLen,sLen,mode,joinSel);
			}
			
		}
		if(strcmp(argv[i], "-skew")==0)
		{
			int rLen=256*1024;
			int sLen=256*1024;
			float oneRatio=0;
			if(argc==(i+5))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				sLen=atoi(argv[i+2])*1024*1024;
				int mode=atoi(argv[i+3]);
				oneRatio=atof(argv[i+4]);
				testSkew(rLen,sLen,mode, oneRatio);
			}
			
		}
	}
	return 0;
}


#endif

