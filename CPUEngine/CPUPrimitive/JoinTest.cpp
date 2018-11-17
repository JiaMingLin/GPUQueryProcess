#include "stdafx.h"
#include "MyThreadPool.h"
#include "Primitive.h"
#include "mapImpl.h"
#include "common.h"
#include "LinkedList.h"
#include "Test.h"
#include <iostream>
using namespace std;
#include "Joins.h"


void testNinlj(int rLen, int sLen, int numThread)
{
	int result=0;
	Record *R=new Record[rLen];
	generateRand(R, TEST_MAX,rLen,0);

	Record *S=new Record[sLen];
	generateRand(S, TEST_MAX,sLen,1);
	
	Record **Rout=(Record**)malloc(sizeof(Record*));
	*Rout=NULL;
	startTime();
	result=ninlj_omp(R,rLen, S, sLen,Rout,numThread);
	double sec=endTime("ninlj");
	cout<<", rLen, "<<rLen<<", sLen, "<<sLen<<", numThreads, "<<numThread<<", sum, "<<result<<", "<<endl;
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	cout<<"\ngather bandwidth, "<< bandwidth<<" MB/sec"<<endl;
	delete R;
	delete S;
	if(*Rout!=NULL)
		delete *Rout;
	free(Rout);
}

void testInlj(int rLen, int sLen, int numThread)
{
	int result=0;
	Record *R=new Record[rLen];
	generateSort(R, TEST_MAX,rLen,0);
	CC_CSSTree *tree=new CC_CSSTree(R, rLen, CSS_TREE_FANOUT);
	Record *S=new Record[sLen];
	generateRand(S, TEST_MAX,sLen,1);
	
	Record **Rout=(Record**)malloc(sizeof(Record*));
	*Rout=NULL;
	startTime();
	result=inlj_omp(R,rLen,tree, S, sLen,Rout,numThread);
	double sec=endTime("inlj");
	cout<<", rLen, "<<rLen<<", sLen, "<<sLen<<", numThreads, "<<numThread<<", sum, "<<result<<", "<<endl;
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	cout<<"\ninlj bandwidth, "<< bandwidth<<" MB/sec"<<endl;
	delete R;
	delete S;
	if(*Rout!=NULL)
		delete *Rout;
	free(Rout);
	delete tree;
}


void testSearch(int rLen, int sLen, int numThread)
{
	int result=0;
	Record *R=new Record[rLen];
	generateSort(R, rLen,rLen,0);
	//cout<<findLargestSmaller(R,0,rLen,10);
	int numQuantile=4;
	int *quanPos=new int [numQuantile];
	findQuantile(R,0,rLen,numQuantile,quanPos);
	for(int i=0;i<numQuantile;i++)
		cout<<quanPos[i]<<", ";
	cout<<endl;
	
	delete R;
}


void testSmj(int rLen, int sLen, int numThread)
{
	int result=0;
	Record *R=new Record[rLen];
	generateRand(R, TEST_MAX,rLen,0);

	Record *S=new Record[sLen];
	generateRand(S, TEST_MAX,sLen,1);	
	Record **Rout=(Record**)malloc(sizeof(Record*));
	*Rout=NULL;
	startTime();
	result=smj_omp(R,rLen, S, sLen,Rout,numThread);
	double sec=endTime("smj");
	cout<<", rLen, "<<rLen<<", sLen, "<<sLen<<", numThreads, "<<numThread<<", sum, "<<result<<", "<<endl;
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	cout<<"\ngather bandwidth, "<< bandwidth<<" MB/sec"<<endl;
	delete R;
	delete S;
	if(*Rout!=NULL)
		delete *Rout;
	free(Rout);
}

void testHj(int rLen, int sLen, int numThread)
{
	int result=0;
	Record *R=new Record[rLen];
	generateRand(R, TEST_MAX,rLen,0);

	Record *S=new Record[sLen];
	generateRand(S, TEST_MAX,sLen,1);	
	Record **Rout=(Record**)malloc(sizeof(Record*));
	*Rout=NULL;
	startTime();
	result=hj_omp(R,rLen, S, sLen,Rout,numThread);
	double sec=endTime("Hj");
	cout<<", rLen, "<<rLen<<", sLen, "<<sLen<<", numThreads, "<<numThread<<", sum, "<<result<<", "<<endl;
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	cout<<"\ngather bandwidth, "<< bandwidth<<" MB/sec"<<endl;
	delete R;
	delete S;
	if(*Rout!=NULL)
		delete *Rout;
	free(Rout);
}

//input as the relations.
void testNinlj_2(Record *R, int rLen, Record*S, int sLen, int numThread=16)
{
	int result=0;
	
	Record **Rout=(Record**)malloc(sizeof(Record*));
	*Rout=NULL;
	startTime();
	result=ninlj_omp(R,rLen, S, sLen,Rout,numThread);
	double sec=endTime("ninlj");
	cout<<", rLen, "<<rLen<<", sLen, "<<sLen<<", numThreads, "<<numThread<<", sum, "<<result<<", "<<endl;
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	cout<<"\ngather bandwidth, "<< bandwidth<<" MB/sec"<<endl;
	if(*Rout!=NULL)
		delete *Rout;
	free(Rout);
}

void testInlj_2(Record *R, int rLen, Record*S, int sLen, int numThread=4)
{
	int result=0;
	qsort(R,rLen,sizeof(Record),compare);
	CC_CSSTree *tree=new CC_CSSTree(R, rLen, CSS_TREE_FANOUT);	
	Record **Rout=(Record**)malloc(sizeof(Record*));
	*Rout=NULL;
	startTime();
	result=inlj_omp(R,rLen,tree, S, sLen,Rout,numThread);
	double sec=endTime("inlj");
	cout<<", rLen, "<<rLen<<", sLen, "<<sLen<<", numThreads, "<<numThread<<", sum, "<<result<<", "<<endl;
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	cout<<"\ninlj bandwidth, "<< bandwidth<<" MB/sec"<<endl;
	if(*Rout!=NULL)
		delete *Rout;
	free(Rout);
	delete tree;
}

void testHj_2(Record *R, int rLen, Record*S, int sLen, int numThread=4)
{
	int result=0;
	Record **Rout=(Record**)malloc(sizeof(Record*));
	*Rout=NULL;
	startTime();
	result=hj_omp(R,rLen, S, sLen,Rout,numThread);
	double sec=endTime("Hj");
	cout<<", rLen, "<<rLen<<", sLen, "<<sLen<<", numThreads, "<<numThread<<", sum, "<<result<<", "<<endl;
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	cout<<"\ngather bandwidth, "<< bandwidth<<" MB/sec"<<endl;
	
	if(*Rout!=NULL)
		delete *Rout;
	free(Rout);
}

void testSmj_2(Record *R, int rLen, Record*S, int sLen, int numThread=8)
{
	int result=0;
	Record **Rout=(Record**)malloc(sizeof(Record*));
	*Rout=NULL;
	startTime();
	result=smj_omp(R,rLen, S, sLen,Rout,numThread);
	double sec=endTime("smj");
	cout<<", rLen, "<<rLen<<", sLen, "<<sLen<<", numThreads, "<<numThread<<", sum, "<<result<<", "<<endl;
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	cout<<"\ngather bandwidth, "<< bandwidth<<" MB/sec"<<endl;
	if(*Rout!=NULL)
		delete *Rout;
	free(Rout);
}

void testSkew(int rLen, int sLen, int mode, double oneRatio)
{
	//percentage of ones.
	//hash join
//	double oneRatio=0.01;
	Record *R=new Record[rLen];
	Record *S=new Record[sLen];	
	if(mode==0)
	{
		//for(oneRatio=0.01;oneRatio<=0.08;oneRatio=oneRatio*2)
		{
			printf("ninlj one ratio, %f\n", oneRatio);
			generateSkew(R,TEST_MAX,rLen,oneRatio,0);
			generateRand(S,TEST_MAX,sLen,1);
			testNinlj_2(R,rLen,S,sLen);
		}
	}
	if(mode==1)
	{
		//for(oneRatio=0.01;oneRatio<=0.08;oneRatio=oneRatio*2)
		{
			printf("inlj one ratio, %f\n", oneRatio);
			generateSkew(R,TEST_MAX,rLen,oneRatio,0);
			generateRand(S,TEST_MAX,sLen,1);
			testInlj_2(R,rLen,S,sLen);
		}
	}
	if(mode==2)
	{
		//for(oneRatio=0.01;oneRatio<=0.08;oneRatio=oneRatio*2)
		{
			printf("Hj one ratio, %f\n", oneRatio);
			generateSkew(R,TEST_MAX,rLen,oneRatio,0);
			generateRand(S,TEST_MAX,sLen,1);
			testHj_2(R,rLen,S,sLen);
		}
	}
	//sort-merge join
	if(mode==3)
	{
		//for(oneRatio=0.01;oneRatio<=0.08;oneRatio=oneRatio*2)
		{
			printf("smj one ratio, %f, S \n", oneRatio);		
			generateSkew(R,TEST_MAX,rLen,oneRatio,0);		
			generateRand(S,TEST_MAX,sLen,1);
			testSmj_2(R,rLen,S,sLen);
		}
	}
	delete R;
	delete S;

}

void testSel(int rLen, int sLen, int mode,float joinSel)
{
	Record* R=new Record[rLen];
	Record* S=new Record[sLen];
	//float joinSel=0;
	if(mode==0)
	{
		//for(joinSel=0.01;joinSel<=0.64;joinSel=joinSel*2)
		{
			printf("Ninlj joinSel, %f, \n", joinSel);
			generateJoinSelectivity(R,rLen,S,sLen,TEST_MAX,joinSel,0);
			testNinlj_2(R,rLen,S,sLen);
		}
	}
	
	if(mode==1)
	{
		//for(joinSel=0.01;joinSel<=0.64;joinSel=joinSel*2)
		{
			printf("Inlj joinSel, %f, S \n", joinSel);
			generateJoinSelectivity(R,rLen,S,sLen,TEST_MAX,joinSel,0);
			testInlj_2(R,rLen,S,sLen);
		}
	}
	if(mode==2)
	{
		//for(joinSel=0.01;joinSel<=0.64;joinSel=joinSel*2)
		{
			printf("Hj joinSel, %f, S \n", joinSel);
			generateJoinSelectivity(R,rLen,S,sLen,TEST_MAX,joinSel,0);
			testHj_2(R,rLen,S,sLen);
		}
	}
	if(mode==3)
	{
		//for(joinSel=0.01;joinSel<=0.64;joinSel=joinSel*2)
		{
			printf("Smj joinSel, %f, S \n", joinSel);
			generateJoinSelectivity(R,rLen,S,sLen,TEST_MAX,joinSel,0);
			testSmj_2(R,rLen,S,sLen);
		}
	}
	delete R;
	delete S;

}

void testJoin(int argc, char **argv)
{

	int i=0;
	//testNinlj(1024*16, 1024*16, 2);
	//testSmj(1024*1024*16, 1024*1024*16, 32);
	//testInlj(1024*1024, 1024*1024,1);
	//testSearch(1024, 1024, 1);
	//testHj(1024*16,1024*16,2);
	for(i=0;i<argc;i++)
	{
		if(strcmp(argv[i], "-ninlj")==0)
		{
			int rLen=256*1024;
			int sLen=256*1024;
			int numThread=8;
			if(argc==(i+4))
			{
				rLen=atoi(argv[i+1])*1024;
				sLen=atoi(argv[i+2])*1024;
				numThread=atoi(argv[i+3]);	
			}
			testNinlj(rLen, sLen, numThread);
		}

		if(strcmp(argv[i], "-inlj")==0)
		{
			int rLen=256*1024;
			int sLen=256*1024;
			int numThread=8;
			if(argc==(i+4))
			{
				rLen=atoi(argv[i+1])*1024;
				sLen=atoi(argv[i+2])*1024;
				numThread=atoi(argv[i+3]);	
			}
			testInlj(rLen, sLen, numThread);
		}


		if(strcmp(argv[i], "-smj")==0)
		{
			int rLen=256*1024;
			int sLen=256*1024;
			int numThread=8;
			if(argc==(i+4))
			{
				rLen=atoi(argv[i+1])*1024;
				sLen=atoi(argv[i+2])*1024;
				numThread=atoi(argv[i+3]);	
				testSmj(rLen, sLen, numThread);
			}
		}

		if(strcmp(argv[i], "-hj")==0)
		{
			int rLen=256*1024;
			int sLen=256*1024;
			int numThread=8;
			if(argc==(i+4))
			{
				rLen=atoi(argv[i+1])*1024;
				sLen=atoi(argv[i+2])*1024;
				numThread=atoi(argv[i+3]);	
				testHj(rLen, sLen, numThread);
			}
		}
		if(strcmp(argv[i], "-skew")==0)
		{
			int rLen=256*1024;
			int sLen=256*1024;
			int numThread=8;
			if(argc==(i+5))
			{
				rLen=atoi(argv[i+1])*1024;
				sLen=atoi(argv[i+2])*1024;
				int mode=atoi(argv[i+3]);
				double oneRatio=atof(argv[i+4]);
				testSkew(rLen, sLen, mode, oneRatio);
			}
		}
		if(strcmp(argv[i], "-sel")==0)
		{
			int rLen=256*1024;
			int sLen=256*1024;
			int numThread=8;
			if(argc==(i+5))
			{
				rLen=atoi(argv[i+1])*1024;
				sLen=atoi(argv[i+2])*1024;
				int mode=atoi(argv[i+3]);
				double joinSel=atof(argv[i+4]);
				testSel(rLen, sLen, mode, joinSel);
			}
		}
	}
}