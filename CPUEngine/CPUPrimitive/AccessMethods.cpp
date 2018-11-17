#include "stdafx.h"
#include "AccessMethods.h"
#include "common.h"
#include "hashTable.h"
#include "Joins.h"


void testTreeSearch(int rLen, int sLen, int numThread)
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
	double sec=endTime("tree search");
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
void testHashSearch(int rLen, int sLen, int numThread)
{
	int result=0;
	Record *R=new Record[rLen];
	generateSort(R, TEST_MAX,rLen,0);
	double bits=log2((double)rLen);
	int intBits=(int)bits;
	if(bits-intBits>=0.0000001)
		intBits++;
	intBits=intBits-1;//each bucket 8 tuples.
	int listLen=(1<<intBits);
	Bound * h_bound=(Bound *)malloc(sizeof(Bound)*listLen);
	buildHashTable(R,rLen,intBits,h_bound);
	Record *S=new Record[sLen];
	generateRand(S, TEST_MAX,sLen,1);	
	Record **Rout=(Record**)malloc(sizeof(Record*));
	*Rout=NULL;
	startTime();
	result=HashSearch_omp(R,rLen,h_bound, listLen,S, sLen,Rout,numThread);
	double sec=endTime("hash search");
	cout<<", rLen, "<<rLen<<", sLen, "<<sLen<<", numThreads, "<<numThread<<", sum, "<<result<<", "<<endl;
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	cout<<"\ninlj bandwidth, "<< bandwidth<<" MB/sec"<<endl;
	delete R;
	delete S;
	if(*Rout!=NULL)
		delete *Rout;
	free(Rout);
	free( h_bound);
}
void testAccessMethods(int argc, char **argv)
{
	int i=0;
	for(i=0;i<argc;i++)
	{
		if(strcmp(argv[i],"-TreeSearch")==0)
		{
			int rLen=8*1024*1024;
			int sLen=8*1024*1024;
			int numThread=4;
			if(argc==(i+4))
			{
				rLen=atoi(argv[i+1])*1024;
				sLen=atoi(argv[i+2])*1024;
				numThread=atoi(argv[i+3]);
			}
			testTreeSearch(rLen,sLen,numThread);
		}

		if(strcmp(argv[i],"-HashSearch")==0)
		{
			int rLen=8*1024*1024;
			int sLen=8*1024*1024;
			int numThread=4;
			if(argc==(i+4))
			{
				rLen=atoi(argv[i+1])*1024;
				sLen=atoi(argv[i+2])*1024;
				numThread=atoi(argv[i+3]);
			}
			testHashSearch(rLen,sLen,numThread);
		}
	}
}