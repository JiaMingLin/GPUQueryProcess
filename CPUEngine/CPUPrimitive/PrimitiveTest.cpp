#include "stdafx.h"
#include "MyThreadPool.h"
#include "Primitive.h"
#include "mapImpl.h"
#include "common.h"
#include "LinkedList.h"
#include "Test.h"
#include "StringLib.h"
#include <iostream>
using namespace std;


void extractInt(void *Rin, void* para, void *Rout)
{
	Record *r=(Record*)Rin;
	int *o=(int*)Rout;
	*o=r->value;
}

void getPartID(void *Rin, void* para, void *Rout)
{
	Record *r=(Record*)Rin;
	int *o=(int*)Rout;
	int numPart=*((int*)para);
	*o=(r->value)%numPart;
}

void testFilter( int rLen, int numThread )
{
	Record* Rin = new Record[rLen];
	generateRand( Rin, TEST_MAX, rLen, 0 );
	Record* Rout;

	startTime();
	filter_openmp( Rin, rLen,  1, 1,&Rout, numThread);
	double sec = endTime( "filter" );

	cout<<"rLen, "<<rLen<<", numThreads, "<<numThread<<endl;
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	cout<<"\n bandwidth, "<< bandwidth<<" MB/sec"<<endl;

	delete Rin;
	delete Rout;
}

void testReduce( int rLen, int numThread, int OPERATOR )
{
	Record* Rin = new Record[rLen];
	generateRand( Rin, TEST_MAX, rLen, 0 );

	long int result = 0;

	startTime();
	result = reduce_openmp( Rin, rLen, numThread, OPERATOR);
	double sec = endTime( "reduce" );

	cout << "result: " << result << endl;
	cout<<"rLen, "<<rLen<<", numThreads, "<<numThread<<endl;
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	cout<<"\n bandwidth, "<< bandwidth<<" MB/sec"<<endl;
	delete Rin;
}

void testMapImpl(int rLen, int numThread)
{
	int result=0;
	Record *Rin=new Record[rLen];
	generateRand(Rin, TEST_MAX,rLen,0);
	int *Rout=new int[rLen];
	startTime();
	mapImpl<int>(Rin,rLen,extractInt, NULL, Rout,numThread);
	double sec=endTime("map");
	cout<<", rLen, "<<rLen<<", numThreads, "<<numThread<<", sum, "<<result<<", "<<endl;
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	cout<<"\n bandwidth, "<< bandwidth<<" MB/sec"<<endl;
	delete Rin;
}

void testSort(int rLen, int numThread)
{
	int result=0;
	Record *Rin=new Record[rLen];
	generateRand(Rin, TEST_MAX,rLen,0);
	Record *Rout=new Record[rLen];
	startTime();
	sort(Rin,rLen,compare,Rout,numThread);
	double sec=endTime("sort");
	validateSort(Rout,rLen);
	cout<<", rLen, "<<rLen<<", numThreads, "<<numThread<<", sum, "<<result<<", "<<endl;
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	cout<<"\n bandwidth, "<< bandwidth<<" MB/sec"<<endl;
	delete Rin;
	delete Rout;
}

void testSplit(int rLen, int numThread, int numPart)
{
	int result=0;
	Record *Rin=new Record[rLen];
	generateRand(Rin, TEST_MAX,rLen,0);
	Record *Rout=new Record[rLen];
	int* startHist=new int[numPart];
	startTime();
	split(Rin,rLen,numPart,Rout,startHist, getPartID,(void*)(&numPart), numThread);
	double sec=endTime("split");
	int i=0;
	for(i=0;i<rLen;i++)
		result+=Rout[i].value%numPart;
	//	cout<<Rout[i].value%numPart<<",";
	cout<<", rLen, "<<rLen<<", numThreads, "<<numThread<<", sum, "<<result<<", numPart, "<<numPart<<endl;
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	cout<<"\n bandwidth, "<< bandwidth<<" MB/sec"<<endl;	
	delete Rin;
	delete Rout;
	delete startHist;
}

void testScan(int rLen, int numThread)
{
	int result=0;
	Record *Rin=new Record[rLen];
	generateRand(Rin, TEST_MAX,rLen,0);
	int* pS=new int[rLen];
	startTime();
	result=scan(Rin,rLen,numThread,pS);
	double sec=endTime("scan");
	cout<<", rLen, "<<rLen<<", numThreads, "<<numThread<<", sum, "<<result<<", "<<endl;
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	cout<<"\n bandwidth, "<< bandwidth<<" MB/sec"<<endl;
	delete Rin;
	delete pS;
}

void testScatter(int rLen, int numThread)
{
	int result=0;
	Record *Rin=new Record[rLen];
	generateRand(Rin, TEST_MAX,rLen,0);
	Record *S=new Record[rLen];
	int *loc=new int[rLen];
	int i=0;
	//for(i=0;i<rLen;i++)
	//	loc[i]=i;
	//randomInt(loc, rLen, rLen);
	generateRandInt(loc,rLen,rLen,0);
	startTime();
	scatter(Rin,loc, S, rLen,numThread);
	double sec=endTime("scatter");
	//result=scan(S, rLen, numThread);
	cout<<", rLen, "<<rLen<<", numThreads, "<<numThread<<", sum, "<<result<<", "<<endl;
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	cout<<"\nscatter bandwidth, "<< bandwidth<<" MB/sec"<<endl;
	delete Rin;
	delete S;
	delete loc;
}

void testGather(int rLen, int numThread)
{
	int result=0;
	Record *S=new Record[rLen];
	generateRand(S, rLen,rLen,0);
	Record *Rin=new Record[rLen];
	int *loc=new int[rLen];
	int i=0;
	//for(i=0;i<rLen;i++)
	//	loc[i]=i;
	//randomInt(loc, rLen, rLen);
	generateRandInt(loc,rLen,rLen,0);
	startTime();
	gather(Rin,loc, S, rLen,numThread);
	double sec=endTime("gather");
//	result=scan(Rin, rLen, numThread);
	cout<<", rLen, "<<rLen<<", numThreads, "<<numThread<<", sum, "<<result<<", "<<endl;
	double dataSize=(double)(sizeof(Record)*rLen)/1024.0/1024.0;//in MB.
	double bandwidth=dataSize/sec;
	cout<<"\ngather bandwidth, "<< bandwidth<<" MB/sec"<<endl;
	delete Rin;
	delete S;
	delete loc;
}

void testLL(int numLL)
{
	LinkedList **llList=(LinkedList **)malloc(sizeof(LinkedList*)*numLL);
	initMLL(llList, numLL);
	int i=0;
	Record r;
	for(i=0;i<numLL;i++)
	{
		r.value=r.rid=i;
		llList[i]->fill(&r);
	}
	Record** Rout=(Record **)malloc(sizeof(Record *));
	int numElement=dumpMLLtoArray(llList,numLL,Rout);
	freeMLL(llList,numLL);
}


void testSortString(int numString, int minLen, int maxLen, int numThread)
{
	char **data=(char**)malloc(sizeof(char*));
	int **len=(int**)malloc(sizeof(int*));
	int **offset=(int**)malloc(sizeof(int*));
	int totalLenInBytes=generateString(numString,minLen,maxLen,data,len,offset);
	cmp_type_t *Rin=(cmp_type_t*)malloc(sizeof(cmp_type_t)*numString);
	cmp_type_t **Rout=(cmp_type_t**)malloc(sizeof(cmp_type_t*));
	int i=0;
	for(i=0;i<numString;i++)
	{
		Rin[i].x=(*offset)[i];
		Rin[i].y=(*len)[i];
	}
	startTime();
	//stlQSCPU( *data, totalLenInBytes, Rin, numString, Rout);
	char ** h_outputKeyArray= (char**)malloc(sizeof(char*));
	int2 ** h_outputKeyListRange=(int2 **)malloc(sizeof(int2 *));
	int numKey=sort_CPU(*data, totalLenInBytes, NULL, -1, Rin, numString, (void**)h_outputKeyArray, NULL, Rout, h_outputKeyListRange);
	printf("numKey, %d\n",numKey);
	endTime("STL sorting");
	printString(*h_outputKeyArray, (cmp_type_t*)*Rout, numString);
}

void testPrimitive(int argc, char **argv)
{
	int i=0;
//	testSplit(16, 2, 4);
//	testSort(32,4);
//	testScan(128,4);
//	testLL(10);
	
	for(i=0;i<argc;i++)
	{
		if(strcmp(argv[i], "-map")==0)
		{
			int rLen=8*1024*1024;
			int numThread=8;
			if(argc==(i+3))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				numThread=atoi(argv[i+2]);

			}
			testMapImpl(rLen, numThread);
		}

		if(strcmp(argv[i], "-reduce")==0)
		{
			int rLen=8*1024*1024;
			int numThread=8;
			if(argc==(i+3))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				numThread=atoi(argv[i+2]);
			}
			testReduce(rLen, numThread,REDUCE_MAX);
		}

		if(strcmp(argv[i], "-filter")==0)
		{
			int rLen=8*1024*1024;
			int numThread=8;
			if(argc==(i+3))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				numThread=atoi(argv[i+2]);
			}
			testFilter(rLen, numThread);
		}

		if(strcmp(argv[i], "-split")==0)
		{
			int rLen=8*1024*1024;
			int numThread=8;
			int numPart=64;
			if(argc==(i+4))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				numThread=atoi(argv[i+2]);
				numPart=atoi(argv[i+3]);

			}
			testSplit(rLen, numThread, numPart);
		}
		
		if(strcmp(argv[i], "-scan")==0)
		{
			int rLen=8*1024*1024;
			int numThread=8;
			if(argc==(i+3))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				numThread=atoi(argv[i+2]);

			}
			testScan(rLen, numThread);
		}

		if(strcmp(argv[i], "-scatter")==0)
		{
			int rLen=8*1024*1024;
			int numThread=8;
			if(argc==(i+3))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				numThread=atoi(argv[i+2]);
			}
			testScatter(rLen, numThread);
		}

		if(strcmp(argv[i], "-sort")==0)
		{
			int rLen=8*1024*1024;
			int numThread=8;
			if(argc==(i+3))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				numThread=atoi(argv[i+2]);
			}
			testSort(rLen, numThread);
		}

		if(strcmp(argv[i], "-string")==0)
		{
			int rLen=4*1024*1024;
			int minLen=4;
			int maxLen=4;
			int numThread=8;
			if(argc==(i+5))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				minLen=atoi(argv[i+2]);
				maxLen=atoi(argv[i+3]);
				numThread=atoi(argv[i+4]);
			}
			testSortString(rLen, minLen,maxLen, numThread);
		}

		if(strcmp(argv[i], "-gather")==0)
		{
			int rLen=8*1024*1024;
			int numThread=8;
			if(argc==(i+3))
			{
				rLen=atoi(argv[i+1])*1024*1024;
				numThread=atoi(argv[i+2]);
			}
			testGather(rLen, numThread);
		}
	}
}