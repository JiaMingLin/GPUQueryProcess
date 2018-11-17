#include "stdafx.h"
#include "Primitive.h"
#include "mapImpl.h"
#include <iostream>
using namespace std;






int scan(Record *Rin, int rLen, int numThread, int* pS)
{
	int result=0;
#ifndef SAVEN_OPen_MP
	result=scan_thread(Rin,rLen,numThread, pS);
#else 
	result=scan_openmp(Rin,rLen,numThread,pS);
#endif
	return result;

}
void scatter(Record *Rin, int *loc, Record *S, int rLen, int numThread)
{
#ifndef SAVEN_OPen_MP
	scatter_thread(Rin,loc, S, rLen,numThread);
#else
	scatter_openmp(Rin,loc, S, rLen);
#endif
}
void gather(Record *Rin, int *loc, Record *S, int rLen, int numThread)
{
#ifndef SAVEN_OPen_MP
	gather_thread(Rin,loc, S, rLen,numThread);
#else
	gather_openmp(Rin,loc, S, rLen);
#endif
}

void split(Record *Rin, int rLen, int numPart, Record* Rout, int* startHist, mapper_t splitFunc, void *para, int numThread)
{
#ifndef SAVEN_OPen_MP
	split_thread(Rin,rLen,numPart,Rout,startHist, splitFunc,para, numThread);
#else 
	split_openmp(Rin,rLen,numPart,Rout,startHist, splitFunc,para, numThread);
#endif
	int i=0;
	//startHist[numPart-1]=rLen;
#ifdef DEBUG_SAVEN
	cout<<"split size, avg, "<<rLen/numPart<<", now: ";
	for(i=0;i<numPart-1;i++)
		cout<<startHist[i+1]-startHist[i]<<", ";
	cout<<endl;
#endif
}

void sort(Record* Rin, int rLen, cmp_func fcn, Record* Rout, int numThread)
{
#ifndef SAVEN_OPen_MP
	sort_thread(Rin,rLen,fcn,Rout, numThread);
#else 
	sort_openmp(Rin,rLen,fcn,qsort_pidfunc, Rout, numThread);
#endif

}