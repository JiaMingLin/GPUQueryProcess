#include "stdafx.h"
#include "common.h"
#include "assert.h"
#include "stdio.h"
#include "omp.h"
#include "CPU_Dll.h"


//the two procedures for the split
void computeHist(int *pidArray, int start, int end, int *localHist, int cpuid, int numThread)
{
	//set_thread_affinity(cpuid,numThread);
	int i=0;
	for(i=start;i<end;i++)
		localHist[pidArray[i]]++;
}

void histToOffset(int *startHist, int **hist, int numPart, int numThread)
{
	int pSum=0;
	int sum=0;
	int i=0,j=0;
	for(i=0;i<numPart;i++)
	{
		startHist[i]=0;
		for(j=0;j<numThread;j++)
			startHist[i]+=hist[j][i];
		sum+=startHist[i];
		startHist[i]=pSum;
		pSum=sum;		
	}
	//now compute the prefix for each partition.
	for(i=0;i<numPart;i++)
	{
		sum=0;pSum=0;
		for(j=0;j<numThread;j++)
		{
			sum+=hist[j][i];
			hist[j][i]=pSum+startHist[i];
			pSum=sum;		
		}
	}
}

void outputLocInSplit(int* pidArray, int start, int end, int *localHist, int *loc, int cpuid, int numThread)
{
	//set_thread_affinity(cpuid,numThread);
	int i=0;
	for(i=start;i<end;i++)
	{
		loc[i]=localHist[pidArray[i]];
		localHist[pidArray[i]]++;
	}
}


void getPivot(Record *Rin, int rLen, int sampleLevel, cmp_func fcn, Record** sampleRout)
{
	int numSample=(1<<sampleLevel);
	Record* sampleRin=(Record*)new Record[numSample];
	*sampleRout=(Record*)new Record[numSample];
	int segSize=rLen/numSample;
	int i=0;
	int tempIndex=0;
	for(i=0;i<rLen && tempIndex<numSample;i=i+segSize)
	{
		sampleRin[tempIndex]=Rin[i];
		tempIndex++;
	}
	assert(tempIndex==numSample);
	
	qsort(sampleRin, numSample,sizeof(Record),fcn);
	
	int j=0;
	int startIndex=0, delta=0;
	numSample=numSample-1;
	tempIndex=0;
	for(i=0;i<sampleLevel;i++)
	{
		delta=1<<(sampleLevel-i);
		startIndex=numSample>>(i+1);		
		for(j=startIndex;j<numSample;j=j+delta)
		{
			(*sampleRout)[tempIndex]=sampleRin[j];
			tempIndex++;
		}
	}
}


int log2(int value)
{
	int result=0;
	while(value>1)
	{
		value=value>>1;
		result++;
	}
	return result;
}

int log2Ceil(int value)
{
	int result=log2(value);
	if(value>(1<<result))
		result++;
	return result;
}

int compare (const void * a, const void * b)
{
  return ( ((Record*)a)->value - ((Record*)b)->value );
}


void validateSort(Record *R, int rLen)
{
	int i=0;
	bool passed=true;
	for(i=1;i<rLen;i++)
	{
		if(R[i].value<R[i-1].value)
		{
			printf("error in sorting: %d, %d, %d, %d\n", i-1, R[i-1].value, i,R[i].value);
			passed=false;
			break;
		}
	}
	if(passed)
		printf("\nsorting passed\n");
}
void validateSplit(Record *R, int rLen, int numPart)
{
	int i=0;
	bool passed=true;
	for(i=1;i<rLen;i++)
	{
		if((R[i].value%numPart)<(R[i-1].value%numPart))
		{
			printf("error in partition: %d, %d, %d, %d\n", i-1, R[i-1].value, i,R[i].value);
			passed=false;
			break;
		}
	}
	if(passed)
		printf("\npartition passed\n");
}


int dumpMLLtoArray(LinkedList **llList, int numLL, Record **Rout)
{
	int* numElementsEachLL=new int[numLL];
	int sum=0;
	int tempSum=0;
	int i=0;
	for(i=0;i<numLL;i++)
	{
		sum+=llList[i]->size();
		numElementsEachLL[i]=tempSum;
		tempSum=sum;
	}
	if(sum!=0)
	{
		*Rout=new Record[sum];
		#pragma omp parallel for
		for(i=0;i<numLL;i++)
		{
			llList[i]->copyToArray(*Rout+numElementsEachLL[i]);
		}
	}
	delete numElementsEachLL;
	return sum;
}


void initMLL(LinkedList **llList, int numLL)
{
	int i=0;
	for(i=0;i<numLL;i++)
	{
		llList[i]=(LinkedList *)malloc(sizeof(LinkedList));
		llList[i]->init();
	}
}

void freeMLL(LinkedList **llList, int numLL)
{
	if(llList!=NULL)
	{
		int i=0;
		for(i=0;i<numLL;i++)
		{
			llList[i]->destroy();
		}
		free(llList);
	}
}


int findLargestSmaller(Record *R, int start, int end, int searchValue)
{
	int mid=(start+end)>>1;
	int result=0;
	if(start==mid)
	{
		int midValue=R[mid].value;
		if(searchValue<=midValue)
			result=mid;
		else//searchValue>midValue
			result=mid+1;

	}
	else
	{
		int midValue=R[mid].value;
		if(searchValue<midValue)
			result=findLargestSmaller(R,start,mid-1,searchValue);
		else
			if(searchValue>midValue)
				result=findLargestSmaller(R,mid+1,end,searchValue);
			else//equal case
			{
				for(result=mid-1;result>=start;result--)
				{
					if(R[result].value!=searchValue)
						break;
				}
				result=result+1;//searchValue<R[result]
			}
	}

	return result;
}

void findQuantile(Record *R, int start, int end, int numQuantile, int* quanPos)
{
	int i=0;
	int j=0;
	//initialization
	//skip to the left if there is duplicate.
	int chunkSize=(end-start)/numQuantile;
	int tempValue=0;
	int tempPos=0;
	for(i=0;i<numQuantile;i++)
	{
		quanPos[i]=i*chunkSize;
		tempPos=quanPos[i];
		tempValue=R[tempPos].value;
		for(j=quanPos[i];j>=0;j--)
		{
			if(tempValue==R[j].value)
				tempPos=j;
			else
				break;
		}
		quanPos[i]=tempPos;
	}
}

void printString(char *data, Record* d_R, int rLen)
{
//#ifdef DEBUG_SAVEN
	int i=0;
	char *str1=NULL;
	char *str2=NULL;
	int result=0;
	int errorTimes=0;
	printf("\n");
	for(i=1;i<rLen;i++)
	{
		result=compareStringCPU(d_R+i,d_R+(i-1));
		if(result<0)
		{
			printf("error in sorting, %d, %s, %d, %s, %d, %s, %d, %s\n", i-2, data+d_R[i-2].value, i-1, str1, i, str2,i+1, data+d_R[i+1].value);
			errorTimes++;
			if(errorTimes>10)
			exit(0);
		}	
		//if(i<15)
		//	printf("%d,%s,", i,str1);
	}
	if(errorTimes==0)
		printf("pass the checking\n");
	else
		printf("fail the checking!!\n");
//#endif
}