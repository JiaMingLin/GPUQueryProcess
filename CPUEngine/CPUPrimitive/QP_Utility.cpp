#include "stdafx.h"
#include "QP_Utility.h"
#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "time.h"
#include "common.h"


/************************************************************************/
/* This function generates <rLen> random tuples; maybe duplicated. 
/************************************************************************/
void generateRand(Record *R, int maxmax, int rLen, int seed)
{
	int i=0;
	const int offset=(1<<15)-1;
	srand(seed);
	for(i=0;i<rLen;i++)
	{
		R[i].value=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%maxmax;
		R[i].rid=i;
	}
}

void generateJoinSelectivity(Record *R, int rLen, Record *S, int sLen, int max, float joinSel,int seed)
{
	int i=0;
	const int offset=(1<<15)-1;
	srand(seed);
	for(i=0;i<rLen;i++)
	{
		R[i].value=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%max;
		R[i].rid=i;
	}
	for(i=0;i<sLen;i++)
	{
		S[i].rid=-1;
		S[i].value=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%max;
	}
	int locR=0;
	int locS=0;
	int retry=0;
	const int MAX_RET=1024;
	double deltaSel=(double)(rLen)/(double)max/1.25;
	joinSel-=deltaSel;
	printf("%f,%f,",deltaSel,joinSel);
	if(joinSel<0)
	{
		joinSel=0-joinSel;
		int numMisses=(int)(joinSel*(float)sLen);
		for(i=0;i<numMisses;i++)
		{
			locR=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%rLen;
			locS=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%sLen;
			if(S[locS].rid==-1)
			{
				S[locS].value=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%max;
				S[locS].rid=1;
				retry=0;
			}
			else
			{
				retry++;
				i--;
				if(retry>MAX_RET)
					break;
			}
		}
	}
	else
	{
		int numHits=(int)(joinSel*(float)sLen);
		for(i=0;i<numHits;i++)
		{
			locR=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%rLen;
			locS=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%sLen;
			if(S[locS].rid==-1)
			{
				S[locS].value=R[locR].value;
				S[locS].rid=1;
				retry=0;
			}
			else
			{
				retry++;
				i--;
				if(retry>MAX_RET)
					break;
			}
		}
	}
	for(i=0;i<sLen;i++)
	{
		S[i].rid=i;
	}
	//for testing
#ifdef DEBUG_SAVEN
	printf("Be careful!!! DEBUGGING IS ENABLED\n");
	qsort(R,rLen,sizeof(Record),compare);
	qsort(S,sLen,sizeof(Record),compare);
#endif
}

void generateRandInt(int *R, int maxmax, int rLen, int seed)
{
	int i=0;
	const int offset=(1<<15)-1;
	srand(seed);
	for(i=0;i<rLen;i++)
	{
		R[i]=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%maxmax;
	}
}

void generateArray(int *R, int base, int step, int max, int rLen, int seed)
{
	int i=0;
	const int offset=(1<<15)-1;
	srand(seed);
	for(i=0;i<rLen;i++)
	{
		R[i*step+base]=i;//((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%max;
	}
}

/*
 *	generate <rLen> sorted Record, in ascending order.
 */

void generateSort(Record *R, int maxmax, int rLen, int seed)
{
	int i=0;
	const int offset=(1<<15)-1;
	srand(seed);
	for(i=0;i<rLen;i++)
	{
		R[i].value=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%maxmax;
		
	}
	qsort(R,rLen,sizeof(Record),compare);
	for(i=0;i<rLen;i++)
	R[i].rid=i;

}

/************************************************************************/
/* This function generates <rLen> distinct tuples; distinct.
/************************************************************************/
/* (1) generate N^0.5 16-bit distinct numbers  (stored in array a);
   (2) generate another N^0.5 16-bit distinct numbers  (stored in array b);
   (3) the result array, x: x[i*N^0.5+j] =(a[i]<<16)£«b[j]                 
/************************************************************************/
//step (1) and (2)
void generate16Bits(int *a, int max, int len, int seed)
{
	const int mask=(1<<16)-1;
	int i=0;
	int j=0;
	int temp=0;
	srand(seed);
	for(i=0;i<len;i++)
	{
		temp=(((rand()<<1)+(rand()&1))&mask)%max;
		for(j=0;j<i;j++)
			if(temp==a[j])
				break;
		if(j==i)
			a[i]=temp;
		else
			i--;	
	}
	//for(i=0;i<len;i++)
	//	printf("%d,",a[i]);
	//printf("\n");
	
}
void generateDistinct(Record *R, int max, int rLen, int seed)
{
	int i=0;
	int j=0;
	int curNum=0;
	int done=0;
	int nSquareRoot=(int)sqrt((double)rLen)+1;
	int *a=(int *)malloc(sizeof(int)*nSquareRoot);
	int *b=(int *)malloc(sizeof(int)*nSquareRoot);
	int maxSqrt=((int)sqrt((double)max)+1);
	generate16Bits(a,maxSqrt,nSquareRoot,seed);
	generate16Bits(b,maxSqrt,nSquareRoot,seed+1);
	for(i=0;i<nSquareRoot && !done;i++)
		for(j=0;j<nSquareRoot;j++)
		{
			R[curNum].value=(a[i]*maxSqrt)+b[j];
			R[curNum].rid=curNum;
			curNum++;
			if(curNum==rLen)
			{
				done=1;
				break;
			}		
		}
	free(a);
	free(b);
}


//generate the  each value for <dup> tuples.
//dup=1,2,4,8,16,32
void generateSkewDuplicates(Record *R,  int max, int rLen,int dup, int seed)
{
	int a=0;
	int i=0;
	int minmin=0;
	int maxmax=2;
	unsigned int mask=(2<<15)-1;
	int seg=rLen/dup;
	srand(seed);
	for(i=0;i<seg;i++)
	{
		R[i].value=((((rand()& mask)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%max;
		if(i==0)
		{
			minmin=maxmax=R[i].value;
		}
		else
		{
			if(minmin>R[i].value) minmin=R[i].value;
			if(maxmax<R[i].value) maxmax=R[i].value;
		}
		R[i].rid=i;
	}
	//copy the seg to all other segs.
	for(a=1;a<dup;a++)
	{
		for(i=0;i<seg;i++)
			R[a*seg+i].value=R[i].value;
	}
	//cout<<"min, "<<minmin<<", max, "<<maxmax<<", rand max, "<<max<<", dup, "<<dup<<endl;

}


void print(Record *R, int rLen)
{
	int i=0;
	printf("Random max=%d\n",RAND_MAX);
	for(i=0;i<rLen;i++)
	{
		printf("%d,%d\n",R[i].rid, R[i].value);
	}
}

void generateSkew(Record *R, int max, int rLen, double oneRatio, int seed)
{
	int numOnes=(int)((double)rLen*oneRatio);
	int i=0;
	for(i=0;i<numOnes;i++)
	{
		R[i].value=1;
		R[i].rid=i;
	}
	const int offset=(1<<15)-1;
	srand(seed);
	for(;i<rLen;i++)
	{
		R[i].value=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%max;
		R[i].rid=i;
	}
	//randomize the array
	randomize(R, rLen, numOnes);
}

void randomize(Record *R, int rLen, int times)
{
	int i=0;
	int temp=0;
	int from=0;
	int to=0;
	srand(times);
	const int offset=(1<<15)-1;
	for(i=0;i<times;i++)
	{
		from=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%rLen;
		to=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%rLen;
		temp=R[from].value;
		R[from].value=R[to].value;
		R[to].value=temp;		
	}
	
}

void randomInt(int *R, int rLen, int times)
{
	int i=0;
	int temp=0;
	int from=0;
	int to=0;
	srand(times);
	const int offset=(1<<15)-1;
	for(i=0;i<times;i++)
	{
		from=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%rLen;
		to=((((rand()& offset)<<15)+(rand()&1))+(rand()<<1)+(rand()&1))%rLen;
		temp=R[from];
		R[from]=R[to];
		R[to]=temp;		
	}
	
}

/************************************************************************/
/* Timing
/************************************************************************/
static clock_t g_startTime;

void startTime()
{
	 g_startTime= clock();
}
double endTime(char *info)
{
	double cpuTime;
	clock_t end = clock();
	cpuTime= (end-g_startTime)/ (double)CLOCKS_PER_SEC;
	printf("%s, time, %.3f sec\n", info, cpuTime);
	return cpuTime;
}


void startTimer(clock_t & l_startTime)
{
	 l_startTime= clock();
}
void endTimer(char *info, clock_t & l_startTime)
{
	double cpuTime;
	clock_t end = clock();
	cpuTime= (end-l_startTime)/ (double)CLOCKS_PER_SEC;
	printf("%s, time, %.3f\n", info, cpuTime);
}


//generate the string.
int generateString(int numString, int minLen, int maxLen, char **data, int **len, int **offset)
{
	*len=(int*)malloc(sizeof(int)*numString);
	*offset=(int*)malloc(sizeof(int)*numString);
	int i=0;
	int sum=0;
	srand(0);
	//randomly generated the length
	for(i=0;i<numString;i++)
	{
		(*offset)[i]=sum;
		if(maxLen!=minLen)
			(*len)[i]=rand()%(maxLen-minLen)+minLen;
		else
			(*len)[i]=minLen;
		sum+=(*len)[i];
	}
	*data=(char*)malloc(sizeof(char)*sum);
	int j=0, tempLen=0;
	for(i=0;i<numString;i++)
	{
		tempLen=(*len)[i];
		for(j=0;j<tempLen;j++)
		{
			(*data)[(*offset)[i]+j]=(char)(rand()%NUM_SYMBOL+'A');
		}
		(*data)[(*offset)[i]+j-1]='\0';
	}
	return sum;
}
