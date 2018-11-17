#include "stdafx.h"
#include "StringLib.h"
#include "common.h"

int generateStringGPU(int numString, int minLen, int maxLen, char **data, int **len, int **offset)
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


void printString(void * rawData, cmp_type_t* d_R, int rLen)
{
//#ifdef DEBUG_SAVEN
	int i=0;
	char *data=(char*)rawData;
	char *str1=NULL;
	char *str2=NULL;
	int result=0;
	int errorTimes=0;
	for(i=1;i<rLen;i++)
	{
		str2=data+d_R[i].x;
		str1=data+d_R[i-1].x;
		result=compareStringCPULocal(str2,str1);
		if(result<0)
		{
			printf("error in sorting, %d, %s, %d, %s, %d, %s, %d, %s\n", i-2, data+d_R[i-2].x, i-1, str1, i, str2,i+1, data+d_R[i+1].x);
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

	/*char *data=(char*)rawData;
	char *str1=NULL;
	int i=0;
	for(i=0;i<rLen;i++)
	{
		
		str1=data+d_R[i].x;
		printf("%d,%s,", i,str1);
	}*/
}