#ifndef STRING_GEN_H
#define STRING_GEN_H
#include "math.h"
#include "stdlib.h"
#include "stdio.h"


#define NUM_SYMBOL 52
/*
return: the total size of the data buffer
*/
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


/*
return: the total size of the data buffer
*/
int generateStringGPU(int numString, int minLen, int maxLen, char **data, int **len, int **offset)
{
	//*len=(int*)malloc(sizeof(int)*numString);
	CPUMALLOC((void**)&(*len), sizeof(int)*numString);
	//*offset=(int*)malloc(sizeof(int)*numString);
	CPUMALLOC((void**)&(*offset), sizeof(int)*numString);
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
	//*data=(char*)malloc(sizeof(char)*sum);
	CPUMALLOC((void**)&(*data), sizeof(char)*sum);
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


#endif

