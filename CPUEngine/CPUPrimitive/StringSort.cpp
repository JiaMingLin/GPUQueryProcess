#include "stdafx.h"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "math.h"
#include "StringLib.h"



void * s_qsRawData=NULL;



int getSTLQsortValue(const void * p1, const void* p2)
{
	cmp_type_t v1=*((cmp_type_t*)p1);
	cmp_type_t v2=*((cmp_type_t*)p2);
	return compareStringCPULocal((void*)(((char*)s_qsRawData)+v1.x),(void*)(((char*)s_qsRawData)+v2.x));
}

inline void swapCPU(cmp_type_t & a, cmp_type_t & b)
{
	// Alternative swap doesn't use a temporary register:
	// a ^= b;
	// b ^= a;
	// a ^= b;
	
    cmp_type_t tmp = a;
    a = b;
    b = tmp;
}

int compareStringCPULocal(const void *d_a, const void *d_b)
{
	char4 *str_a=(char4*)d_a;
	char4 *str_b=(char4*)d_b;
	int i=0;
	int result=0;
	char4 cura, curb;
	cura=str_a[i];
	curb=str_b[i];
	while(result==0)
	{	
		//loop 0
		if((cura.x=='\0') || (curb.x=='\0'))
		{
			//if(a[0]=='\0' && b[0]=='\0')
			if(cura.x==curb.x)
				result=0;
			else
				if(cura.x=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(cura.x>curb.x)
		{
			result=1;
			break;
		}
		else if(cura.x<curb.x)
		{
			result=-1;
			break;
		}
		//loop 1
		if((cura.y=='\0') || (curb.y=='\0'))
		{
			//if(a[0]=='\0' && b[0]=='\0')
			if(cura.y==curb.y)
				result=0;
			else
				if(cura.y=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(cura.y>curb.y)
		{
			result=1;
			break;
		}
		else if(cura.y<curb.y)
		{
			result=-1;
			break;
		}
		//loop 2
		if((cura.z=='\0') || (curb.z=='\0'))
		{
			//if(a[0]=='\0' && b[0]=='\0')
			if(cura.z==curb.z)
				result=0;
			else
				if(cura.z=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(cura.z>curb.z)
		{
			result=1;
			break;
		}
		else if(cura.z<curb.z)
		{
			result=-1;
			break;
		}
		//loop 3
		if((cura.w=='\0') || (curb.w=='\0'))
		{
			//if(a[0]=='\0' && b[0]=='\0')
			if(cura.w==curb.w)
				result=0;
			else
				if(cura.w=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(cura.w>curb.w)
		{
			result=1;
			break;
		}
		else if(cura.w<curb.w)
		{
			result=-1;
			break;
		}		
		i++;
		cura=str_a[i];
		curb=str_b[i];			
	}
	return result;
}


void stlQSCPU(void* rawData, int totalLenInBytes, cmp_type_t *Rin, int rLen, cmp_type_t** Rout)
{
	s_qsRawData=rawData;
	qsort(Rin, rLen, sizeof(cmp_type_t), getSTLQsortValue);
	*Rout=(cmp_type_t*)malloc(sizeof(cmp_type_t)*rLen);
	memcpy( *Rout, Rin, sizeof(cmp_type_t)*rLen );
	
}

void copyChunks_CPU(void *h_inputValArray, int2* h_PA, int rLen, void *h_outputValArray)
{
	//compute the prefix sum;
	int* pSum=(int*)malloc(sizeof(int)*rLen);
	int i=0;
	int sum=0;
	for(i=0;i<rLen;i++)
	{
		pSum[i]=sum;
		sum+=h_PA[i].y;		
	}
	//now copy the items.
	char * source=(char*)h_inputValArray;
	char * dest=(char*)h_outputValArray;
	int size;
	int start_source, start_dest;
	for(i=0;i<rLen;i++)
	{
		start_source=h_PA[i].x;
		size=h_PA[i].y;
		start_dest=pSum[i];
		memcpy(dest+start_dest, source+start_source, size*sizeof(char));
		h_PA[i].x=start_dest;
	}
	
}

int getChunkBoundary_CPU(void *h_source, cmp_type_t* h_Rin, int rLen, int2 ** h_outputKeyListRange)
{
	int i=0;
	int resultNumChunks=1;
	for(i=1;i<rLen;i++)
	{
		if(getSTLQsortValue(&(h_Rin[i]), &(h_Rin[i-1]))!=0)
			resultNumChunks++;
	}
	*h_outputKeyListRange=(int2*)malloc(sizeof(int2)*resultNumChunks);
	int curChunk=0;
	(*h_outputKeyListRange)[curChunk].x=0;
	curChunk++;
	for(i=1;i<rLen;i++)
	{
		if(getSTLQsortValue(&(h_Rin[i]), &(h_Rin[i-1]))!=0)
		{
			(*h_outputKeyListRange)[curChunk].x=i;
			(*h_outputKeyListRange)[curChunk-1].y=i;
			curChunk++;
		}
	}
	(*h_outputKeyListRange)[curChunk].y=rLen;
	return resultNumChunks;
}


int sort_CPU (void * h_inputKeyArray, int totalKeySize, void * h_inputValArray, int totalValueSize, 
		  cmp_type_t * h_inputPointerArray, int rLen, 
		  void ** h_outputKeyArray, void ** h_outputValArray, 
		  cmp_type_t ** h_outputPointerArray, int2 ** h_outputKeyListRange
		  )
{
	int numDistinctKey=0;
	int totalLenInBytes=-1;
	*h_outputPointerArray=(cmp_type_t*)malloc(sizeof(cmp_type_t)*rLen);	
	stlQSCPU(h_inputKeyArray, totalLenInBytes, h_inputPointerArray, rLen, h_outputPointerArray);

	//!we first scatter the values and then the keys. so that we can reuse d_PA. 
	int2 *h_PA=(int2*)malloc( sizeof(int2)*rLen);
	int i=0;
	for(i=0;i<rLen;i++)
	{
		h_PA[i].x=(*h_outputPointerArray)[i].z;
		h_PA[i].y=(*h_outputPointerArray)[i].w;
	}
	//scatter the values.
	if(h_inputValArray!=NULL)
	{
		*h_outputValArray=(char*)malloc(sizeof(char)*totalValueSize);
		copyChunks_CPU(h_inputValArray, h_PA, rLen, *h_outputValArray);
		//set the pointer array
		for(i=0;i<rLen;i++)
		{
			(*h_outputPointerArray)[i].z=h_PA[i].x;
			(*h_outputPointerArray)[i].w=h_PA[i].y;
		}
	}
	//scatter the keys.
	for(i=0;i<rLen;i++)
	{
		h_PA[i].x=(*h_outputPointerArray)[i].x;
		h_PA[i].y=(*h_outputPointerArray)[i].y;
	}
	if(h_inputKeyArray!=NULL)
	{
		*h_outputKeyArray=(char*)malloc(sizeof(char)*totalKeySize);
		copyChunks_CPU(h_inputKeyArray, h_PA, rLen, *h_outputKeyArray);
		//set the pointer array
		for(i=0;i<rLen;i++)
		{
			(*h_outputPointerArray)[i].x=h_PA[i].x;
			(*h_outputPointerArray)[i].y=h_PA[i].y;
		}
	}	
	//find the boudary for each key.

	numDistinctKey=getChunkBoundary_CPU(*h_outputKeyArray, *h_outputPointerArray, rLen, h_outputKeyListRange);

	return numDistinctKey;

}



