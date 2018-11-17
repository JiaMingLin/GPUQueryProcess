
#ifndef _HASH_TABLE_SEARCH_
#define _HASH_TABLE_SEARCH_

#include "stdio.h"
#include "stdlib.h"
#include "GPU_Dll.h"



////construct the hash table in the main memory
//inline int HASH(int value)
//{
//	//return ( ( value>>(30-intBits) )&( (1<<intBits)-1 ) );
//	//return ( (value>>1) &( (1<<intBits)-1 ) );
//	return ( (value>>1) &( (NUM_RECORDS_R>>1)-1));
//	
//}
#define HASH(VALUE, RLEN) ( ((VALUE)>>1) &( ((RLEN)>>1)-1))
void buildHashTable(Record* h_R, int rLen, int intBits, Bound *h_bound)
{
	//qsort(h_R, rLen, sizeof(Record), compareInt2);
	int curL=0;
	int start=0;
	int end=0;
	int curValue=0;
	//int index=0;
	for(curL=0;curL<rLen;curL++)
	{
		start=curL;
		curValue=HASH(h_R[curL].y,rLen);
		curL++;
		end=curL;
		for(;curL<rLen;curL++)
		if(HASH(h_R[curL].y,rLen)!=curValue)//one row.
		{
			end=curL;
			curL--;
			break;
		}
		h_bound[curValue].start=start;
		h_bound[curValue].end=end;
		/*printf("curValue, %d, [%d, %d], ",curValue, start, end);
		index++;
		if(index%10==9) printf("\n");*/
	}
		
}


__global__ void optProbe_kernel(Bound *d_bound, int rLen, int *d_keys, int* d_oSize, Bound* d_oBound, int blockSize, int pEnd, int from, int to)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bid=bx+by*gridDim.x;
	int tid=tx+ty*blockDim.x;
	int numThread=blockDim.x;
	unsigned int rStart=bid*blockSize;
	unsigned int rEnd=rStart+blockSize;
	if(rEnd>pEnd)
		rEnd=pEnd;
	int tmp;
	int key=0;
	int targetLoc=0;
	Bound tempInt2;
	for( int i = rStart; (i+tid) < rEnd; i=i+numThread)
	{
		tmp=i+tid;
		key=d_keys[tmp];
		targetLoc=HASH(key,rLen);
		if(targetLoc>=from && targetLoc<to)
		{
			tempInt2=d_bound[targetLoc];
			d_oSize[tmp]=tempInt2.end-tempInt2.start;
			d_oBound[tmp]=tempInt2;
		}
	}	
}

__global__ void location_kernel(int* d_loc, int* d_sum, Bound* d_oBound)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bid=bx+by*gridDim.x;
	int tid=tx+ty*blockDim.x;
	int resultID=bid*blockDim.x+tid;
	Bound tmpInt2=d_oBound[resultID];
	unsigned int rStart=tmpInt2.start;
	unsigned int localEnd=tmpInt2.end-tmpInt2.start;
	int gStart=d_sum[resultID];	
	for( int i = 0; i < localEnd; i++)
	{
		d_loc[i+gStart]=i+rStart;
	}	
}


__global__ void optFetch_kernel(Record *d_R, Record* d_result, int* d_loc, int blockSize, int pEnd, int from, int to)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bid=bx+by*gridDim.x;
	int tid=tx+ty*blockDim.x;
	int numThread=blockDim.x;
	unsigned int rStart=bid*blockSize;
	unsigned int rEnd=rStart+blockSize;
	if(rEnd>pEnd)
		rEnd=pEnd;
	int tmp;
	//int key=0;
	int targetLoc=0;
	//Record tempInt2;
	for( int i = rStart; (i+tid) < rEnd; i=i+numThread)
	{
		tmp=i+tid;
		targetLoc=d_loc[tmp];
		if(targetLoc>=from && targetLoc<to)
		{
			d_result[tmp]=d_R[targetLoc];
		}
	}	
}


extern "C"
void GPUCopy_BuildHashTable( Record* h_R, int rLen, int intBits, Bound* h_bound )
{
	buildHashTable(h_R, rLen, intBits, h_bound);
}


#endif
