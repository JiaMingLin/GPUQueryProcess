#ifndef STRING_QUICKSORT_KERNEL_CU
#define STRING_QUICKSORT_KERNEL_CU



#include "stdio.h"
#include "StringCmp.cu"
#define MIN(a,b)  (((a) < (b)) ? (a) : (b))
#define MAX(a,b)  (((a) > (b)) ? (a) : (b))
/*
level by level
*/
//deltaCount means the number of threads in one partitions.
//pivot is the index in d_R;
__global__ void
binaryPart_level(void *d_rawData, int totalLenInBytes, int numPart, cmp_type_t* d_R, int *d_iBoud,  int blockSize, int *d_Count, int deltaCount, cmp_type_t *d_pivot, int pivotStart)
{
	//__shared__ Record tempPivot[3];
	__shared__ cmp_type_t shared_pivot;
	__shared__ int pStart;
	__shared__ int pEnd;
	int by = blockIdx.y;
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int ty = threadIdx.y;	
	int tid=tx+ty*blockDim.x;
	//int bid=bx+by*gridDim.x;
	int numThread=blockDim.x;
	int resultID=(bx)*numThread+tid;
	int pivot;	
	if(tid==0)
	{
		pStart=d_iBoud[(by<<1)];//we use by to locate the inputBoundary
		pEnd=d_iBoud[(by<<1)+1];//we use by to locate the inputBoundary
		shared_pivot=d_pivot[by+pivotStart];
		//printf("%d, %d, %d, %d\n",bid, pStart,pEnd,shared_pivot);
	}
	/*if(pStart==pEnd)
		return;*/
	__syncthreads();
	
	unsigned int rStart=pStart+bx*blockSize;
	unsigned int rEnd=rStart+blockSize;
	
	int tmp=0;
	if(rEnd>pEnd || (bx+1)==gridDim.x)//the last thread block:)
		rEnd=pEnd;
	int zeros=0;
	int ones=0;
	int compareValue;
	for( int i = rStart; (i+tid) < rEnd; i=i+numThread)
	{
		tmp=i+tid;
		//if(d_R[tmp]<pivot)
		//if(compareString((void*)(((char*)d_rawData)+d_R[tmp].x),(void*)(((char*)d_rawData)+shared_pivot.x))<0)
		//if(<0)
		//	zeros++;
		//else
		//	ones++;
		compareValue=getCompareValue(d_rawData, d_R[tmp].x, shared_pivot.x);
		if(compareValue<0)
			zeros++;
		else
			if(compareValue>0)
			ones++;
			else //compareValue==0
				if((resultID&1)==0)
					zeros++;
				else
					ones++;
		
	}
	pivot=deltaCount*(by<<1);
	pivot+=resultID;
	d_Count[pivot]=zeros;
	d_Count[deltaCount+pivot]=ones;
	//zeroCount[resultID+numThread*gridDim.x]=ones;
	//printf("\n");
}


__global__ void
write_level(void *d_rawData, int totalLenInBytes, int numPart, cmp_type_t* d_R, int *d_iBoud,  int blockSize, int *d_Sum, int deltaCount, int numChunk, cmp_type_t *d_output, int *d_oBound, cmp_type_t *d_pivot,int pivotStart)
{
	__shared__ cmp_type_t shared_pivot;
	__shared__ int pStart;
	__shared__ int pEnd;
	int by = blockIdx.y;
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tid=tx+ty*blockDim.x;
	//int bid=bx+by*gridDim.x;
	int numThread=blockDim.x;
	int resultID=(bx)*numThread+tid;
	int pivot=0;	
	if(tid==0)
	{
		pStart=d_iBoud[(by<<1)];//we use by to locate the inputBoundary
		pEnd=d_iBoud[(by<<1)+1];//we use by to locate the inputBoundary
		//int a=(pStart+pEnd)>>1;
		//int a=pStart+bx%(pEnd-pStart);
		shared_pivot=d_pivot[by+pivotStart];
		//printf("%s, ", ((char*)d_rawData)+shared_pivot.x);
	}
	/*if(pStart==pEnd)
	{
		return;
	}*/
	__syncthreads();
	pivot=shared_pivot.x;
	unsigned int rStart=pStart+bx*blockSize;
	unsigned int rEnd=rStart+blockSize;
	
	int tmp=0;
	if(rEnd>pEnd || (bx+1)==gridDim.x)//the last thread block:)
		rEnd=pEnd;
	tmp=deltaCount*(by<<1);
	int zeros=d_Sum[resultID+tmp]+pStart;
	int ones=d_Sum[resultID+tmp+deltaCount]+pStart;
	//insert the boundary
	if(resultID==0)
	{
		tmp=(by<<2);
		d_oBound[tmp]=pStart;
		d_oBound[1+tmp]=ones;
		d_oBound[2+tmp]=ones;
		d_oBound[3+tmp]=pEnd;
	}
	cmp_type_t tmpValue;
	int compareValue;
	for( int i = rStart; (i+tid) < rEnd; i=i+numThread)
	{
			tmp=i+tid;
			tmpValue=d_R[tmp];
			//from=tmp;
			//if(tmpValue<pivot)
			//if(compareString((void*)(((char*)d_rawData)+tmpValue.x),(void*)(((char*)d_rawData)+shared_pivot.x))<0)
			//if(getCompareValue(d_rawData, tmpValue.x, shared_pivot.x)<0)
			//{
			//	//d_output[zeros]=tmpValue;
			//	tmp=zeros;
			//	zeros++;
			//}
			//else
			//{
			//	//d_output[ones]=tmpValue;
			//	tmp=ones;
			//	ones++;
			//}
			compareValue=getCompareValue(d_rawData, tmpValue.x, shared_pivot.x);
			if(compareValue<0)
			{
				tmp=zeros;
				zeros++;
			}
			else if(compareValue>0)
			{
				tmp=ones;
				ones++;
			}
			else 
				if((resultID&1)==0)					
				{
					tmp=zeros;
					zeros++;
				}
				else
				{
					tmp=ones;
					ones++;
				}
			d_output[tmp]=tmpValue;
			//if(tmpValue.x==148)
			//	printf("from:%d, to: %d, %d, %d; range, %d, %d\n",from, tmp, shared_pivot.x,tmpValue.x, rStart, rEnd);
	}
}

///////////////////////////////////////////////////////////////
/*
no preset pivot.
*/

__global__ void
binaryPart_level_selfPivot(void *d_rawData, int totalLenInBytes, int numPart, cmp_type_t* d_R, int *d_iBoud,  int blockSize, int *d_Count, int deltaCount, cmp_type_t *d_pivot)
{
	//__shared__ Record tempPivot[3];
	__shared__ cmp_type_t shared_pivot;
	__shared__ int pStart;
	__shared__ int pEnd;
	int by = blockIdx.y;
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int ty = threadIdx.y;	
	int tid=tx+ty*blockDim.x;
	int bid=bx+by*gridDim.x;
	int numThread=blockDim.x;
	int resultID=(bx)*numThread+tid;
	int pivot;	
	if(tid==0)
	{
		pStart=d_iBoud[(by<<1)];//we use by to locate the inputBoundary
		pEnd=d_iBoud[(by<<1)+1];
		int localEnd=pEnd-1;
		int a=(pStart+localEnd)>>1;
		int localMax=0;
		int localMin=0;		
		int l=d_R[pStart].x;
		int r=d_R[localEnd].x;
		int m=d_R[a].x;
		int localMaxIndex=0;
		int localMinIndex=0;
		//R[pStart]>R[pEnd]
		//if(compareString((void*)(((char*)d_rawData)+l),(void*)(((char*)d_rawData)+r))>0)
		if(getCompareValue(d_rawData, l, r)>0)
		{
			localMax=l;
			localMin=r;
			localMaxIndex=pStart;
			localMinIndex=localEnd;
		}
		else
		{
			localMax=r;
			localMin=l;
			localMaxIndex=localEnd;
			localMinIndex=pStart;
		}
		//R[localMax]<R[a]
		//if(compareString((void*)(((char*)d_rawData)+localMax),(void*)(((char*)d_rawData)+m))<0)
		if(getCompareValue(d_rawData, localMax, m)<0)
		{
			pivot=localMaxIndex;
		}
		else
		{
			//if(compareString((void*)(((char*)d_rawData)+localMin),(void*)(((char*)d_rawData)+m))<0)
			if(getCompareValue(d_rawData, localMin, m)<0)
				pivot=a;
			else
				pivot=localMinIndex;
		}
		shared_pivot=d_R[pivot];
		d_pivot[bid]=shared_pivot;
		//if(bid<=1)
		//	printf("bid, %d, pivot, %d, \n", bid, shared_pivot.x);
		
	}
	/*if(pStart==pEnd)
		return;*/
	__syncthreads();
	
	unsigned int rStart=pStart+bx*blockSize;
	unsigned int rEnd=rStart+blockSize;
	
	int tmp=0;
	if(rEnd>pEnd || (bx+1)==gridDim.x)//the last thread block:)
		rEnd=pEnd;
	int zeros=0;
	int ones=0;
	int compareValue=0;
	for( int i = rStart; (i+tid) < rEnd; i=i+numThread)
	{
		tmp=i+tid;
		//if(d_R[tmp]<pivot)
		//if(compareString((void*)(((char*)d_rawData)+d_R[tmp].x),(void*)(((char*)d_rawData)+shared_pivot.x))<0)
		//if(getCompareValue(d_rawData, d_R[tmp].x, shared_pivot.x)<0)
		//	zeros++;
		//else
		//	ones++;
		compareValue=getCompareValue(d_rawData, d_R[tmp].x, shared_pivot.x);
		if(compareValue<0)
			zeros++;
		else
			if(compareValue>0)
			ones++;
			else //compareValue==0
				if((resultID&1)==0)
					zeros++;
				else
					ones++;
		
	}
	pivot=deltaCount*(by<<1);
	pivot+=resultID;
	d_Count[pivot]=zeros;
	d_Count[deltaCount+pivot]=ones;
	//zeroCount[resultID+numThread*gridDim.x]=ones;
	//printf("\n");
}


__global__ void
write_level_selfPivot(void *d_rawData, int totalLenInBytes, int numPart, cmp_type_t* d_R, int *d_iBoud,  int blockSize, int *d_Sum, int deltaCount, int numChunk, cmp_type_t *d_output, int *d_oBound, cmp_type_t *d_pivot)
{
	__shared__ cmp_type_t shared_pivot;
	__shared__ int pStart;
	__shared__ int pEnd;
	int by = blockIdx.y;
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tid=tx+ty*blockDim.x;
	int bid=bx+by*gridDim.x;
	int numThread=blockDim.x;
	int resultID=(bx)*numThread+tid;
	int pivot=0;	
	if(tid==0)
	{
		pStart=d_iBoud[(by<<1)];//we use by to locate the inputBoundary
		pEnd=d_iBoud[(by<<1)+1];//we use by to locate the inputBoundary
		shared_pivot=d_pivot[bid];
		//if(bid<=1)
		//	printf("bid, %d, pivot, %d, \n", bid, shared_pivot.x);
	}
	/*if(pStart==pEnd)
	{
		return;
	}*/
	__syncthreads();
	pivot=shared_pivot.x;
	unsigned int rStart=pStart+bx*blockSize;
	unsigned int rEnd=rStart+blockSize;
	
	int tmp=0;
	if(rEnd>pEnd || (bx+1)==gridDim.x)//the last thread block:)
		rEnd=pEnd;
	tmp=deltaCount*(by<<1);
	int zeros=d_Sum[resultID+tmp]+pStart;
	int ones=d_Sum[resultID+tmp+deltaCount]+pStart;
	//insert the boundary
	if(resultID==0)
	{
		tmp=(by<<2);
		d_oBound[tmp]=pStart;
		d_oBound[1+tmp]=ones;
		d_oBound[2+tmp]=ones;
		d_oBound[3+tmp]=pEnd;
	}
	cmp_type_t tmpValue;
	int compareValue=0;
	for( int i = rStart; (i+tid) < rEnd; i=i+numThread)
	{
			tmp=i+tid;
			tmpValue=d_R[tmp];
			//from=tmp;
			//if(tmpValue<pivot)
			//if(compareString((void*)(((char*)d_rawData)+tmpValue.x),(void*)(((char*)d_rawData)+shared_pivot.x))<0)
			//if(getCompareValue(d_rawData,tmpValue.x, shared_pivot.x)<0)
			//{
			//	tmp=zeros;
			//	zeros++;
			//}
			//else
			//{
			//	tmp=ones;
			//	ones++;
			//}
			compareValue=getCompareValue(d_rawData, tmpValue.x, shared_pivot.x);
			if(compareValue<0)
			{
				tmp=zeros;
				zeros++;
			}
			else if(compareValue>0)
			{
				tmp=ones;
				ones++;
			}
			else 
				if((resultID&1)==0)					
				{
					tmp=zeros;
					zeros++;
				}
				else
				{
					tmp=ones;
					ones++;
				}
			d_output[tmp]=tmpValue;
			//if(tmpValue.x==22)
			//{
			//	printf("range, %d, %d, pivot, %d, from, %d, to %d\n", rStart, rEnd,shared_pivot.x, from, tmp); 
			//}
	}
}



#endif

