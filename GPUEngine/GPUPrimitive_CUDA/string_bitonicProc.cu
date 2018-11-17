#ifndef STRING_BITONIC_PROC_H
#define STRING_BITONIC_PROC_H
#include "string_bitonicProc_kernel.cu"
//#define NUM_BLOCK_PER_CHUNK_BITONIC_SORT 512//b256

/*
@totalLenInBytes, is not used. 
*/
void string_bitonicSortMultipleBlocks(void* d_rawData, int totalLenInBytes, cmp_type_t * d_values, int* d_bound, int numBlock, cmp_type_t * d_output)
{
	int numThreadsPerBlock_x=SHARED_MEM_INT2;
	int numThreadsPerBlock_y=1;
	int numBlock_x=NUM_BLOCK_PER_CHUNK_BITONIC_SORT;
	int numBlock_y=1;
	int numChunk=numBlock/numBlock_x;
	if(numBlock%numBlock_x!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*numBlock_x;
		end=start+numBlock_x;
		if(end>numBlock)
			end=numBlock;
		//printf("bitonicSortMultipleBlocks_kernel: %d, range, %d, %d\n", i, start, end);
		string_bitonicSortMultipleBlocks_kernel<<<grid,thread>>>(d_rawData, totalLenInBytes, d_values, d_bound, start, end-start, d_output);
		cudaThreadSynchronize();
	}
//	cudaThreadSynchronize();
}

void initialize(cmp_type_t *d_data, int rLen, cmp_type_t value)
{
	int numThreadsPerBlock_x=512;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		initialize_kernel<<<grid,thread>>>(d_data, start, rLen, value);
	} 
	cudaThreadSynchronize();
}

void int4toint2(int4 *d_data, int rLen, Record* d_output)
{
	int numThreadsPerBlock_x=512;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		int4toint2_kernel<<<grid,thread>>>(d_data, start, rLen, d_output);
	} 
	cudaThreadSynchronize();
}


void getIntYArray(Record *d_data, int rLen, int* d_output)
{
	int numThreadsPerBlock_x=512;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		getIntYArray_kernel<<<grid,thread>>>(d_data, start, rLen, d_output);
	} 
	cudaThreadSynchronize();
}

void getXYArray(cmp_type_t *d_data, int rLen, Record* d_output)
{
	int numThreadsPerBlock_x=512;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		getXYArray_kernel<<<grid,thread>>>(d_data, start, rLen, d_output);
	} 
	cudaThreadSynchronize();
}

void getZWArray(cmp_type_t *d_data, int rLen, Record* d_output)
{
	int numThreadsPerBlock_x=512;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		getZWArray_kernel<<<grid,thread>>>(d_data, start, rLen, d_output);
	} 
	cudaThreadSynchronize();
}



void setXYArray(cmp_type_t *d_data, int rLen, Record* d_value)
{
	int numThreadsPerBlock_x=512;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		setXYArray_kernel<<<grid,thread>>>(d_data, start, rLen, d_value);
	} 
	cudaThreadSynchronize();
}

void setZWArray(cmp_type_t *d_data, int rLen, Record* d_value)
{
	int numThreadsPerBlock_x=512;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		setZWArray_kernel<<<grid,thread>>>(d_data, start, rLen, d_value);
	} 
	cudaThreadSynchronize();
}

#endif
