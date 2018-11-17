#ifndef _GPU_DLL_H_
#define _GPU_DLL_H_

#include "cuCSSTree.h"

typedef struct
{
	IDataNode* data;
	unsigned int nDataNodes;

	IDirectoryNode* dir;
	unsigned int nDirNodes;

	int search(int key, Record** Rout)
	{
		return 0;
	}
	void print()
	{
		
	}
} CUDA_CSSTree;

struct Bound{
	int start;//x
	int end;//y
};

//memory opeations
extern "C" __declspec( dllexport ) void CopyCPUToGPU( void* d_destData, void* h_srcData, int sizeInBytes );

extern "C" __declspec( dllexport ) void CopyGPUToCPU( void * h_destData, void* d_srcData, int sizeInBytes);

extern "C" __declspec( dllexport ) void GPUAllocate( void** d_data, int sizeInBytes );

extern "C" __declspec( dllexport ) void CPUAllocateByCUDA( void** h_data, int sizeInBytes );

extern "C" __declspec( dllexport ) void GPUFree( void* d_data);
extern "C" __declspec( dllexport ) void CPUFreeByCUDA( void* h_data);

////////////////////////////////////////////////////////////////////////////////
//GPUOnly_
////////////////////////////////////////////////////////////////////////////////

//selection
extern "C" __declspec( dllexport ) int GPUOnly_PointSelection( Record* d_Rin, int rLen, int matchingKeyValue, Record** d_Rout, 
															  int numThreadPB = 32, int numBlock = 256);

extern "C" __declspec( dllexport ) int GPUOnly_RangeSelection(Record* d_Rin, int rLen, int rangeSmallKey, int rangeLargeKey, Record **d_Rout, 
															  int numThreadPB = 64, int numBlock = 512 );

//projection
extern "C" __declspec( dllexport ) void GPUOnly_Projection( Record* d_Rin, int rLen, Record* d_projTable, int pLen, 
														   int numThread = 256, int numBlock = 256 );

//aggregation
extern "C" __declspec( dllexport ) int GPUOnly_AggMax( Record* d_Rin, int rLen, Record** d_Rout, 
													  int numThread = 256, int numBlock = 1024 );

extern "C" __declspec( dllexport ) int GPUOnly_AggMin( Record* d_Rin, int rLen, Record** d_Rout, 
													  int numThread = 256, int numBlock = 1024 );

extern "C" __declspec( dllexport ) int GPUOnly_AggSum( Record* d_Rin, int rLen, Record** d_Rout, 
													  int numThread = 256, int numBlock = 1024 );

extern "C" __declspec( dllexport ) int GPUOnly_AggAvg( Record* d_Rin, int rLen, Record** d_Rout, 
													  int numThread = 256, int numBlock = 1024 );

//group by
extern "C" __declspec( dllexport ) int GPUOnly_GroupBy( Record* d_Rin, int rLen, Record* d_Rout, int** d_startPos, 
													   int numThread = 64, int numBlock = 1024 );

//agg after group by
extern "C" __declspec( dllexport ) void GPUOnly_agg_max_afterGroupBy( Record* d_Rin, int rLen, int* d_startPos, int numGroups, Record* d_Ragg, int* d_aggResults,
																	 int numThread = 256 );


extern "C" __declspec( dllexport ) void GPUOnly_agg_min_afterGroupBy( Record* d_Rin, int rLen, int* d_startPos, int numGroups, Record* d_Ragg, int* d_aggResults,
																	 int numThread = 256 );


extern "C" __declspec( dllexport ) void GPUOnly_agg_sum_afterGroupBy( Record* d_Rin, int rLen, int* d_startPos, int numGroups, Record* d_Ragg, int* d_aggResults,
																	 int numThread = 256 );


extern "C" __declspec( dllexport ) void GPUOnly_agg_avg_afterGroupBy( Record* d_Rin, int rLen, int* d_startPos, int numGroups, Record* d_Ragg, int* d_aggResults,
																	 int numThread = 256 );

//data structure
extern "C" __declspec( dllexport ) int GPUOnly_BuildTreeIndex( Record* d_Rin, int rLen, CUDA_CSSTree** d_tree );  

extern "C" __declspec( dllexport ) int GPUOnly_HashSearch( Record* d_R, int rLen, Bound* d_bound, int* d_keys, int sLen, Record** d_result, 
														  int numThread = 512 );

extern "C" __declspec( dllexport ) int GPUOnly_TreeSearch( Record* d_Rin, int rLen, CUDA_CSSTree* tree, Record* d_Sin, int sLen, Record** d_Rout );

//sort
extern "C" __declspec( dllexport ) void GPUOnly_bitonicSort( Record* d_Rin, int rLen, Record* d_Rout, 
						 int numThreadPB = 128, int numBlock = 1024);

extern "C" __declspec( dllexport ) void GPUOnly_RadixSort( Record* d_Rin, int rLen, Record* d_Rout );

extern "C" __declspec( dllexport ) void GPUOnly_QuickSort( Record* d_Rin, int rLen, Record* d_Rout );


//join

extern "C" __declspec( dllexport ) int GPUOnly_ninlj( Record *d_R, int rLen, Record *d_S, int sLen, Record** d_Rout );

extern "C" __declspec( dllexport ) int GPUOnly_smj(Record *d_R, int rLen, Record *d_S, int sLen, Record** d_Joinout);

extern "C" __declspec( dllexport ) int GPUOnly_hj( Record* d_Rin, int rLen, Record* d_Sin, int sLen, Record** d_Rout );

extern "C" __declspec( dllexport ) int GPUOnly_inlj(Record* d_Rin, int rLen, CUDA_CSSTree* d_tree, Record* d_Sin, int sLen, Record** d_Rout );

extern "C" __declspec( dllexport ) int GPUOnly_mj( Record* d_Rin, int rLen, Record* d_Sin, int sLen, Record** d_Joinout );  


//partition
extern "C" __declspec( dllexport ) void GPUOnly_Partition( Record* d_Rin, int rLen, int numPart, Record* d_Rout, int* d_startHist,
														  int numThreadPB = -1, int numBlock = -1);

///////////////////////////////////////////////////////////////////////////////////
//GPUCopy_
////////////////////////////////////////////////////////////////////////////////////

//selection
extern "C" __declspec( dllexport ) int GPUCopy_PointSelection( Record* h_Rin, int rLen, int matchingKeyValue, Record** h_Rout, 
															  int numThreadPB = 32, int numBlock = 256);

extern "C" __declspec( dllexport ) int GPUCopy_RangeSelection( Record* h_Rin, int rLen, int rangeSmallKey, int rangeLargeKey, Record** h_Rout, 
															  int numThreadPB = 64, int numBlock = 512);

//projection
extern "C" __declspec( dllexport ) void GPUCopy_Projection( Record* h_Rin, int rLen, Record* h_projTable, int pLen, 
														   int numThread = 256, int numBlock = 256 );


//aggregation
extern "C" __declspec( dllexport ) int GPUCopy_AggMax( Record* h_Rin, int rLen, Record** d_Rout,
													  int numThread = 256, int numBlock = 1024 );

extern "C" __declspec( dllexport ) int GPUCopy_AggMin( Record* h_Rin, int rLen, Record** d_Rout, 
													  int numThread = 256, int numBlock = 1024 );

extern "C" __declspec( dllexport ) int GPUCopy_AggSum( Record* h_Rin, int rLen, Record** d_Rout, 
													  int numThread = 256, int numBlock = 1024 );

extern "C" __declspec( dllexport ) int GPUCopy_AggAvg( Record* h_Rin, int rLen, Record** d_Rout, 
													  int numThread = 256, int numBlock = 1024 );

//group by
extern "C" __declspec( dllexport ) int	GPUCopy_GroupBy( Record* h_Rin, int rLen, Record* h_Rout, int** h_startPos, 
					int numThread = 64, int numBlock = 1024 );

//agg after group by
extern "C" __declspec( dllexport ) void GPUCopy_agg_max_afterGroupBy( Record* h_Rin, int rLen, int* h_startPos, int numGroups, Record* h_Ragg, int* h_aggResults, 
								  int numThread = 256 ); 

extern "C" __declspec( dllexport ) void GPUCopy_agg_min_afterGroupBy( Record* h_Rin, int rLen, int* h_startPos, int numGroups, Record* h_Ragg, int* h_aggResults, 
								  int numThread = 256 ); 

extern "C" __declspec( dllexport ) void GPUCopy_agg_sum_afterGroupBy( Record* h_Rin, int rLen, int* h_startPos, int numGroups, Record* h_Ragg, int* h_aggResults, 
								  int numThread = 256 ); 

extern "C" __declspec( dllexport ) void GPUCopy_agg_avg_afterGroupBy( Record* h_Rin, int rLen, int* h_startPos, int numGroups, Record* h_Ragg, int* h_aggResults, 
								  int numThread = 256 ); 

//data structure
extern "C" __declspec( dllexport ) void GPUCopy_BuildHashTable( Record* h_R, int rLen, int intBits, Bound* h_bound );

extern "C" __declspec( dllexport ) int GPUCopy_BuildTreeIndex( Record* h_Rin, int rLen, CUDA_CSSTree** tree );

extern "C" __declspec( dllexport ) int GPUCopy_HashSearch( Record* h_R, int rLen, Bound* h_bound, int inBits, Record* h_S, int sLen, Record** h_Rout, 
														  int numThread = 512 );

extern "C" __declspec( dllexport ) int GPUCopy_TreeSearch( Record* h_Rin, int rLen, CUDA_CSSTree* tree, Record* h_Sin, int sLen, Record** h_Rout );

//for joins

//sort
extern "C" __declspec( dllexport ) void GPUCopy_bitonicSort( Record* h_Rin, int rLen, Record* h_Rout, 
															int numThreadPB = 128, int numBlock = 1024 );

extern "C" __declspec( dllexport ) void GPUCopy_QuickSort( Record* h_Rin, int rLen, Record* h_Rout );

extern "C" __declspec( dllexport ) void GPUCopy_RadixSort( Record* h_Rin, int rLen, Record* h_Rout );

//join

extern "C" __declspec( dllexport ) int GPUCopy_ninlj( Record* h_R, int rLen, Record* h_S, int sLen, Record** h_Rout );

extern "C" __declspec( dllexport ) int	GPUCopy_smj( Record* h_R, int rLen, Record* h_S, int sLen, Record** h_Joinout );

extern "C" __declspec( dllexport ) int GPUCopy_hj( Record* h_Rin, int rLen, Record* d_Sin, int sLen, Record** h_Rout );

extern "C" __declspec( dllexport ) int GPUCopy_inlj( Record* h_Rin, int rLen, CUDA_CSSTree* h_tree, Record* h_Sin, int sLen, Record** h_Rout );

extern "C" __declspec( dllexport ) int GPUCopy_mj( Record* h_Rin, int rLen, Record* h_Sin, int sLen, Record** h_Joinout );

//partition
extern "C" __declspec( dllexport ) void GPUCopy_Partition( Record* h_Rin, int rLen, int numPart, Record* d_Rout, int* d_startHist, 
														  int numThreadPB = -1, int numBlock = -1 );


//Interface 1: get all RIDs into an array. You need to allocate d_RIDList. 
extern "C" __declspec( dllexport ) void GPUOnly_getRIDList(Record* d_Rin, int rLen, int** d_RIDList, int numThreadPerBlock = 512, int numBlock = 256);

// Interface 2: copy a relation to another relation. You need to allocate d_destRin.
extern "C" __declspec( dllexport ) void GPUOnly_copyRelation(Record* d_srcRin, int rLen, Record** d_destRin, int numThreadPerBlock = 512, int numBlock = 256);

// Interface3: set the RID according to the RID list.
extern "C" __declspec( dllexport ) void GPUOnly_setRIDList(int* d_RIDList, int rLen, Record* d_destRin, int numThreadPerBlock = 512, int numBlock = 256);

extern "C" __declspec( dllexport ) void GPUOnly_setValueList(int* d_ValueList, int rLen, Record* d_destRin, int numThreadPerBlock = 512, int numBlock = 256);

extern "C" __declspec( dllexport ) void GPUOnly_getValueList(Record* d_Rin, int rLen, int** d_ValueList, int numThreadPerBlock = 512, int numBlock = 256);

extern "C" __declspec( dllexport ) void resetGPU();
 
#endif

