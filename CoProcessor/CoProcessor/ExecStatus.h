#ifndef EXEC_STATUS
#define EXEC_STATUS

#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "db.h"
#include "QP_Utility.h"
#include "Database.h"
#include "CoProcessor.h"

extern Database *easedb;

typedef enum{
	DATA_ON_CPU,
	DATA_ON_GPU,
	DATA_ON_UNKNOWN
}DATA_RESIDENCE;


struct ExecStatus
{
//attributes,
	//one RID list corresponds to one base table.
	char RIDTableNames [MAX_TABLE_PER_QUERY][NAME_MAX_LENGTH];
	int* RID_baseTable [MAX_TABLE_PER_QUERY];
	int RIDLen[MAX_TABLE_PER_QUERY];
	DATA_RESIDENCE dataRes[MAX_TABLE_PER_QUERY];
	int numTables;
	//one columnNames corresponds to one table, one base table, 	
	int* finalResult;
	int numResultColumn;
	int numResultRow;
	bool hasGroupBy;
	int* groupByStartPos;
	int groupByNumGroup;
	Record* groupByRelation;
	DATA_RESIDENCE grpRes;
	int groupByRlen;
	bool GPUONLY_QP;
	
//methods,
	int getTableID(char* tableName, char *columnName)
	{
		int id=-1;
		//find the exact column.
		for(int i=0;i<numTables;i++)
		{
			if(strcmp(tableName,RIDTableNames[i])==0)
			{
				id=i;
				break;
			}
		}
		//not found in the plan status, we fetch it from easedb.
		if(id==-1)
		{
			//otherwise, we find the base table.
			strcpy(RIDTableNames[numTables],tableName);
			id=numTables;
			RID_baseTable[id]=NULL;			
			numTables++;
		}
		return id;	
	}
 
	int getDataTable(int id, char* columnName, Record** Rout)
	{
		assert(id>=0 && id<numTables);
		int resultLen;
		if(RID_baseTable[id]==NULL)
		{
			easedb->getTable(columnName,Rout,&resultLen,GPUONLY_QP);
			RIDLen[id]=resultLen;
		}
		else
		{
			resultLen=RIDLen[id];
			if(GPUONLY_QP)
			{//need to make sure the input are in the GPU: RIDLIst
				if(resultLen>0)
				{
					GPUAllocate((void**)Rout, sizeof(Record)*resultLen);
					if(dataRes[id]==DATA_ON_CPU)
					{
						int* gpuRIDList;
						GPUAllocate((void**)&gpuRIDList, sizeof(int)*resultLen);
						CopyCPUToGPU(gpuRIDList,RID_baseTable[id],resultLen*sizeof(int));
						DATA_TO_GPU(resultLen*sizeof(int));
						Record* baseTable=NULL;
						//fetch a table on the CPU
						int rLen=0;
						easedb->getTable(columnName,&baseTable,&rLen,false);
						Record* cpuRout=new Record[resultLen];
						for(int i=0;i<resultLen;i++)
							cpuRout[i].rid=RID_baseTable[id][i];
						CPU_Projection(baseTable,RIDLen[id],cpuRout,resultLen,1);
						CopyCPUToGPU(*Rout,cpuRout,resultLen*sizeof(Record));
						DATA_TO_GPU(resultLen*sizeof(Record));
						delete cpuRout;
						delete RID_baseTable[id];
						RID_baseTable[id]=gpuRIDList;
						dataRes[id]=DATA_ON_GPU;
						
						//since are going to do the projection, let's make it on the CPU.
						//and transfer the RIDList and the base table to the GPU.
					}
					else
					{
						GPUOnly_setRIDList(RID_baseTable[id],resultLen,*Rout);
						Record* baseTable=NULL;
						int rLen;
						easedb->getTable(columnName,&baseTable,&rLen,GPUONLY_QP);
						GPUOnly_Projection(baseTable,rLen,*Rout,resultLen);	
						//GPUDEBUG_Record(*Rout,resultLen);
						GPUFree(baseTable);
					}
				}
				else
					*Rout=NULL;
			}
			else
			{
				*Rout=new Record[resultLen];
				int i=0;
				if(dataRes[id]==DATA_ON_GPU)
				{
					int* cpuRIDList;
					CPUAllocateByCUDA((void**)&cpuRIDList,sizeof(int)*resultLen);
					CopyGPUToCPU(cpuRIDList,RID_baseTable[id],sizeof(int)*resultLen);
					DATA_FROM_GPU(sizeof(int)*resultLen);
					GPUFree(RID_baseTable[id]);
					RID_baseTable[id]=cpuRIDList;
				}
				for(i=0;i<resultLen;i++)
					(*Rout)[i].rid=RID_baseTable[id][i];
				Record* baseTable=NULL;
				int rLen;
				easedb->getTable(columnName,&baseTable,&rLen,GPUONLY_QP);
				CPU_Projection(baseTable,RIDLen[id],*Rout,resultLen,1);
			}
		}
		return resultLen;
	}
	//always get the original table. we don't need to adapt.
	int getBaseTable(int id, char* columnName, Record** Rout)
	{
		assert(id>=0 && id<numTables);
		int rLen;
		easedb->getTable(columnName,Rout,&rLen,GPUONLY_QP);
		return rLen;
	}
	//this one needs also assert the GPUONLY_QP and dataRes.
	int getRIDList(int id, char* columnName,int** RIDList)
	{
		assert(id>=0 && id<numTables);
		assert((dataRes[id]==DATA_ON_GPU && GPUONLY_QP) || (dataRes[id]==DATA_ON_CPU && (!GPUONLY_QP)));
		if(RID_baseTable[id]==NULL)
		{
			Record *Rout;
			int resultLen=0, i=0;
			easedb->getTable(columnName,&Rout,&resultLen,GPUONLY_QP);
			if( GPUONLY_QP)
			{
				GPUOnly_getRIDList(Rout,resultLen,&(RID_baseTable[id]));		
			}
			else
			{
				RID_baseTable[id]=new int[resultLen];
				for(i=0;i<resultLen;i++)
					RID_baseTable[id][i]=Rout[i].rid;			
			}
			RIDLen[id]=resultLen;
		}
		*RIDList=RID_baseTable[id];
		return RIDLen[id];
	}

	void addJoinTable(int ID1, int ID2, Record *dt,int rLen)
	{
		assert(ID1>=0 && ID1<numTables);
		assert(ID2>=0 && ID2<numTables);
//		assert((dataRes[ID1]==DATA_ON_GPU && GPUONLY_QP) || (dataRes[ID1]==DATA_ON_CPU && (!GPUONLY_QP)));
//		assert((dataRes[ID2]==DATA_ON_GPU && GPUONLY_QP) || (dataRes[ID2]==DATA_ON_CPU && (!GPUONLY_QP)));
if( GPUONLY_QP)
{
		if(RID_baseTable[ID1]!=NULL)
			GPUFree(RID_baseTable[ID1]);
		if(RID_baseTable[ID2]!=NULL)
			GPUFree(RID_baseTable[ID2]);
		if(rLen<=0)
		{
			GPUAllocate((void**)&(RID_baseTable[ID1]),sizeof(int));
			GPUAllocate((void**)&(RID_baseTable[ID2]),sizeof(int));
		}
		else 
		{
			GPUOnly_getRIDList(dt,rLen,&(RID_baseTable[ID1]));
			GPUOnly_getValueList(dt,rLen,&(RID_baseTable[ID2]));
		}
		dataRes[ID1]=DATA_ON_GPU;
		dataRes[ID2]=DATA_ON_GPU;
}
else
{
		if(RID_baseTable[ID1]!=NULL)
			delete RID_baseTable[ID1];
		if(RID_baseTable[ID2]!=NULL)
			delete RID_baseTable[ID2];
		if(rLen==0)
		{
			RID_baseTable[ID1]=new int[1];
			RID_baseTable[ID2]=new int[1];
		}
		else 
		{
			RID_baseTable[ID1]=new int[rLen];
			RID_baseTable[ID2]=new int[rLen];
		}
		int i=0;
		for(i=0;i<rLen;i++)
		{
			RID_baseTable[ID1][i]=dt[i].rid;
			RID_baseTable[ID2][i]=dt[i].value;
		}
		dataRes[ID1]=DATA_ON_CPU;
		dataRes[ID2]=DATA_ON_CPU;
}
		RIDLen[ID1]=rLen;
		RIDLen[ID2]=rLen;
	}
//add table??? we don't know whether it is from gpu or cpu.
	void addDataTable(int id, Record *dt,int rLen, DATA_RESIDENCE dataStorePlace)
	{
		assert(id>=0 && id<numTables);
if( GPUONLY_QP)
{
		if(RID_baseTable[id]!=NULL)
		{
			if(dataRes[id]==DATA_ON_GPU)
				GPUFree(RID_baseTable[id]);
			else
				delete RID_baseTable[id];
		}
		if(rLen<=0)
			GPUAllocate((void**)&(RID_baseTable[id]),sizeof(int));
		else 
		{
			Record *tempDT=dt;
			if(dataStorePlace==DATA_ON_CPU)
			{
				GPUAllocate((void**)&tempDT,rLen*sizeof(Record));
				CopyCPUToGPU(tempDT,dt,rLen*sizeof(Record));	
				DATA_TO_GPU(rLen*sizeof(Record));
			}
			GPUOnly_getRIDList(tempDT,rLen,&(RID_baseTable[id]));
		}
		dataRes[id]=DATA_ON_GPU;
}
else
{
		if(RID_baseTable[id]!=NULL)
		{
			if(dataRes[id]==DATA_ON_CPU)
				delete RID_baseTable[id];
			else
				GPUFree(RID_baseTable[id]);
		}
		if(rLen==0)
			RID_baseTable[id]=new int[1];
		else 
			RID_baseTable[id]=new int[rLen];
		Record *tempDT=dt;
		if(dataStorePlace==DATA_ON_GPU)
		{
			CPUAllocateByCUDA((void**)&tempDT,rLen*sizeof(Record));
			CopyGPUToCPU(tempDT,dt,rLen*sizeof(Record));
			DATA_FROM_GPU(rLen*sizeof(Record));
		}
		int i=0;
		for(i=0;i<rLen;i++)
			RID_baseTable[id][i]=tempDT[i].rid;
		dataRes[id]=DATA_ON_CPU;
}
		RIDLen[id]=rLen;
	}
	void addRIDList(int id, int *dt,int rLen, DATA_RESIDENCE dataStorePlace)
	{
		assert(id>=0 && id<numTables);
		if(dataRes[id]==DATA_ON_GPU)
			GPUFree(RID_baseTable[id]);
		else
			free(RID_baseTable[id]);
		//even in adaptive mode, we change the execution mode and then addRIDList.
		//it assert: (dataStorePlace==DATA_ON_GPU && GPUONLY_QP) || (dataStorePlace==DATA_ON_CPU && (!GPUONLY_QP))
		assert((dataStorePlace==DATA_ON_GPU && GPUONLY_QP) || (dataStorePlace==DATA_ON_CPU && (!GPUONLY_QP)));
		RID_baseTable[id]=dt;
		RIDLen[id]=rLen;
	}
	void init(bool pGPUONLY_QP)
	{
		GPUONLY_QP=pGPUONLY_QP;
		numTables=0;
		groupByStartPos=NULL;
		for(int i=0;i<MAX_TABLE_PER_QUERY;i++)
		{
			RID_baseTable[i]=NULL;
			dataRes[i]=DATA_ON_UNKNOWN;
		}
		finalResult=NULL;
		numResultColumn=-1;
		numResultRow=-1;
		hasGroupBy=false;
	}
	void destory()
	{
		for(int i=0;i<MAX_TABLE_PER_QUERY;i++)
		{
			if(RID_baseTable[i]!=NULL)
			{
				if(dataRes[i]==DATA_ON_GPU)
					GPUFree(RID_baseTable[i]);
				else
					delete RID_baseTable[i];
			}
		}
	}
};


#endif


