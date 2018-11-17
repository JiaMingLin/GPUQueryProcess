#include <string.h>
#include "QueryPlanTree.h"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "windows.h"

#define ADA_START_TYPE (GROUP_BY)
#define ADA_END_TYPE (JOIN_HJ)

extern HANDLE gpuInUsedMutex;
extern bool isGPUavailable;
void QueryPlanTree::buildTree(char * str)
{
	int i = 0;
	root = construct_plan_tree(str, &i);
	curActiveNode=0;
	totalNumNode=0;
	Marshup(root);	
}

QueryPlanTree::~QueryPlanTree()
{
	int i=0;
	for(i=0;i<totalNumNode;i++)
		delete nodeVec[i];
	planStatus->destory();
	free(planStatus);
	hasLock=false;
}

QueryPlanNode * QueryPlanTree::construct_plan_tree(char * str, int * index)
{
	int i, j = *index;
	
	 if (str[j] == '$')
	{
		*index = j + 2;
		return NULL;
	}


	QueryPlanNode * node = new QueryPlanNode();

	for (i = j; str[i] != '\0'; i++)
	{
		if (str[i] == ':')
			break;
	}
	node->initialNode((char*)(str + j),this->planStatus, this->GPUONLY_QP);
	i++;

	//recursive
	node->left = construct_plan_tree(str, &i);
	node->right = construct_plan_tree(str, &i);
	*index = i;
	return node;
}
//get an operator and execute it.
//the status should be updated into the query plan.
void QueryPlanTree::execute(EXEC_MODE eM)
{
	ThreadOp* resultOp=getNextOp(eM);
	ThreadOp* previousOp=resultOp;
	QueryPlanNode* curNode=(QueryPlanNode*)(nodeVec[curActiveNode]);
	EXEC_MODE tempEM=eM;
	if(GPUONLY_QP)
	{
		if(!hasLock)
		{
			cout<<"*******waiting lock"<<endl;
			WaitForSingleObject( gpuInUsedMutex, INFINITE );
			hasLock=true;	
			cout<<"*******get lock"<<endl;
		}
	}
	while(resultOp!=NULL)
	{	
		//first, we get the type, next, we init the op.
		curNode=(QueryPlanNode*)(nodeVec[curActiveNode]);
		clock_t tt;
		startTimer(tt);
		if(eM==EXEC_ADAPTIVE)
		{
			if(curNode->optType>=ADA_START_TYPE && curNode->optType<=ADA_END_TYPE)
			{
				//obtained the gpuInUsedMutex
				//setHighPriority();				
				if(!hasLock)
				{
					cout<<"*******waiting lock"<<endl;
					WaitForSingleObject( gpuInUsedMutex, INFINITE );
					hasLock=true;	
					cout<<"*******get lock"<<endl;
				}
				isGPUavailable=false;							
				this->GPUONLY_QP=true;
				this->planStatus->GPUONLY_QP=true;
				curNode->initOp(this->GPUONLY_QP);
				resultOp->execute(EXEC_GPU_ONLY);
				eM=EXEC_GPU_ONLY;
			}
			else
			{
				curNode->initOp(this->GPUONLY_QP);
				resultOp->execute(EXEC_CPU);
			}
		}
		else
		{
			curNode->initOp(this->GPUONLY_QP);
			resultOp->execute(eM);
		}
		if(this->GPUONLY_QP)
			endTimer(OpToString(curNode->optType,EXEC_GPU_ONLY),tt);
		else
			endTimer(OpToString(curNode->optType,tempEM),tt);
		curNode->PostExecution();
		previousOp=resultOp;
		resultOp=getNextOp(eM);		
	}
	//store the result;
	q_Rout=previousOp->Rout;
	q_numResult=previousOp->numResult;
}



ThreadOp* QueryPlanTree::getNextOp(EXEC_MODE eM)
{
	ThreadOp* resultOp=NULL;
	
	if(totalNumNode>curActiveNode)
	{	
		resultOp=((QueryPlanNode*)(nodeVec[curActiveNode]))->getNextOp(eM);
		if(resultOp==NULL)
		{
			if((curActiveNode+1)==totalNumNode) //the last node, output result
			{
				QueryPlanNode* curNode=nodeVec[curActiveNode];
				int numResult=curNode->tOp->numResult;
if( GPUONLY_QP || eM==EXEC_GPU_ONLY)
{
				Record *RoutOnGPU=curNode->tOp->Rout;
#ifdef DEBUG
				int i=0;
				Record* Rout=NULL;
				cout<<numResult<<endl;
				if(numResult!=0)
				{
					CPUAllocateByCUDA((void**)&Rout,sizeof(Record)*numResult);
					CopyGPUToCPU(Rout,RoutOnGPU,sizeof(Record)*numResult);
					for(i=0;i<numResult;i++)
					{
						cout<<Rout[i].rid<<", "<<Rout[i].value<<"\t";
					}
					CPUFreeByCUDA(Rout);
				}
#endif
}
else
{
				Record *Rout=curNode->tOp->Rout;
#ifdef DEBUG
				int i=0;
				cout<<numResult<<endl;
				if(numResult!=0)
				{
					for(i=0;i<numResult;i++)
					{
						cout<<Rout[i].rid<<", "<<Rout[i].value<<"\t";
					}
				}
#endif
}//end if GPUQP
			}
			else
			{
				//update the plan status here.
				curActiveNode++;
				if(totalNumNode>curActiveNode)
				resultOp=((QueryPlanNode*)(nodeVec[curActiveNode]))->getNextOp(eM);
			}
		}
	}
	return resultOp;
}

void QueryPlanTree::Marshup(QueryPlanNode * node)
{
	QueryPlanNode * left=node->left;
	QueryPlanNode * right=node->right;
	if (left != NULL)
	{
		Marshup(left);
	}
	if (right != NULL)
	{
		Marshup(right);
	}
	nodeVec.push_back(node);
	totalNumNode++;
}

