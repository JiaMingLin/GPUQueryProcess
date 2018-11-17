#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "QueryPlanNode.h"
#include "QueryPlanTree.h"
#include "QP_Utility.h"
#include "time.h"
#include "assert.h"
#include "Database.h"
#include "SingularThreadOp.h"
#include "SortThreadOp.h"
#include "BinaryThreadOp.h"
#include "GroupByThreadOp.h"
#include <iostream>
using namespace std;

extern Database *easedb;


/*
1. we do not support group by yet.
2. //currently we only support range query [] and point query.
3. we only support one column selection.
4. if the table has index, we always put it before the other table. 
*/

void QueryPlanNode::initialNode(char * str, ExecStatus *status, bool pGPUONLY_QP)
{
	planStatus=status;
	GPUONLY_QP=pGPUONLY_QP;

	int i = 0, j, k;

	if (str[0] == '$')
		return;

	for (j = i; str[j] != ';'; j++);
	str[j] = '\0';
	if (strcmp(str, "SEL") == 0)
		optType = SELECTION;
	else if (strcmp(str, "JOIN") == 0)
		optType = TYPE_JOIN;
	else if (strcmp(str, "PRO") == 0)
		optType = PROJECTION;
	else if (strcmp(str, "AGG") == 0)
		optType = TYPE_AGGREGATION;
	else if (strcmp(str, "ORD") == 0)
		optType = ORDER_BY;
	else if (strcmp(str, "GRP") == 0)
		optType = GROUP_BY;

	i = j + 1;

	for (j = i; str[j] != ';'; j++);
	str[j] = '\0';
	//table1 = new char[j - i + 1];
	table1=(char*)malloc(sizeof(char)*(j - i + 1));
	strcpy(table1, str + i);
	i = j + 1;

	for (j = i; str[j] != ';'; j++);
	str[j] = '\0';
	//table2 = new char[j - i + 1];
	table2=(char*)malloc(sizeof(char)*(j - i + 1));
	strcpy(table2, str + i);

	int num = 0;
	i = j + 1;
	for (j = i; str[j] != ';'; j++)
		if (str[j] == ',')
			num++;
	num_col = num;
	//columns = new char * [num];
	columns=(char**)malloc(sizeof(char*)*num);
	for (k = 0; k < num; k++)
	{
		for (j = i; str[j] != ','; j++);
		str[j] = '\0';
		//columns[k] = new char[j - i + 1];
		columns[k]=(char*)malloc(sizeof(char)*(j - i + 1));
		strcpy(columns[k], str + i);
		i = j + 1;
	}
	i++;

	k = 0;
	predicateRoot = new PredicateTree();
	predicateRoot->root=predicateRoot->construct_predicate_tree(str + i, &k,num_col,columns);	
	//check the detail type.
	//1. aggregtion, type is in table2.
	
	nodeStatus=STATUS_UNDONE;
	tOp=NULL;
	//this->createOp();
}






QueryPlanNode::QueryPlanNode()
{
	optType = TYPE_UNKNOWN;
	table1 = NULL;
	table2 = NULL;
	columns = NULL;
	num_col = 0;
}

QueryPlanNode::~QueryPlanNode()
{
	if(tOp!=NULL)
		delete tOp;
}

void QueryPlanNode::execute(int nodeLevel)
{
	//cout<<"level: "<<nodeLevel<<", "<<OpToString(this->optType)<<endl;
}

void QueryPlanNode::createOp()
{
	if(optType==TYPE_AGGREGATION)
	{
		if(planStatus->hasGroupBy==false)
		{
			if(strcmp(table2,"SUM")==0)
				optType=AGG_SUM;
			else if(strcmp(table2,"AVG")==0) 
				optType=AGG_AVG;
			else if(strcmp(table2,"MIN")==0) 
				optType=AGG_MIN;
			else if(strcmp(table2,"MAX")==0) 
				optType=AGG_MAX;
			tOp=new SingularThreadOp(optType);
		}
		else
		{
			if(strcmp(table2,"SUM")==0)
				optType=AGG_SUM_AFTER_GROUP_BY;
			else if(strcmp(table2,"AVG")==0) 
				optType=AGG_AVG_AFTER_GROUP_BY;
			else if(strcmp(table2,"MIN")==0) 
				optType=AGG_MIN_AFTER_GROUP_BY;
			else if(strcmp(table2,"MAX")==0) 
				optType=AGG_MAX_AFTER_GROUP_BY;
			tOp=new AggAfterGroupByThreadOp(optType);
		}
	}
	else if(optType==SELECTION)//we need to get the matching key values.
	{
		tOp=new SelectionOp(optType);
	}
	else if(optType==TYPE_JOIN)
	{
		Record* Sin=NULL;
		int sLen=0;
		optType=getJoinType();
		if(optType==JOIN_INLJ)
		{
			tOp=new IndexJoinThreadOp(optType);
		}
		else
		{
			tOp=new BinaryThreadOp(optType);
		}
	}
	else if(optType==ORDER_BY)
	{
		tOp=new SortThreadOp(optType);
	}
	else if(optType==PROJECTION)
	{
		tOp=new ProjectionOp(optType);
	}
	else if(optType==GROUP_BY)
	{
		tOp=new GroupByThreadOp(optType);		
	}
}


void QueryPlanNode::initOp(bool pGPUONLY_QP)
{
	GPUONLY_QP=pGPUONLY_QP;
	Record* Rin=NULL;
	int rLen=0;
	if(optType>=AGG_SUM && optType<=AGG_COUNT)
	{
		ID0=planStatus->getTableID(table1,columns[0]);
		rLen=planStatus->getDataTable(ID0,columns[0],&Rin);
		((SingularThreadOp*)tOp)->init(Rin,rLen,GPUONLY_QP);
	}
	else if(optType>=AGG_SUM_AFTER_GROUP_BY && optType<=AGG_MIN_AFTER_GROUP_BY)
	{
		ID0=planStatus->getTableID(table1,columns[0]);
		rLen=planStatus->getDataTable(ID0,columns[0],&Rin);
		((AggAfterGroupByThreadOp*)tOp)->init(Rin,rLen,
			planStatus->groupByRelation,planStatus->groupByRlen, planStatus->groupByNumGroup, planStatus->groupByStartPos,GPUONLY_QP);
	}
	else if(optType==SELECTION)//we need to get the matching key values.
	{
		int lowerKey=0, higherKey=0;
		getSelOprand(&lowerKey,&higherKey);
		assert(num_col==1);		
		ID0=planStatus->getTableID(table1,columns[0]);
		rLen=planStatus->getDataTable(ID0,columns[0],&Rin);
		((SelectionOp*)tOp)->init(Rin,rLen,lowerKey,higherKey,GPUONLY_QP);
	}
	else if(optType>=JOIN_NINLJ && optType<=JOIN_HJ)
	{
		Record* Sin=NULL;
		int sLen=0;
		if(optType==JOIN_INLJ)
		{
			//get the index on the CPU and the GPU
			ID0=planStatus->getTableID(table1,columns[0]);
			rLen=planStatus->getDataTable(ID0,columns[0],&Rin);	
			ID1=planStatus->getTableID(table2,columns[1]);
			sLen=planStatus->getDataTable(ID1,columns[1],&Sin);
			CC_CSSTree *cpu_tree=NULL;
			CUDA_CSSTree *gpu_tree=NULL;
			//easedb->createTreeIndex(table1,columns[0]);
			easedb->getTreeIndex(columns[0],&cpu_tree, &gpu_tree);
			((IndexJoinThreadOp*)tOp)->init(Rin,rLen,cpu_tree,gpu_tree,Sin,sLen,GPUONLY_QP);
		}
		else
		{
			ID0=planStatus->getTableID(table1,columns[0]);
			rLen=planStatus->getDataTable(ID0,columns[0],&Rin);			
			ID1=planStatus->getTableID(table2,columns[1]);
			sLen=planStatus->getDataTable(ID1,columns[1],&Sin);
			//GPUDEBUG_Record(Sin,sLen);
			((BinaryThreadOp*)tOp)->init(Rin,rLen,Sin,sLen,GPUONLY_QP);
		}
	}
	else if(optType==ORDER_BY)
	{
		ID0=planStatus->getTableID(table1,columns[0]);
		rLen=planStatus->getDataTable(ID0,columns[0],&Rin);
		((SortThreadOp*)tOp)->init(Rin,rLen,GPUONLY_QP);
	}
	else if(optType==PROJECTION)
	{
		ID0=planStatus->getTableID(table1,columns[0]);
		rLen=planStatus->getBaseTable(ID0,columns[0],&Rin);
		int* RIDList=NULL;
		int RIDLen=planStatus->getRIDList(ID0,columns[0],&RIDList);
		((ProjectionOp*)tOp)->init(Rin,rLen,RIDList,RIDLen,GPUONLY_QP);
	}
	else if(optType==GROUP_BY)
	{
		planStatus->hasGroupBy=true;
		ID0=planStatus->getTableID(table1,columns[0]);
		rLen=planStatus->getDataTable(ID0,columns[0],&Rin);
		((GroupByThreadOp*)tOp)->init(Rin,rLen,GPUONLY_QP);		
	}
}


void QueryPlanNode::PostExecution()
{
	Record* Rin=NULL;
	int rLen=0;
	DATA_RESIDENCE dataStore=DATA_ON_CPU;
	if(this->GPUONLY_QP==true)
		dataStore=DATA_ON_GPU;

	if(optType==TYPE_AGGREGATION)
	{
		cout<<"post execution is not required for"<<OpToString(optType,EXEC_CPU)<<endl;
	}
	else if(optType==SELECTION)//we need to get the matching key values.
	{		
		planStatus->addDataTable(ID0,tOp->Rout,tOp->numResult,dataStore);
	}
	else if(optType==JOIN_NINLJ||optType==JOIN_INLJ||optType==JOIN_SMJ||optType==JOIN_HJ)
	{
		planStatus->addJoinTable(ID0,ID1,tOp->Rout,tOp->numResult);
	}
	else if(optType==ORDER_BY)
	{
		planStatus->addDataTable(ID0,tOp->Rout,tOp->numResult,dataStore);
	}
	else if(optType==PROJECTION)
	{
		planStatus->addDataTable(ID0,tOp->Rout,tOp->numResult,dataStore);
	}
	else if(optType==GROUP_BY)
	{
		planStatus->addDataTable(ID0,tOp->Rout,tOp->numResult,dataStore);
		planStatus->groupByStartPos=((GroupByThreadOp*)tOp)->startPos;
		planStatus->groupByNumGroup=((GroupByThreadOp*)tOp)->numGroup;
		planStatus->groupByRelation=((GroupByThreadOp*)tOp)->Rout;
		planStatus->groupByRlen=((GroupByThreadOp*)tOp)->rLen;
	}
}
ThreadOp* QueryPlanNode::getNextOp(EXEC_MODE eM)
{
	if(tOp==NULL)
		createOp();

	ThreadOp* resultOp=NULL;
	if(tOp->isDone())
	{
		nodeStatus=STATUS_DONE;
	}
	else
		resultOp=tOp->getNextOp(eM);
	return resultOp;
}



//currently we only support range query [] and point query.
void QueryPlanNode::getSelOprand(int* lowerKey, int* higherKey)
{
	COMP_TYPE t1, t2;
	int leftopt;
	int rightopt;
	if (strcmp(predicateRoot->root->opt, "=") == 0)
	{
		leftopt = getOperandType(predicateRoot->root->left->opt);
		rightopt = getOperandType(predicateRoot->root->right->opt);
		if ((leftopt == OPT_COL && rightopt == OPT_NUM)
			|| (leftopt == OPT_NUM && rightopt == OPT_COL))
		{
			char *col, *num;
			t1=getCompare(predicateRoot->root,&col,&num);
			*lowerKey=*higherKey=atoi(num);
		}
	}
	if (strcmp(predicateRoot->root->opt, "AND") == 0)
	{
		char * t1col, * t1num, *t2col, *t2num;
		t1 = getCompare(predicateRoot->root->left, &t1col, &t1num);
		t2 = getCompare(predicateRoot->root->right, &t2col, &t2num);
		if (((t1 == CMP_BIGER && t2 == CMP_SMALLER) || (t1 == CMP_SMALLER && t2 == CMP_BIGER))
			&& strcmp(t1col, t2col) == 0)			
		{
			if(t1==CMP_BIGER)
			{
				*lowerKey=atoi(t1num);
				*higherKey=atoi(t2num);
			}
			else//t2 is the smaller
			{
				*lowerKey=atoi(t2num);
				*higherKey=atoi(t1num);
			}
		}
	}

}

OP_MODE QueryPlanNode::getJoinType(void)
{
	int leftopt;
	int rightopt;
	if (strcmp(predicateRoot->root->opt, "=") == 0)
	{
		leftopt = getOperandType(predicateRoot->root->left->opt);
		rightopt = getOperandType(predicateRoot->root->right->opt);
		int tableID=easedb->getTableID(columns[0]);
		bool hasIndex=false;//if one has tree index, we use INLJ
		if(easedb->tPro[tableID].cpu_treeindex!=0 &&  easedb->tPro[tableID].gpu_treeindex!=0)
			hasIndex=true;
		bool sorted=false;//if both relations are sorted, we use SMJ.
		if (leftopt == OPT_COL && rightopt == OPT_COL)
		{
			if(hasIndex)
				return JOIN_INLJ;
			else if(sorted)
				return JOIN_SMJ;
			else 
				return JOIN_HJ;
		}
	}
	return JOIN_NINLJ;
}
