#include "stdafx.h"
#include "CC_CSSTree.h"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "LinkedList.h"


//return the start position of searching the key.
int CC_CSSTree::search(int key)
{
	int i=0;
	int curIndex=0;
	int curNode=0;
	int j=0;
	int oldJ=0;
	//search
	for(i=0;i<level;i++)
	{
		for(j=0;j<blockSize;j++)
		{
			if(ntree[curIndex+j]==-1)
				break;
			if(key<=ntree[curIndex+j])
				break;
		}
		curNode=(fanout*(curNode)+j+1);
		curIndex=curNode*blockSize;
//#ifdef DEBUG
//		cout<<curNode<<", "<<j<<", "<<ntree[curIndex]<<";   ";
//#endif
	}
	curIndex=(curNode-numNode)*blockSize;
	if(curIndex>numRecord) curIndex=numRecord-1;
	//cout<<"I: "<<curIndex<<", ";//cout<<endl;
	return curIndex;
}

int cc_constructCSSTree(Record *Rin, int rLen, CC_CSSTree **tree)
{
	__DEBUG__("cc_constructCSSTree");
	*tree=new CC_CSSTree(Rin,rLen,CSS_TREE_FANOUT);
	return (*tree)->numNode;
}

int cc_equiTreeSearch(Record *Rin, int rLen, CC_CSSTree *tree, int keyForSearch, Record** Rout)
{
	__DEBUG__("cc_equiTreeSearch");
	int result=0;
	int i=0;
	int curIndex=tree->search(keyForSearch);
	cout<<curIndex<<", ";
	LinkedList *ll=(LinkedList*)malloc(sizeof(LinkedList));
	ll->init();
	for(i=curIndex-1;i>0;i--)
		if(Rin[i].value==keyForSearch)
		{
			ll->fill(Rin+i);
			result++;
		}
		else
			if(Rin[i].value<keyForSearch)
			break;
	for(i=curIndex;i<rLen;i++)
		if(Rin[i].value==keyForSearch)
		{
			ll->fill(Rin+i);
			result++;
		}
		else
			if(Rin[i].value>keyForSearch)
			break;
	(*Rout)=(Record*)malloc(sizeof(Record)*result);
	ll->copyToArray((*Rout));
	delete ll;
	return result;
}

int cc_multi_equiTreeSearch(Record *Rin, int rLen, CC_CSSTree *tree, int* searchKeys, int nKey, Record** Rout)
{
	__DEBUG__("cc_multi_equiTreeSearch");
	int result=0;
	int i=0;
	LinkedList *ll=(LinkedList*)malloc(sizeof(LinkedList));
	ll->init();
	int k=0;
	int curIndex=0;
	int keyForSearch;
	for(k=0; k<nKey; k++)
	{
		keyForSearch=searchKeys[k];
		curIndex=tree->search(keyForSearch);
		for(i=curIndex-1;i>0;i--)
			if(Rin[i].value==keyForSearch)
			{
				ll->fill(Rin+i);
				result++;
			}
			else
				if(Rin[i].value<keyForSearch)
				break;
		for(i=curIndex;i<rLen;i++)
			if(Rin[i].value==keyForSearch)
			{
				ll->fill(Rin+i);
				result++;
			}
			else
				if(Rin[i].value>keyForSearch)
				break;
	}
	(*Rout)=(Record*)malloc(sizeof(Record)*result);
	ll->copyToArray((*Rout));
	delete ll;
	return result;
}



