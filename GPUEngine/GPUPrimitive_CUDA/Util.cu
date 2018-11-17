
//Ke Yang

#ifndef _Util_CPP_
#define _Util_CPP_

//includes, within project
#include "Header.cu"
#include "LinkedList.cu"

//generate nRec random records, with distinct val in (0, maxVal) if bDistinct.
void NewRelation(Record* pRec, const int nRec, const unsigned int maxVal)
{
#define Rand31(maxVal) ((((rand() & 1)<<30) + (rand()<<15) + rand()) % (maxVal)) //can generate 0~32bits
	printf("randomizing records...\t");
	//maxRVal can be any. but when bDist is true, maxVal can't < nR
	unsigned int randVal;
	for(int i = 0; i < nRec; ++i)
	{
		randVal = Rand31(maxVal);
		pRec[i].x = i;
		pRec[i].y = randVal;
	}
	printf("finished.\n");
}

//prefix sum of an int array. unlike cuda-prescan, it needs no additional buffer
void PresumFragArray_GT(int* A, const int nA)
{
	int val, sum = 0;
	for(int i = 0; i < nA; ++i)
	{
		val = A[i];
		A[i] = sum;
		sum += val;
	}
}

////////////////////////////added radix-sort merge
void RadixSort_GT(Record* pSrcR, const int nR)
{
	//internal pingpong buffer
	Record* pDstR = (Record*) malloc(nR * sizeof(Record));

	//keys are assumed to be int32.
	const int radix = 8; //num of bits for a radix, should completely divide 32. 
	const int buckSize = 1 << radix;
	const int radixMask = (1 << radix) - 1;
	int* histo = (int*) malloc(buckSize * sizeof(int));
	int sum, val;
	for(int pass = 0; pass < (32 / radix); ++pass)
	{
		//reset
		for(int i = 0; i < buckSize; ++i)
			histo[i] = 0;
		//histo
		for(int rIdx = 0; rIdx < nR; ++rIdx)
		{
			unsigned char digi = (unsigned char)((pSrcR[rIdx].y >> (pass * radix)) & radixMask);	//uchar
			histo[digi] = histo[digi] + 1;
		}
		//cumulative histo
		sum = 0;
		for(unsigned int digiIdx = 0; digiIdx < buckSize; ++digiIdx)
		{
			val = histo[digiIdx];
			histo[digiIdx] = sum;
			sum = sum + val;
		}
		//rearrange(sort)
		for(int rIdx = 0; rIdx < nR; ++rIdx)
		{
			unsigned char digi = (unsigned int)((pSrcR[rIdx].y >> (pass * radix)) & radixMask);
			pDstR[histo[digi]] = pSrcR[rIdx];
			histo[digi] = histo[digi] + 1;
		}
		//swap pingpong buffer pointers
		Record* pTmp = pSrcR;
		pSrcR = pDstR;
		pDstR = pTmp;
	}
	SAFE_FREE(histo);
	SAFE_FREE(pDstR);
}

//merge the sorted relations R, S into RS (assume space is enough), refer to Raghu's DBMS textbook
int MergeSorted_GT(Record** pRS, Record* pSrcR, const int nR, Record* pSrcS, const int nS)
{
	int nRS = 0;
	LinkedList *ll=(LinkedList*)malloc(sizeof(LinkedList)); 
	ll->init();
	Record tmpRS;

	Record* pR = pSrcR;
	Record* pS = pSrcS;
	Record* pS1;
	while((pR < pSrcR + nR) && (pS < pSrcS + nS))
	{
		while((pS < pSrcS + nS) && (pR->y > pS->y))
			++pS;
		while((pR < pSrcR + nR) && (pR->y < pS->y))
			++pR;

		//=: might has duplications with the same key. nestloop: R only 1 pass, S multiple passes
		pS1 = pS;	// Backup the start pos of duplications
		while((pR < pSrcR + nR) && (pR->y == pS->y))
		{
			pS1 = pS; //rewind to the start of dup
			while((pS1 < pSrcS + nS) && (pS1->y == pR->y))
			{
				//output rs
				tmpRS.x = pR->x;
				tmpRS.y = pS1->x;
				ll->fill(&tmpRS);

				++nRS;
				++pS1;
			}
			++pR;
		}
	}

	*pRS = (Record*)malloc(sizeof(Record) * nRS);
	ll->copyToArray(*pRS);
	ll->destroy();
	if(ll)
	{
		SAFE_FREE(ll);
		ll = NULL;
	}
	return nRS;
}

//radix sort all 32bits of R, S in 4 passes of 8bit-radices, merge to output
int RadixSortMergeJoin_GT(Record** pGold, Record* h_R, const unsigned int nR, Record* h_S, const unsigned int nS)
{
	Record* pSrcR = (Record*)malloc(sizeof(Record) * nR);
	for(unsigned int rIdx = 0; rIdx < nR; ++rIdx)
		pSrcR[rIdx] = h_R[rIdx];
	//SAFE_FREE(h_R);
	//h_R = NULL;

	Record* pSrcS = (Record*)malloc(sizeof(Record) * nS);
	for(unsigned int rIdx = 0; rIdx < nS; ++rIdx)
		pSrcS[rIdx] = h_S[rIdx];
	//SAFE_FREE(h_S);
	//h_S = NULL;

	//1. radix sort R
	RadixSort_GT(pSrcR, nR);
	//2. radix sort S
	RadixSort_GT(pSrcS, nS);
	//3. merge
	int nRS = MergeSorted_GT(pGold, pSrcR, nR, pSrcS, nS);
 
	SAFE_FREE(pSrcR);
	SAFE_FREE(pSrcS);
	return nRS;
}



#endif
