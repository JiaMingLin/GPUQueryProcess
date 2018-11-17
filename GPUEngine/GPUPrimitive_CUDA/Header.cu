
#ifndef _HEADER_H_
#define _HEADER_H_

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <limits.h>	//rand
#include <assert.h>
#include <math.h>	//ceil

#include "cutil.h"

/////////////////////////////////////////////////////////////////////////defines
#ifndef mymax
#define mymax(a,b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef mymin
#define mymin(a,b) (((a) < (b)) ? (a) : (b))
#endif

#define SAFE_FREE(p) {if(p) {free(p); (p)=NULL;} };
#define SAFE_CUDA_FREE(p) {if(p) {CUDA_SAFE_CALL(cudaFree(p)); (p)=NULL;} };

#define IntCeilDiv(a, b) ( (int)ceilf((a) / float(b))	)

///////////////////////////////////////general define

//debug?
#define _bDebug 0

#ifdef BINARY_SEARCH_HASH
#define _bSortPart 1 //sort each partition, so probe will bisearch rather than scan the part
#else
#define _bSortPart 0 //sort each partition, so probe will bisearch rather than scan the part
#endif
#define _maxPartLen (_bDebug? 2: 512)	//max partition length. Limited by shared memory size (4k to be safe): sizeof(Rec) * maxPartLen <= 4k
#define HASH(v) (_bDebug ? ((unsigned int) v) : ((unsigned int)( (v >> 7) ^ (v >> 13) ^ (v >>21) ^ (v) )) )



#endif

