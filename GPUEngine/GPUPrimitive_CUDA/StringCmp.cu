#ifndef STRING_CMP_CU
#define STRING_CMP_CU
#include <common.cu>



//#define CHAR2_READ
#define CHAR4_READ
#ifndef CHAR4_READ
#ifndef CHAR2_READ
__device__ int compareString(const void *d_a, const void *d_b)
{
	char *str_a=(char*)d_a;
	char *str_b=(char*)d_b;
	int i=0;
	int result=0;
	char a, b;
	a=str_a[i];
	b=str_b[i];
	
	while(result==0)
	{	
		if((a=='\0') || (b=='\0') )
		{
			if(a==b)
				result=0;
			else
				if(a=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(a>b)
		{
			result=1;
			break;
		}
		else if(a<b)
		{
			result=-1;
			break;
		}
		i++;
		a=str_a[i];
		b=str_b[i];
	}
	//printf("%s, %s, %d \n", (char*)d_a,(char*)d_b,result);
	return result;
}
#else
__device__ int compareString(const void *d_a, const void *d_b)
{
	char2 *str_a=(char2*)d_a;
	char2 *str_b=(char2*)d_b;
	int i=0;
	int result=0;
	char2 cura, curb;
	cura=str_a[i];
	curb=str_b[i];
	char a, b;
	//int j=0;
	while(result==0)
	{	
		a=cura.x;b=curb.x;
		if((a=='\0') || (b=='\0') )
		{
			if(a==b)
				result=0;
			else
				if(a=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(a>b)
		{
			result=1;
			break;
		}
		else if(a<b)
		{
			result=-1;
			break;
		}
		//printf("%c, %c, %d; ", a,b,result);
		//y
		a=cura.y;b=curb.y;
		if((a=='\0') || (b=='\0') )
		{
			if(a==b)
				result=0;
			else
				if(a=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(a>b)
		{
			result=1;
			break;
		}
		else if(a<b)
		{
			result=-1;
			break;
		}
		i++;
		cura=str_a[i];
		curb=str_b[i];
	}	
	return result;
}
#endif//CHAR2_READ

#else//we are using this one.
__device__ int compareString(const void *d_a, const void *d_b)
{
	char4 *str_a=(char4*)d_a;
	char4 *str_b=(char4*)d_b;
	int i=0;
	int result=0;
	char4 cura, curb;
	cura=str_a[i];
	curb=str_b[i];
	char a, b;
	//int j=0;
	while(result==0)
	{	
		a=cura.x;b=curb.x;
		if((a=='\0') || (b=='\0') )
		{
			if(a==b)
				result=0;
			else
				if(a=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(a>b)
		{
			result=1;
			break;
		}
		else if(a<b)
		{
			result=-1;
			break;
		}
		//printf("%c, %c, %d; ", a,b,result);
		//y
		a=cura.y;b=curb.y;
		if((a=='\0') || (b=='\0') )
		{
			if(a==b)
				result=0;
			else
				if(a=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(a>b)
		{
			result=1;
			break;
		}
		else if(a<b)
		{
			result=-1;
			break;
		}
		//printf("%c, %c, %d; ", a,b,result);
		//z
		a=cura.z;b=curb.z;
		if((a=='\0') || (b=='\0') )
		{
			if(a==b)
				result=0;
			else
				if(a=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(a>b)
		{
			result=1;
			break;
		}
		else if(a<b)
		{
			result=-1;
			break;
		}
		//printf("%c, %c, %d; ", a,b,result);
		//w
		a=cura.w;b=curb.w;
		if((a=='\0') || (b=='\0') )
		{
			if(a==b)
				result=0;
			else
				if(a=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(a>b)
		{
			result=1;
			break;
		}
		else if(a<b)
		{
			result=-1;
			break;
		}
		
		i++;
		cura=str_a[i];
		curb=str_b[i];
	}	
	return result;
	//a good char4:)
	/*char4 *str_a=(char4*)d_a;
	char4 *str_b=(char4*)d_b;
	int i=0;
	int result=0;
	char4 cura, curb;
	cura=str_a[i];
	curb=str_b[i];
	while(result==0)
	{	
		//loop 0
		if((cura.x=='\0') || (curb.x=='\0'))
		{
			//if(a[0]=='\0' && b[0]=='\0')
			if(cura.x==curb.x)
				result=0;
			else
				if(cura.x=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(cura.x>curb.x)
		{
			result=1;
			break;
		}
		else if(cura.x<curb.x)
		{
			result=-1;
			break;
		}
		//loop 1
		if((cura.y=='\0') || (curb.y=='\0'))
		{
			//if(a[0]=='\0' && b[0]=='\0')
			if(cura.y==curb.y)
				result=0;
			else
				if(cura.y=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(cura.y>curb.y)
		{
			result=1;
			break;
		}
		else if(cura.y<curb.y)
		{
			result=-1;
			break;
		}
		//loop 2
		if((cura.z=='\0') || (curb.z=='\0'))
		{
			//if(a[0]=='\0' && b[0]=='\0')
			if(cura.z==curb.z)
				result=0;
			else
				if(cura.z=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(cura.z>curb.z)
		{
			result=1;
			break;
		}
		else if(cura.z<curb.z)
		{
			result=-1;
			break;
		}
		//loop 3
		if((cura.w=='\0') || (curb.w=='\0'))
		{
			//if(a[0]=='\0' && b[0]=='\0')
			if(cura.w==curb.w)
				result=0;
			else
				if(cura.w=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(cura.w>curb.w)
		{
			result=1;
			break;
		}
		else if(cura.w<curb.w)
		{
			result=-1;
			break;
		}		
		i++;
		cura=str_a[i];
		curb=str_b[i];			
	}
	return result;*/
	//unrolling
	/*char *str_a=(char*)d_a;
	char *str_b=(char*)d_b;
	int i=0;
	int result=0;
	char a[2];
	char b[2];
	a[0]=str_a[i];
	if(a[0]!='\0')
	a[1]=str_a[i+1];
	b[0]=str_b[i];	
	if(b[0]!='\0')
	b[1]=str_b[i+1];
	i=i+2;
	while(true)
	{	
		if(a[0]=='\0' || b[0]=='\0')
		{
			//if(a[0]=='\0' && b[0]=='\0')
			if(a[0]==b[0])
				result=0;
			else
				if(a[0]=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(a[0]>b[0])
		{
			result=1;
			break;
		}
		else if(a[0]<b[0])
		{
			result=-1;
			break;
		}

		if(a[1]=='\0' || b[1]=='\0')
		{
			//if(a[1]=='\0' && b[1]=='\0')
			if(a[1]==b[1])
				result=0;
			else
				if(a[1]=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(a[1]>b[1])
		{
			result=1;
			break;
		}
		else if(a[1]<b[1])
		{
			result=-1;
			break;
		}
		
		//
		a[0]=str_a[i];
		if(a[0]!='\0')
		a[1]=str_a[i+1];
		b[0]=str_b[i];	
		if(b[0]!='\0')
		b[1]=str_b[i+1];
		i=i+2;
	}
	return result;*/
	//loop unrolling
	//loop 0
		/*if(a[0]=='\0' || b[0]=='\0')
		{
			if(a[0]=='\0' && b[0]=='\0')
				result=0;
			else
				if(a[0]=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
			if(a[0]>b[0])
			{
				result=1;
				break;
			}
			else if(a[0]<b[0])
			{
				result=-1;
				break;
			}
		//loop 1
		if(a[1]=='\0' || b[1]=='\0')
		{
			if(a[1]=='\0' && b[1]=='\0')
				result=0;
			else
				if(a[1]=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
			if(a[1]>b[1])
			{
				result=1;
				break;
			}
			else if(a[1]<b[1])
			{
				result=-1;
				break;
			}*/
	//only char.
	/*char *str_a=(char*)d_a;
	char *str_b=(char*)d_b;
	int i=0;
	int result=0;
	char a, b;
	a=str_a[i];
	b=str_b[i];
	
	while(result==0)
	{	
		if((a=='\0') || (b=='\0') )
		{
			if(a==b)
				result=0;
			else
				if(a=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(a>b)
		{
			result=1;
			break;
		}
		else if(a<b)
		{
			result=-1;
			break;
		}
		i++;
		a=str_a[i];
		b=str_b[i];
	}
	//printf("%s, %s, %d \n", (char*)d_a,(char*)d_b,result);
	return result;*/
	/*char2 *str_a=(char2*)d_a;
	char2 *str_b=(char2*)d_b;
	int i=0;
	int result=0;
	char2 cura, curb;
	char a[2], b[2];
	cura=str_a[i];
	curb=str_b[i];	
	a[0]=cura.x;
	a[1]=cura.y;
	b[0]=curb.x;	
	b[1]=curb.y;	
	while(result==0)
	{	
		if((a[0]=='\0') || (b[0]=='\0'))
		{
			//if(a[0]=='\0' && b[0]=='\0')
			if(a[0]==b[0])
				result=0;
			else
				if(a[0]=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(a[0]>b[0])
		{
			result=1;
			break;
		}
		else if(a[0]<b[0])
		{
			result=-1;
			break;
		}

		if((a[1]=='\0') || (b[1]=='\0'))
		{
			//if(a[1]=='\0' && b[1]=='\0')
			if(a[1]==b[1])
				result=0;
			else
				if(a[1]=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(a[1]>b[1])
		{
			result=1;
			break;
		}
		else if(a[1]<b[1])
		{
			result=-1;
			break;
		}
		
		i++;
		cura=str_a[i];
		curb=str_b[i];	
		a[0]=cura.x;
		a[1]=cura.y;
		b[0]=curb.x;	
		b[1]=curb.y;		
	}
	return result;*/
	//the char4 version, there is some error:(
	/*char4 *str_a=(char4*)d_a;
	char4 *str_b=(char4*)d_b;
	int i=0;
	int result=0;
	char4 cura, curb;
	char a[4],b[4];
	cura=str_a[i];
	curb=str_b[i];	
	a[0]=cura.x;
	a[1]=cura.y;
	a[2]=cura.z;
	a[3]=cura.w;
	b[0]=curb.x;
	b[1]=curb.y;
	b[2]=curb.z;
	b[3]=curb.w;
	while(result==0)
	{	
		//loop 0
		if((a[0]=='\0') || (b[0]=='\0'))
		{
			//if(a[0]=='\0' && b[0]=='\0')
			if(a[0]==b[0])
				result=0;
			else
				if(a[0]=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(a[0]>b[0])
		{
			result=1;
			break;
		}
		else if(a[0]<b[0])
		{
			result=-1;
			break;
		}
		//loop 1
		if((a[1]=='\0') || (b[1]=='\0'))
		{
			//if(a[1]=='\0' && b[1]=='\0')
			if(a[1]==b[1])
				result=0;
			else
				if(a[1]=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(a[1]>b[1])
		{
			result=1;
			break;
		}
		else if(a[1]<b[1])
		{
			result=-1;
			break;
		}
		//loop 2
		if((a[2]=='\0') || (b[2]=='\0'))
		{
			//if(a[1]=='\0' && b[1]=='\0')
			if(a[2]==b[2])
				result=0;
			else
				if(a[2]=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(a[2]>b[2])
		{
			result=1;
			break;
		}
		else if(a[2]<b[2])
		{
			result=-1;
			break;
		}
		//loop 3
		if((a[3]=='\0') || (b[3]=='\0'))
		{
			//if(a[1]=='\0' && b[1]=='\0')
			if(a[3]==b[3])
				result=0;
			else
				if(a[3]=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(a[3]>b[3])
		{
			result=1;
			break;
		}
		else if(a[3]<b[3])
		{
			result=-1;
			break;
		}
		
		i++;
		cura=str_a[i];
		curb=str_b[i];	
		a[0]=cura.x;
		a[1]=cura.y;
		a[2]=cura.z;
		a[3]=cura.w;
		b[0]=curb.x;
		b[1]=curb.y;
		b[2]=curb.z;
		b[3]=curb.w;		
	}
	return result;*/
	
	//the char version. 
	/*char *str_a=(char*)d_a;
	char *str_b=(char*)d_b;
	int i=0;
	int result=0;
	char a, b;
	a=str_a[i];
	b=str_b[i];
	while(a!='\0' && b!='\0' && result==0)
	{		
		if(a==b)
			i++;
		else
			if(a>b)
			{
				result=1;
			}
			else
			{
				result=-1;
			}
		if(result==0)
		{
			a=str_a[i];
			b=str_b[i];
		}
	}
	if(result==0)
	{
		if(a=='\0' && b=='\0')
			result=0;
		else
			if(a=='\0')
				result=-1;
			else
				result=1;
	}
	return result;*/
	/*char a[4], b[4];
	for(i=0;i<4&& result==0;i++)
	{
		a[i]=str_a[i];
		b[i]=str_b[i];
		
	}
	if(a[0]>b[0])
	{
		result=1;
	}
	else if(a[0]<b[0])
	{
		result=-1;
	}
	if(result==0)
	{
		if(a[1]>b[1])
		{
			result=1;
		}
		else if(a[1]<b[1])
		{
			result=-1;
		}
	}
	if(result==0)
	{
		if(a[2]>b[2])
		{
			result=1;
		}
		else if(a[2]<b[2])
		{
			result=-1;
		}
	}

	if()
	{
		result=1;
	}
	else if(a[1]<b[1])
	{
		result=-1;
	}*/
	/*i++;
	a=str_a[i];
	b=str_b[i];
	if(a>b)
	{
		result=1;
	}
	else if(a<b)
	{
		result=-1;
	}
	i++;
	if(str_a[i]>str_b[i])
	{
		result=1;
	}
	else if(str_a[i]<str_b[i])
	{
		result=-1;
	}
	i++;
	if(str_a[i]>str_b[i])
	{
		result=1;
	}
	else if(str_a[i]<str_b[i])
	{
		result=-1;
	}
	return result;*/
}
#endif

int compareStringCPU(const void *d_a, const void *d_b)
{
	/*char *str_a=(char*)d_a;
	char *str_b=(char*)d_b;
	int i=0;
	int result=0;
	while(str_a[i]!='\0' && str_b[i]!='\0' && result==0)
	{
		if(str_a[i]==str_b[i])
			i++;
		else
			if(str_a[i]>str_b[i])
			{
				result=1;
			}
			else
			{
				result=-1;
			}
	}
	if(result==0)
	{
		if(str_a[i]=='\0' && str_b[i]=='\0')
			result=0;
		else
			if(str_a[i]=='\0')
				result=-1;
			else
				result=1;
	}
	return result;*/
	/*char2 *str_a=(char2*)d_a;
	char2 *str_b=(char2*)d_b;
	int i=0;
	int result=0;
	//char2 cura, curb;
	char a[2], b[2];
	//cura=str_a[i];
	//curb=str_b[i];	
	a[0]=str_a[i].x;
	a[1]=str_a[i].y;
	b[0]=str_b[i].x;	
	b[1]=str_b[i].y;	
	while(true)
	{	
		if(a[0]=='\0' || b[0]=='\0')
		{
			//if(a[0]=='\0' && b[0]=='\0')
			if(a[0]==b[0])
				result=0;
			else
				if(a[0]=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(a[0]>b[0])
		{
			result=1;
			break;
		}
		else if(a[0]<b[0])
		{
			result=-1;
			break;
		}

		if(a[1]=='\0' || b[1]=='\0')
		{
			//if(a[1]=='\0' && b[1]=='\0')
			if(a[1]==b[1])
				result=0;
			else
				if(a[1]=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(a[1]>b[1])
		{
			result=1;
			break;
		}
		else if(a[1]<b[1])
		{
			result=-1;
			break;
		}
		
		i++;
		//if(i==2) break;
		a[0]=str_a[i].x;
		a[1]=str_a[i].y;
		b[0]=str_b[i].x;	
		b[1]=str_b[i].y;
		
	}
	return result;*/
    char4 *str_a=(char4*)d_a;
	char4 *str_b=(char4*)d_b;
	int i=0;
	int result=0;
	char4 cura, curb;
	cura=str_a[i];
	curb=str_b[i];
	while(result==0)
	{	
		//loop 0
		if((cura.x=='\0') || (curb.x=='\0'))
		{
			//if(a[0]=='\0' && b[0]=='\0')
			if(cura.x==curb.x)
				result=0;
			else
				if(cura.x=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(cura.x>curb.x)
		{
			result=1;
			break;
		}
		else if(cura.x<curb.x)
		{
			result=-1;
			break;
		}
		//loop 1
		if((cura.y=='\0') || (curb.y=='\0'))
		{
			//if(a[0]=='\0' && b[0]=='\0')
			if(cura.y==curb.y)
				result=0;
			else
				if(cura.y=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(cura.y>curb.y)
		{
			result=1;
			break;
		}
		else if(cura.y<curb.y)
		{
			result=-1;
			break;
		}
		//loop 2
		if((cura.z=='\0') || (curb.z=='\0'))
		{
			//if(a[0]=='\0' && b[0]=='\0')
			if(cura.z==curb.z)
				result=0;
			else
				if(cura.z=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(cura.z>curb.z)
		{
			result=1;
			break;
		}
		else if(cura.z<curb.z)
		{
			result=-1;
			break;
		}
		//loop 3
		if((cura.w=='\0') || (curb.w=='\0'))
		{
			//if(a[0]=='\0' && b[0]=='\0')
			if(cura.w==curb.w)
				result=0;
			else
				if(cura.w=='\0')
					result=-1;
				else
					result=1;
			break;
		}
		else
		if(cura.w>curb.w)
		{
			result=1;
			break;
		}
		else if(cura.w<curb.w)
		{
			result=-1;
			break;
		}		
		i++;
		cura=str_a[i];
		curb=str_b[i];			
	}
	return result;
}


void printString(void * rawData, cmp_type_t* d_R, int rLen)
{
//#ifdef DEBUG_SAVEN
	int i=0;
	char *data=(char*)rawData;
	char *str1=NULL;
	char *str2=NULL;
	int result=0;
	int errorTimes=0;
	for(i=1;i<rLen;i++)
	{
		str2=data+d_R[i].x;
		str1=data+d_R[i-1].x;
		result=compareStringCPU(str2,str1);
		if(result<0)
		{
			printf("error in sorting, %d, %s, %d, %s, %d, %s, %d, %s\n", i-2, data+d_R[i-2].x, i-1, str1, i, str2,i+1, data+d_R[i+1].x);
			errorTimes++;
			if(errorTimes>10)
			exit(0);
		}	
		//if(i<15)
		//	printf("%d,%s,", i,str1);
	}
	if(errorTimes==0)
		printf("pass the checking\n");
	else
		printf("fail the checking!!\n");
//#endif

	/*char *data=(char*)rawData;
	char *str1=NULL;
	int i=0;
	for(i=0;i<rLen;i++)
	{
		
		str1=data+d_R[i].x;
		printf("%d,%s,", i,str1);
	}*/
}


__device__ int getCompareValue(void *d_rawData, int v1, int v2)
{
	int compareValue=0;
	if((v1==-1) || (v2==-1))
	{
		if(v1==v2)
			compareValue=0;
		else
			if(v1==-1)
				compareValue=-1;
			else
				compareValue=1;
	}
	else
		compareValue=compareString((void*)(((char*)d_rawData)+v1),(void*)(((char*)d_rawData)+v2)); 
	/*{
		if(v1>v2)
			compareValue=1;
		else if(v1<v2)
			compareValue=-1;
	}*/
	return compareValue;
}
void * s_qsRawData=NULL;

int getSTLQsortValue(const void * p1, const void* p2)
{
	cmp_type_t v1=*((cmp_type_t*)p1);
	cmp_type_t v2=*((cmp_type_t*)p2);
	return compareStringCPU((void*)(((char*)s_qsRawData)+v1.x),(void*)(((char*)s_qsRawData)+v2.x));
}

#endif
