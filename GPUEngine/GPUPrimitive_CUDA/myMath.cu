#pragma once

#include <cutil.h>
#include <cuda.h>

__device__ __host__
unsigned int uintFloorLog(unsigned int base, unsigned int num)
{
	//unsigned int result = 0;

	//for(unsigned int temp = 0; temp <= num; temp *= base)
	//	result++;

	//return result;

	return int(logf(float(base))/logf(float(num)));
}


__device__ __host__
unsigned int uintCeilingLog(unsigned int base, unsigned int num)
{
	unsigned int result = 0;

	for(unsigned int temp = 1; temp < num; temp *= base)
		result++;

	return result;
}

__device__ __host__
unsigned int uintPower(unsigned int base, unsigned int pow)
{
	unsigned int result = 1;

	for(; pow; pow--)
		result *= base;

	return result;
}

__device__ __host__
unsigned int uintCeilingDiv(unsigned int dividend, unsigned int divisor)
{
	return (dividend + divisor - 1) / divisor;
}


#define divRoundDown(n,s)  ((n) / (s))
#define divRoundUp(n,s)    (((n) / (s)) + ((((n) % (s)) > 0) ? 1 : 0))

