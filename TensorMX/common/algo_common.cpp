#include "algo_common.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>


void* CNN_GPU_MemMaloc(int _iGpuIndex, unsigned int _uiMemSize)
{
	cudaError_t eResult = cudaGetDevice(&_iGpuIndex);
	if (cudaSuccess != eResult)
	{
		printf("CNN_GPU_MemMaloc %d. Cublas won't be available.\n", _iGpuIndex);
		return NULL;
	}

	//检查当前显存使用量
	size_t sFreeMem = 0;
	size_t sTotalMem = 0;
	if (cudaSuccess != cudaMemGetInfo(&sFreeMem, &sTotalMem))
	{
		printf("CNN_GPU_MemMaloc. Cublas won't be available.\n");
		return NULL;
	}

	size_t ui64MinMemTh = (size_t)(sTotalMem * 0.05);
	if (sFreeMem < ui64MinMemTh + _uiMemSize)
	{
		return NULL;
	}

	void* pMemBuffer = NULL;
	if (cudaSuccess != cudaMalloc(&pMemBuffer, _uiMemSize))
	{
		return NULL;
	}

	return pMemBuffer;
}

int CNN_GPU_Memcpy(const size_t N, const void* X, void* Y)
{
	if (X != Y)
	{
		if (cudaSuccess != cudaMemcpy(Y, X, N, cudaMemcpyDefault))
		{
			return -1;
		}
		else
		{
			return 0;  // NOLINT(caffe/alt_fn)
		}		
	}
	else
	{
		return 0;  // NOLINT(caffe/alt_fn)
	}
}
int CNN_GPU_MemFree(void* _pMemBuffer)
{
	if (NULL == _pMemBuffer)
	{
		return 0;
	}
	if (cudaSuccess != cudaFree(_pMemBuffer))
	{
		return -1;
	}

	return 0;
}
int Get_Type_Szie(DataType	_type)
{
	int iTypeSize = 0;
	if (_type == DataType::kFLOAT)
	{
		iTypeSize = sizeof(float);
	}
	else if (DataType::kINT32 == _type)
	{
		iTypeSize = sizeof(int);
	}
	else if (DataType::kHALF == _type)
	{
		iTypeSize = sizeof(short);
	}
	else if (DataType::kINT8 == _type)
	{
		iTypeSize = sizeof(char);
	}
	else {
		printf("CNN_ConvLayer  biasWeights Type is not support! \n!");
	}
	return iTypeSize;
}
size_t get_bolck_size(Dims _stInPut)
{
	size_t iCount = 1;
	for (int i = 0; i < _stInPut.nbDims; i++)
	{
		iCount *= _stInPut.d[i];
	}
	return iCount;
}

