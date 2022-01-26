#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include"IActivationLayer.h"
// CUDA: use 512 threads per block

template <typename Dtype> __global__ void SiLUForward(const int n, const Dtype* in, Dtype* out)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n)
		return;
	double x = in[i];

	if (x <= -20.)
	{
		out[i] = FLT_EPSILON;
	}
	else if (x >= 20.)
	{
		out[i] = x;
	}
	else {
		out[i] = x / (1. + exp(-x));
	}
}

template <typename Dtype> __global__ void TanhForward(const int n, Dtype* in, Dtype* out) 
{
	//for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
	//{
	//	out[i] = tanh(in[i]);
	//}
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	out[i] = tanh(in[i]);
}

template <typename Dtype> __global__ void SigmoidForward(const int n, Dtype* in, Dtype* out) 
{
	//for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x) 
	//{
	//	out[index] = 1. / (1. + exp(-in[index]));
	//}
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	out[i] = 1. / (1. + exp(-in[i]));
}


template <typename Dtype> __global__ void ReLUForward(const int n, Dtype* in, Dtype* out, Dtype negative_slope) 
{
	//for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x) 
	//{
	//	Dtype x = in[index];
	//	out[index] = x > 0 ? x : x * negative_slope;
	//}
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;

	Dtype x = in[i];
	out[i] = x > 0 ? x : x * negative_slope;
}


int IActivationLayer::forward(void* _pInData, Dims _stInPut, void* _pOutData, ActivateMode _eMode)
{
	int count = 1;//
	for (int i = 0; i < _stInPut.nbDims; i++) 
	{
		count *= _stInPut.d[i];
	}

	switch (_eMode)
	{
		case ReLU:
		{
			ReLUForward << <CNN_GET_BLOCKS(count), CNN_CUDA_NUM_THREADS >> >( count, (float*)_pInData, (float*)_pOutData, 0.f);
		}
		break;
		case Sigmoid:
		{
			SigmoidForward << <CNN_GET_BLOCKS(count), CNN_CUDA_NUM_THREADS >> >(count, (float*)_pInData, (float*)_pOutData);
		}
		break;
		case Tanh:
		{
			TanhForward << <CNN_GET_BLOCKS(count), CNN_CUDA_NUM_THREADS >> >(count, (float*)_pInData, (float*)_pOutData);
		}
		break;
		case Silu:
		{
			SiLUForward<float> << <CNN_GET_BLOCKS(count), CNN_CUDA_NUM_THREADS >> >(count, (float*)_pInData, (float*)_pOutData);
		}
		break;
	}
	cudaDeviceSynchronize();
	return 0;
}


