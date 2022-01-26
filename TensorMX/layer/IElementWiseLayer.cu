#include "IElementWiseLayer.h"

template <typename Dtype> __global__ void ElementLayer_Prod(const int n, const Dtype* inA, Dtype*inB, Dtype* out)
{
	//for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
	//{
	//	out[i] = inA[i] * inB[i];
	//}
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n)
		return;

	out[i] = inA[i] * inB[i];
}
template <typename Dtype> __global__ void ElementLayer_Sum(const int n, const Dtype* inA, Dtype*inB, Dtype* out)
{
	//for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
	//{
	//	out[i] = inA[i] + inB[i];
	//}
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	out[i] = inA[i] + inB[i];
}
template <typename Dtype> __global__ void ElementLayer_Max(const int n, const Dtype* inA, Dtype*inB, Dtype* out)
{
	//for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
	//{
	//	Dtype x = inA[i];
	//	Dtype y = inB[i];
	//	if (x > y)
	//	{
	//		out[i] = x;
	//	}
	//	else
	//	{
	//		out[i] = y;
	//	}		
	//}
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n)
		return;

	Dtype x = inA[i];
	Dtype y = inB[i];
	if (x > y)
	{
		out[i] = x;
	}
	else
	{
		out[i] = y;
	}
}

int IElementWiseLayer::forward(void* _pInA, void* _pInB, Dims _stInPut, ELTWISE_OPT_TYPE_E _eOptTyoe, void* _pOutY)
{
	int iCount = 1;//
	for (int i = 0; i < _stInPut.nbDims; i++) {
		iCount *= _stInPut.d[i];
	}
	switch (_eOptTyoe)
	{
	case CNN_ELTWISE_PROD:
	{
		ElementLayer_Prod<float> << <CNN_GET_BLOCKS(iCount), CNN_CUDA_NUM_THREADS >> >(iCount, (float*)_pInA, (float*)_pInB, (float*)_pOutY);
	}
	break;
	case CNN_ELTWISE_SUM:
	{
		ElementLayer_Sum<float> << <CNN_GET_BLOCKS(iCount), CNN_CUDA_NUM_THREADS >> >(iCount, (float*)_pInA, (float*)_pInB, (float*)_pOutY);
	}
	break;
	case CNN_ELTWISE_MAX:
	{
		ElementLayer_Max<float> << <CNN_GET_BLOCKS(iCount), CNN_CUDA_NUM_THREADS >> >(iCount, (float*)_pInA, (float*)_pInB, (float*)_pOutY);
	}
	break;
	default:
		//LOG(FATAL) << "Unknown elementwise operation.";
		break;	//nothing
	}
	cudaDeviceSynchronize();
	return 0;
}