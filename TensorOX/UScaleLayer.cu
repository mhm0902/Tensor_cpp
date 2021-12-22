#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "IScaleLayer.h"



template <typename Dtype> __global__ void ScaleForward(const int n, const Dtype* in,const Dtype* scale, 
	const int scale_dim, const int inner_dim, Dtype* out)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
	{
		const int scale_index = (i / inner_dim) % scale_dim;
		out[i] = in[i] * scale[scale_index];
	}
}

template <typename Dtype> __global__ void ScaleBiasForward(const int n, const Dtype* in, const Dtype* scale,
	const Dtype* bias, const int scale_dim, const int inner_dim, Dtype* out)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
	{
		const int scale_index = (i / inner_dim) % scale_dim;
		out[i] = in[i] * scale[scale_index] + bias[scale_index];
	}
}

template <typename Dtype> int IScaleLayer_forward_GPU(
	const Dtype* _pfIn, const Dtype* _pfScale, const Dtype* _pfBias, const int _iScaleDim, 
	const int _iInnerDim, int _iTheads, Dtype* _pdOut)
{
	if (_pfBias != NULL  )
	{
		ScaleBiasForward<float> << <CNN_GET_BLOCKS(_iTheads), CNN_CUDA_NUM_THREADS >> >(
			_iTheads, _pfIn, _pfScale, _pfBias,_iScaleDim, _iInnerDim, _pdOut);
	}
	else
	{
		ScaleForward<float> << <CNN_GET_BLOCKS(_iTheads), CNN_CUDA_NUM_THREADS >> >(
			_iTheads, _pfIn, _pfBias, _iScaleDim, _iInnerDim, _pdOut);
	}

	return 0;
}

