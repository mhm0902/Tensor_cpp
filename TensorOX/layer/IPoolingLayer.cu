#include "IPoolingLayer.h"
#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cfloat>
#include <vector>
#include <algorithm>


template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads, const Dtype* const bottom_data,
	const int num, const int channels, const int height, const int width,
	const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w,
	const int stride_h, const int stride_w, const int pad_h, const int pad_w,
	Dtype* const top_data, int* mask, Dtype* top_mask)
{
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		const int pw = index % pooled_width;
		const int ph = (index / pooled_width) % pooled_height;
		const int c = (index / pooled_width / pooled_height) % channels;
		const int n = index / pooled_width / pooled_height / channels;
		int hstart = ph * stride_h - pad_h;
		int wstart = pw * stride_w - pad_w;
		const int hend = min(hstart + kernel_h, height);
		const int wend = min(wstart + kernel_w, width);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		Dtype maxval = -FLT_MAX;
		int maxidx = -1;
		const Dtype* const bottom_slice = bottom_data + (n * channels + c) * height * width;
		for (int h = hstart; h < hend; ++h)
		{
			for (int w = wstart; w < wend; ++w)
			{
				if (bottom_slice[h * width + w] > maxval)
				{
					maxidx = h * width + w;
					maxval = bottom_slice[maxidx];
				}
			}
		}
		top_data[index] = maxval;
		//if (mask)
		//{
		//	mask[index] = maxidx;
		//}
		//else
		//{
		//	top_mask[index] = maxidx;
		//}
	}
}


template <typename Dtype>
__global__ void AvePoolForward(const int nthreads, const Dtype* const bottom_data,
	const int num, const int channels, const int height, const int width,
	const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w,
	const int stride_h, const int stride_w, const int pad_h, const int pad_w, Dtype* const top_data)
{
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		const int pw = index % pooled_width;
		const int ph = (index / pooled_width) % pooled_height;
		const int c = (index / pooled_width / pooled_height) % channels;
		const int n = index / pooled_width / pooled_height / channels;
		int hstart = ph * stride_h - pad_h;
		int wstart = pw * stride_w - pad_w;
		int hend = min(hstart + kernel_h, height + pad_h);
		int wend = min(wstart + kernel_w, width + pad_w);
		const int pool_size = (hend - hstart) * (wend - wstart);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		hend = min(hend, height);
		wend = min(wend, width);
		Dtype aveval = 0;
		const Dtype* const bottom_slice = bottom_data + (n * channels + c) * height * width;
		for (int h = hstart; h < hend; ++h)
		{
			for (int w = wstart; w < wend; ++w)
			{
				aveval += bottom_slice[h * width + w];
			}
		}
		top_data[index] = aveval / pool_size;
	}
}


int IPoolingLayer::forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims _stOutPut)
{
	int top_count = _stInPut.d[0] * _stInPut.d[1] * _stOutPut.d[2] * _stOutPut.d[3];

	_stOutPut.nbDims = _stInPut.nbDims;
	_stOutPut.d[0] = _stInPut.d[0];
	_stOutPut.d[1] = _stInPut.d[1];
	_stOutPut.d[2] = _stInPut.d[2] / m_stStride.d[0];
	_stOutPut.d[3] = _stInPut.d[3] / m_stStride.d[1];

	switch (m_eMode)
	{
	case POOL_MAX:
	{
		int *mask = NULL;
		float* top_mask = NULL;

		MaxPoolForward<float> <<<CNN_GET_BLOCKS(top_count), CNN_CUDA_NUM_THREADS >>> (
			top_count, (float*)_pInData, _stInPut.d[0], _stInPut.d[1], _stInPut.d[2], _stInPut.d[03],
		_stOutPut.d[2], _stOutPut.d[3],
		m_stKernel.d[0], m_stKernel.d[1],
		m_stStride.d[0], m_stStride.d[1],
		m_stPadding.d[0], m_stPadding.d[1],
		(float*)_pOutData,
		mask, top_mask);
	}
	break;
	case POOL_AVE:
	{
		AvePoolForward<float> <<<CNN_GET_BLOCKS(top_count), CNN_CUDA_NUM_THREADS >>> (
			top_count, (float*)_pInData, _stInPut.d[0], _stInPut.d[1], _stInPut.d[2], _stInPut.d[03],
			_stOutPut.d[2], _stOutPut.d[3],
			m_stKernel.d[0], m_stKernel.d[1],
			m_stStride.d[0], m_stStride.d[1],
			m_stPadding.d[0], m_stPadding.d[1],
			(float*)_pOutData );
	}
	break;
	case POOL_MAX_AVERAGE_BLEND:	//ÔÝ²»Ö§³Ö
	default:
		break;	//nothing
	}
	cudaError_t status = cudaDeviceSynchronize();

	if (cudaSuccess != status)
	{
		printf("IPoolingLayer::forward error:%d\n", status);
		return -1;
	}
	return 0;
}