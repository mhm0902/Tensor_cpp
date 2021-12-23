#include "IUpsampleLayer.h"
#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
//图像缩放（相同数据类型）线性插值
template <typename Dtype>
__global__ void YL_ResizeLinear_kernel(const Dtype* _pSrcData, Dtype* _pDstData,
	const int _iSrcImgW, const int _iSrcImgH, const int _iDstImgW, const int _iDstImgH, int _iChnNum,
	const int _iSrcOffX, const int _iSrcOffY, const float _f32ScaleW, const float _f32ScaleH)
{
	const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

	if (dst_x < _iDstImgW && dst_y < _iDstImgH)
	{
		//const float src_x = dst_x * _f32ScaleW + _iSrcOffX;
		//const float src_y = dst_y * _f32ScaleH + _iSrcOffY;
		//same to cpu version
		const float src_x = (dst_x + 0.5f) * _f32ScaleW - 0.5f + _iSrcOffX;
		const float src_y = (dst_y + 0.5f) * _f32ScaleH - 0.5f + _iSrcOffY;

		const int x1 = __float2int_rd(src_x);
		const int y1 = __float2int_rd(src_y);
		const int x2 = x1 + 1;
		const int y2 = y1 + 1;
		const int x2_read = ::min(x2, _iSrcImgW - 1);
		const int y2_read = ::min(y2, _iSrcImgH - 1);
		_pDstData[dst_y * _iDstImgW + dst_x] = (Dtype)(_pSrcData[y1 * _iSrcImgW + x1] * (x2 - src_x) * (y2 - src_y) +
			_pSrcData[y1 * _iSrcImgW + x2_read] * (src_x - x1) * (y2 - src_y) +
			_pSrcData[y2_read * _iSrcImgW + x1] * (x2 - src_x) * (src_y - y1) +
			_pSrcData[y2_read * _iSrcImgW + x2_read] * (src_x - x1) * (src_y - y1));
	}
}

//图像缩放（相同数据类型）近邻插值
template <typename Dtype>
__global__ void YL_ResizeNearest_kernel(const Dtype* _pSrcData, Dtype* _pDstData,
	const int _iSrcImgW, const int _iSrcImgH, const int _iDstImgW, const int _iDstImgH, int _iChnNum,
	const int _iSrcOffX, const int _iSrcOffY, const float _f32ScaleW, const float _f32ScaleH)
{	
	int n = blockIdx.y / _iChnNum;
	int c= blockIdx.y%_iChnNum;
	int ndetla= _iChnNum*_iDstImgH*_iDstImgW;
	int sdetla= _iChnNum*_iSrcImgH*_iSrcImgW;
	int hcnt = blockIdx.x / _iSrcOffX;
	int wcnt = blockIdx.x%_iSrcOffX;
	int dst_x = blockDim.x * wcnt + threadIdx.x;
	int dst_y = blockDim.y * hcnt + threadIdx.y;

	if (dst_x < _iDstImgW && dst_y < _iDstImgH)
	{
		const int src_x = (int)(dst_x * _f32ScaleW);
		const int src_y = (int)(dst_y * _f32ScaleH);
		//if (c==1&& dst_x == 960 && dst_y == 480)
		//{
		//	printf("test");
		//}
		if (src_x<_iSrcImgW && src_y<_iSrcImgH)
			_pDstData[n*ndetla+c*_iDstImgH*_iDstImgW +dst_y * _iDstImgW + dst_x] = (Dtype)
					(_pSrcData[n*sdetla+c*_iSrcImgH*_iSrcImgW +src_y * _iSrcImgW + src_x] );
	}
}

int IUpsampleLayer::forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims _stOutPut)
{
	int n = _stInPut.d[0];
	int cnt = 1;
	for (int i = 0;i < _stOutPut.nbDims;i++)
	{
		cnt *= _stOutPut.d[i];
	}
	dim3 grid, block = {8,8};
	float wscale = float(_stInPut.d[3]) /float(_stOutPut.d[3]);
	float hscale = float(_stInPut.d[2]) / float(_stOutPut.d[2]);
	int blockmax = block.x*block.y*block.z;
	int ycnt = (_stOutPut.d[2] + block.y - 1) / block.y;
	int xcnt =(_stOutPut.d[3] + block.x - 1) / block.x;
	grid.x = ycnt*xcnt;
	grid.y = _stOutPut.d[1]* _stOutPut.d[0];
	YL_ResizeNearest_kernel << <grid, block >> > ((float*)_pInData, (float*)_pOutData, _stInPut.d[3], _stInPut.d[2], _stOutPut.d[3], _stOutPut.d[2]
		, _stOutPut.d[1], xcnt, ycnt, wscale, hscale);
	cudaError_t status= cudaDeviceSynchronize();
	return 0;
}