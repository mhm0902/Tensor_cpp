#include "IConcatenationLayer.h"

//template <typename Dtype>
//__global__ void CNN_Concat_Kernel(const int nthreads, const Dtype* in_data,
//	const int num_concats, const int concat_size,
//	const int top_concat_axis, const int bottom_concat_axis,
//	const int offset_concat_axis, Dtype* out_data)
//{
//	//CUDA_KERNEL_LOOP(index, nthreads)
//	//{
//	//	const int total_concat_size = concat_size * bottom_concat_axis;
//	//	const int concat_num = index / total_concat_size;
//	//	const int concat_index = index % total_concat_size;
//	//	const int top_index = concat_index + (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
//	//	out_data[top_index] = in_data[index];
//	//}
//}

IConcatenationLayer::IConcatenationLayer()
{
}

IConcatenationLayer::~IConcatenationLayer()
{

}

int IConcatenationLayer::forward(void* _pInA, Dims _stInA, void* _pInB, Dims _stInB, void* _pOutY, Dims &_stOutY)
{
	if (NULL == _pInA || NULL == _pInB || NULL == _pOutY)
	{
		return -1;
	}
	if (_stInA.d[0] != _stInB.d[0] || _stInA.d[2] != _stInB.d[2] || _stInA.d[3] != _stInB.d[3]) //n h w
	{
		return -2;
	}
	int iBatch = _stInA.d[0];
	_stOutY.nbDims = _stInA.nbDims;
	_stOutY.d[0] = iBatch;						//n
	_stOutY.d[1] = _stInA.d[1] + _stInB.d[1];	//c
	_stOutY.d[2] = _stInA.d[2];					//h
	_stOutY.d[3] = _stInA.d[3];					//w

	int iSizeA = _stInA.d[1] * _stInA.d[2] * _stInA.d[3];
	int iSzieB = _stInB.d[1] * _stInB.d[2] * _stInB.d[3];

	float* pTensorA = (float*)_pInA;
	float* pTensorB = (float*)_pInB;
	float* pTensorY = (float*)_pOutY;

	for (int i = 0; i < iBatch; i++)
	{
		CNN_GPU_Memcpy( iSizeA, pTensorA + i*iSizeA, pTensorY + i*(iSizeA + iSzieB) );
		CNN_GPU_Memcpy( iSzieB, pTensorB + i*iSzieB, pTensorY + i*(iSizeA + iSzieB) + iSizeA);
	}

	return 0;
}