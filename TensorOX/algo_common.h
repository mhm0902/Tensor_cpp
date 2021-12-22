#ifndef __ALGO_COMMON_H__
#define __ALGO_COMMON_H__

#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdint.h>
#include<map>
#include<iostream>
#include<assert.h>
#include <fstream>
#include <sstream>
#include <vector>

//#ifdef __cplusplus
//extern "C"
//{
//#endif

static constexpr int INPUT_H = 640;  // yolov5's input height and width must be divisible by 32.
static constexpr int INPUT_W = 640;
static int CLASS_NUM = 80;

//!
//! \enum DataType
//!
//! \brief The type of weights and tensors.
//!
#define CUDNN_VERSION_MIN(major, minor, patch) (CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))

template <typename Dtype>
inline cudnnStatus_t setTensor4dDesc(cudnnTensorDescriptor_t* desc, int n, int c, int h, int w,
	int stride_n, int stride_c, int stride_h, int stride_w)
{
	return cudnnSetTensor4dDescriptorEx(*desc, CUDNN_DATA_FLOAT, n, c, h, w, stride_n, stride_c, stride_h, stride_w);
}

template <typename Dtype>
inline cudnnStatus_t setTensor4dDesc(cudnnTensorDescriptor_t* desc, int n, int c, int h, int w)
{
	const int stride_w = 1;
	const int stride_h = w * stride_w;
	const int stride_c = h * stride_h;
	const int stride_n = c * stride_c;
	return setTensor4dDesc<Dtype>(desc, n, c, h, w, stride_n, stride_c, stride_h, stride_w);
}

const int CNN_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CNN_GET_BLOCKS(const int N) {
	return (N + CNN_CUDA_NUM_THREADS - 1) / CNN_CUDA_NUM_THREADS;
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

#define MAX_DIMS 8  //!< The maximum number of dimensions supported for a tensor.
enum class DataType
{		
	kFLOAT	= 0,//! 32-bit floating point format.		
	kHALF	= 1,//! IEEE 16-bit floating-point format.		
	kINT8	= 2,//! 8-bit integer representing a quantized floating-point value.		
	kINT32	= 3,//! Signed 32-bit integer format.		
	kBOOL	= 4//! 8-bit boolean. 0 = false, 1 = true, other values undefined.
};

class Weights
{
	public:
		DataType	type;		//!< The type of the weights.
		const void*	values;		//!< The weight values, in a contiguous array.
		int			count;      //!< The number of weights in the array.
};

class Dims
{
	public:
		int nbDims;                              //!< The number of dimensions.
		int d[MAX_DIMS];                         //!< The extent of each dimension.
};

//!
//! \class DimsHW
//! \brief Descriptor for two-dimensional spatial data.
//!
class DimsHW : public Dims
{
public:
	//!
	//! \brief Construct an empty Dims2 object.
	//!
	DimsHW()
	{
		nbDims = 2;
		d[0] = d[1] = 0;
	}

	//!
	//! \brief Construct a Dims2 from 2 elements.
	//!
	//! \param d0 The first element.
	//! \param d1 The second element.
	//!
	DimsHW(int d0, int d1)
	{
		nbDims = 2;
		d[0] = d0;
		d[1] = d1;
	}
};



//!
//! \class Dims3
//! \brief Descriptor for three-dimensional data.
//!
class DimsCHW : public Dims
{
public:
	//!
	//! \brief Construct an empty Dims3 object.
	//!
	DimsCHW()
	{
		nbDims = 3;
		d[0] = d[1] = d[2] = 0;
	}

	//!
	//! \brief Construct a Dims3 from 3 elements.
	//!
	//! \param d0 The first element.
	//! \param d1 The second element.
	//! \param d2 The third element.
	//!
	DimsCHW(int d0, int d1, int d2)
	{
		nbDims = 3;
		d[0] = d0;
		d[1] = d1;
		d[2] = d2;
	}
	//!
	//! \brief Get the channel count.
	//!
	//! \return The channel count.
	//!
	int c() const
	{
		return d[0];
	}
	//!
	//! \brief Get the height.
	//!
	//! \return The height.
	//!
	int h() const
	{
		return d[1];
	}
	//!
	//! \brief Get the width.
	//!
	//! \return The width.
	//!
	int w() const
	{
		return d[2];
	}
};

//!
//! \class Dims4
//! \brief Descriptor for four-dimensional data.
//!
class DimsNCHW : public Dims
{
public:
	//!
	//! \brief Construct an empty Dims2 object.
	//!
	DimsNCHW()
	{
		nbDims = 4;
		d[0] = d[1] = d[2] = d[3] = 0;
	}

	//!
	//! \brief Construct a Dims4 from 4 elements.
	//!
	//! \param d0 The first element.
	//! \param d1 The second element.
	//! \param d2 The third element.
	//! \param d3 The fourth element.
	//!
	DimsNCHW(int d0, int d1, int d2, int d3)
	{
		nbDims = 4;
		d[0] = d0;
		d[1] = d1;
		d[2] = d2;
		d[3] = d3;
	}

	//!
	//! \brief Get the index count.
	//!
	//! \return The index count.
	//!
	int n() const
	{
		return d[0];
	}

	//!
	//! \brief Get the channel count.
	//!
	//! \return The channel count.
	//!
	int c() const
	{
		return d[1];
	}

	//!
	//! \brief Get the height.
	//!
	//! \return The height.
	//!
	int h() const
	{
		return d[2];
	}

	//!
	//! \brief Get the width.
	//!
	//! \return The width.
	//!
	int w() const
	{
		return d[3];
	}
};

enum class ScaleMode : int32_t
{
	kUNIFORM = 0,    //!< Identical coefficients across all elements of the tensor.
	kCHANNEL = 1,    //!< Per-channel coefficients.
	kELEMENTWISE = 2 //!< Elementwise coefficients.
};

void* CNN_GPU_MemMaloc(int _iGpuIndex, unsigned int _uiMemSize);
int CNN_GPU_Memcpy(const size_t N, const void* X, void* Y);
int CNN_GPU_MemFree(void* _pMemBuffer);
int Get_Type_Szie(DataType	_type);
int get_bolck_size(Dims _stInPut);
//#ifdef __cplusplus
//}
//#endif

#endif//__ALGO_COMMON_H__
