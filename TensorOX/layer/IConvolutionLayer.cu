#include "IConvolutionLayer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>
#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <algorithm>

template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w, const int height_col, const int width_col,
	Dtype* data_col)
{
	CUDA_KERNEL_LOOP(index, n)
	{
		const int h_index = index / width_col;
		const int h_col = h_index % height_col;
		const int w_col = index % width_col;
		const int c_im = h_index / height_col;
		const int c_col = c_im * kernel_h * kernel_w;
		const int h_offset = h_col * stride_h - pad_h;
		const int w_offset = w_col * stride_w - pad_w;
		Dtype* data_col_ptr = data_col;
		data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
		const Dtype* data_im_ptr = data_im;
		data_im_ptr += (c_im * height + h_offset) * width + w_offset;
		for (int i = 0; i < kernel_h; ++i)
		{
			for (int j = 0; j < kernel_w; ++j)
			{
				int h_im = h_offset + i * dilation_h;
				int w_im = w_offset + j * dilation_w;
				*data_col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
					data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
				data_col_ptr += height_col * width_col;
			}
		}
	}
}

template <typename Dtype>
void CNN_Im2Col_GPU(const Dtype* data_im, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w, Dtype* data_col)
{
	// We are going to launch channels * height_col * width_col kernels, each
	// kernel responsible for copying a single-channel grid.
	int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	int num_kernels = channels * height_col * width_col;
	// NOLINT_NEXT_LINE(whitespace/operators)
	im2col_gpu_kernel<Dtype> << <CNN_GET_BLOCKS(num_kernels),
		CNN_CUDA_NUM_THREADS >> >(num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
			pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col, width_col, data_col);

#ifdef USE_GLOG
	CUDA_POST_KERNEL_CHECK;
#else
	if (cudaSuccess != cudaPeekAtLastError())
	{
		printf("im2col_gpu_kernel error:%d\n");
		return;
	}
#endif
}

template
void CNN_Im2Col_GPU<float>(const float* data_im, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w, float* data_col);
template
void CNN_Im2Col_GPU<double>(const double* data_im, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w, double* data_col);

namespace fecnn {
	// 需要与处理的数据类型保持一致
#define sizeofStorageT 4
#define sizeofComputeT 4
#define CPUStorage2ComputeT(x) (x)
#define CPUCompute2StorageT(x) (x)
#define GPUStorage2ComputeT(x) (x)
#define GPUCompute2StorageT(x) (x)
#define GPUgemm cublasSgemm
#define GPUasum cublasSasum
#define ISNAN(x) (std::isnan(x)) //测试某个浮点数是否是 非数字
#define ComputeT_MIN FLT_MIN //1.17549e-38

	void FatalError(const int lineNumber = 0) {
		std::cerr << "FatalError";
		// if (fileName!="") std::cerr<<" at FILE "<<fileName;
		if (lineNumber != 0) std::cerr << " at LINE " << lineNumber;
		std::cerr << ". Program Terminated." << std::endl;
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}

	void checkCUDA(const char* fileName, const int lineNumber, cudaError_t status) {
		if (status != cudaSuccess) {
			std::cerr << "CUDA failure at FILE " << fileName << " at LINE " << lineNumber << ": " << status << std::endl;
			FatalError();
		}
	}

	void checkCUBLAS(const char* fileName, const int lineNumber, cublasStatus_t status) {
		if (status != CUBLAS_STATUS_SUCCESS) {
			std::cerr << "CUBLAS failure at FILE " << fileName << " at LINE " << lineNumber << ": ";
			switch (status) {
			case CUBLAS_STATUS_SUCCESS:             std::cerr << "CUBLAS_STATUS_SUCCESS" << std::endl; break;
			case CUBLAS_STATUS_NOT_INITIALIZED:     std::cerr << "CUBLAS_STATUS_NOT_INITIALIZED" << std::endl; break;
			case CUBLAS_STATUS_ALLOC_FAILED:        std::cerr << "CUBLAS_STATUS_ALLOC_FAILED" << std::endl; break;
			case CUBLAS_STATUS_INVALID_VALUE:       std::cerr << "CUBLAS_STATUS_INVALID_VALUE" << std::endl; break;
			case CUBLAS_STATUS_ARCH_MISMATCH:       std::cerr << "CUBLAS_STATUS_ARCH_MISMATCH" << std::endl; break;
			case CUBLAS_STATUS_MAPPING_ERROR:       std::cerr << "CUBLAS_STATUS_MAPPING_ERROR" << std::endl; break;
			case CUBLAS_STATUS_EXECUTION_FAILED:    std::cerr << "CUBLAS_STATUS_EXECUTION_FAILED" << std::endl; break;
			case CUBLAS_STATUS_INTERNAL_ERROR:      std::cerr << "CUBLAS_STATUS_INTERNAL_ERROR" << std::endl; break;
			case CUBLAS_STATUS_NOT_SUPPORTED:       std::cerr << "CUBLAS_STATUS_NOT_SUPPORTED" << std::endl; break;
			case CUBLAS_STATUS_LICENSE_ERROR:       std::cerr << "CUBLAS_STATUS_LICENSE_ERROR" << std::endl; break;
			}
			FatalError();
		}
		checkCUDA(fileName, lineNumber, cudaGetLastError());
	}
	// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

	// CUDA: use 512 threads per block
	const int FECNN_CUDA_NUM_THREADS = 512;

	// CUDA: number of blocks for threads.
	inline int FECNN_GET_BLOCKS(const int N) {
		return (N + FECNN_CUDA_NUM_THREADS - 1) / FECNN_CUDA_NUM_THREADS;
	}
	void fecnn_gpu_gemm(cublasHandle_t cublasHandle, const cublasOperation_t TransA,
		const cublasOperation_t TransB, const int M, const int N, const int K,
		const StorageT alpha, const StorageT* A, const StorageT* B, const StorageT beta,
		StorageT* C) {
		// Note that cublas follows fortran order.
		int lda = (TransA == CUBLAS_OP_N) ? K : M;
		int ldb = (TransB == CUBLAS_OP_N) ? N : K;
		cublasOperation_t cuTransA =
			(TransA == CUBLAS_OP_N) ? CUBLAS_OP_N : CUBLAS_OP_T;
		cublasOperation_t cuTransB =
			(TransB == CUBLAS_OP_N) ? CUBLAS_OP_N : CUBLAS_OP_T;
		checkCUBLAS(__FILE__, __LINE__,
			cublasSgemm(cublasHandle, cuTransB, cuTransA,
				N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
	}

	// @fixme 只适合计算float型
	void fecnn_gpu_gemv(cublasHandle_t cublasHandle, const cublasOperation_t TransA, const int M,
		const int N, const StorageT alpha, const StorageT* A, const StorageT* x,
		const StorageT beta, StorageT* y) {
		cublasOperation_t cuTransA =
			(TransA == CUBLAS_OP_N) ? CUBLAS_OP_T : CUBLAS_OP_N;
		checkCUBLAS(__FILE__, __LINE__,
			cublasSgemv(cublasHandle, cuTransA, N, M, &alpha,
				A, N, x, 1, &beta, y, 1));
	}

	__global__ void im2col_gpu_kernel(const int n, const StorageT* data_im,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		const int height_col, const int width_col,
		StorageT* data_col) {
		CUDA_KERNEL_LOOP(index, n) {
			const int h_index = index / width_col;
			const int h_col = h_index % height_col;
			const int w_col = index % width_col;
			const int c_im = h_index / height_col;
			const int c_col = c_im * kernel_h * kernel_w;
			const int h_offset = h_col * stride_h - pad_h;
			const int w_offset = w_col * stride_w - pad_w;

			StorageT* data_col_ptr = data_col;
			data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;

			const StorageT* data_im_ptr = data_im;
			data_im_ptr += (c_im * height + h_offset) * width + w_offset;

			for (int i = 0; i < kernel_h; ++i) {
				for (int j = 0; j < kernel_w; ++j) {
					int h_im = h_offset + i * dilation_h;
					int w_im = w_offset + j * dilation_w;
					*data_col_ptr =
						(h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
						data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
					data_col_ptr += height_col * width_col;
				}
			}
		}
	}

	void im2col_gpu(const StorageT* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		StorageT* data_col) {

		// std::cout<<"channels="<<channels<<" height="<<height<<" width="<<width<<std::endl; // conv1:[1,28,28] conv2:[20,12,12]
		// std::cout<<"dilation_h="<<dilation_h<<" dilation_w="<<dilation_w<<std::endl; // =1,1

		// We are going to launch channels * height_col * width_col kernels, each
		// kernel responsible for copying a single-channel grid.
		int height_col = (height + 2 * pad_h -
			(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
		int width_col = (width + 2 * pad_w -
			(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
		int num_kernels = channels * height_col * width_col;

		// std::cout<<"height_col="<<height_col<<" width_col="<<width_col<<" channels="<<channels<<std::endl; // conv1:[24,24,1] ; conv2:[8,8,20]

		// NOLINT_NEXT_LINE(whitespace/operators)
		im2col_gpu_kernel << <FECNN_GET_BLOCKS(num_kernels),
			FECNN_CUDA_NUM_THREADS >> > (
				num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
				pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
				width_col, data_col);
	}

	__global__ void col2im_gpu_kernel(const int n, const StorageT* data_col,
		const int height, const int width, const int channels,
		const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		const int height_col, const int width_col,
		StorageT* data_im)
	{
		CUDA_KERNEL_LOOP(index, n)
		{
			StorageT val = 0;
			const int w_im = index % width + pad_w;
			const int h_im = (index / width) % height + pad_h;
			const int c_im = index / (width * height);
			int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
			int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
			// compute the start and end of the output
			const int w_col_start = (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
			int tmp = (int)(w_im / stride_w + 1);
			const int w_col_end = tmp < width_col ? tmp : width_col;
			const int h_col_start = (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
			tmp = (int)(h_im / stride_h + 1);
			const int h_col_end = tmp < height_col ? tmp : height_col;
			// TODO: use LCM of stride and dilation to avoid unnecessary loops
			for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
				for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
					int h_k = (h_im - h_col * stride_h);
					int w_k = (w_im - w_col * stride_w);
					if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
						h_k /= dilation_h;
						w_k /= dilation_w;
						int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
							height_col + h_col) * width_col + w_col;
						val += data_col[data_col_index];
					}
				}
			}
			data_im[index] = val;
		}
	}

	void col2im_gpu(const StorageT* data_col, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h,
		const int stride_w, const int dilation_h, const int dilation_w,
		StorageT* data_im) {
		int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
			stride_h + 1;
		int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
			stride_w + 1;
		int num_kernels = channels * height * width;
		// To avoid involving atomic operations, we will launch one kernel per
		// bottom dimension, and then in the kernel add up the top dimensions.
		// NOLINT_NEXT_LINE(whitespace/operators)
		col2im_gpu_kernel << <FECNN_GET_BLOCKS(num_kernels),
			FECNN_CUDA_NUM_THREADS >> > (
				num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
				pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
				height_col, width_col, data_im);
	}

}

IConvolutionLayer::IConvolutionLayer()
{
	m_pstBias = NULL;
	m_pstWeights = NULL;
	m_bHasBias = false;
	//卷积参数
	m_iGroup = 1;
	m_stDilation = DimsHW(1, 1);	//核宽度方向填充像素数
	m_stStride = DimsHW(1, 1);	//宽度方向步长
	m_stPadding = DimsHW(0, 0);	//宽度方向填充像素数

								//m_iChannelAxis	= 0;			//通道数所在维度（后续扩展用）
}
IConvolutionLayer::IConvolutionLayer(int _nbOutputMaps, Dims _kernelSize, Weights _kernelWeights, Weights _biasWeights)
{
	m_pstBias = NULL;
	m_pstWeights = NULL;
	m_bHasBias = false;

	if (_biasWeights.count > 0 && _biasWeights.values != NULL)
	{
		m_bHasBias = true;
		m_pstBias = (Weights*)malloc(sizeof(Weights));		//Bias数据

		if (NULL == m_pstBias)
		{
			printf("CNN_ConvLayer m_pstBias malloc error\n!");
			return;
		}
		m_pstBias->type = _biasWeights.type;
		m_pstBias->count = _biasWeights.count;

		int iTypeSize = Get_Type_Szie(_biasWeights.type);

		if (0 == iTypeSize)
		{
			return;
		}

		m_pstBias->values = CNN_GPU_MemMaloc(0, _biasWeights.count * iTypeSize);

		if (NULL == m_pstBias->values)
		{
			printf("CNN_ConvLayer m_pstBias->values malloc error\n!");
			return;
		}

		int iStatus = CNN_GPU_Memcpy(_biasWeights.count * iTypeSize, _biasWeights.values, (void*)(m_pstBias->values));

		if (iStatus != 0)
		{
			printf("CNN_ConvLayer m_pstBias->values CNN_GPU_Memcpy error\n!");
			return;
		}
	}
	if (_kernelWeights.count < 1 && _kernelWeights.values == NULL)
	{
		printf("CNN_ConvLayer kernelWeights error\n!");
		return;
	}
	m_pstWeights = (Weights*)malloc(sizeof(Weights));		//数据

	if (NULL == m_pstWeights)
	{
		printf("CNN_ConvLayer m_pstWeights malloc error\n!");
		return;
	}
	m_pstWeights->type = _kernelWeights.type;
	m_pstWeights->count = _kernelWeights.count;

	int iTypeSize = Get_Type_Szie(_biasWeights.type);

	if (0 == iTypeSize)
	{
		return;
	}

	m_pstWeights->values = CNN_GPU_MemMaloc(0, _kernelWeights.count * iTypeSize);

	if (NULL == m_pstWeights->values)
	{
		printf("CNN_ConvLayer m_pstWeights->values malloc error\n!");
		return;
	}

	int iStatus = CNN_GPU_Memcpy(_kernelWeights.count * iTypeSize, _kernelWeights.values, (void*)(m_pstWeights->values));

	if (iStatus != 0)
	{
		printf("CNN_ConvLayer m_pstWeights->values CNN_GPU_Memcpy error\n!");
		return;
	}

	m_nbOutputMaps = _nbOutputMaps;
	m_stKernel = _kernelSize;

	//卷积参数
	m_iGroup = 0;
	m_stDilation = DimsHW(1, 1);	//核宽度方向填充像素数
	m_stStride = DimsHW(1, 1);	//宽度方向步长
	m_stPadding = DimsHW(0, 0);	//宽度方向填充像素数

	//m_iChannelAxis	= 0;			//通道数所在维度（后续扩展用）
}

IConvolutionLayer::~IConvolutionLayer()
{
	if (NULL != m_pstBias)
	{
		if (NULL != m_pstBias->values)
		{
			CNN_GPU_MemFree((void*)(m_pstBias->values));
			m_pstBias->values = NULL;
		}
		free(m_pstBias);
		m_pstBias = NULL;
	}
	if (NULL != m_pstWeights)
	{
		if (NULL != m_pstWeights->values)
		{
			CNN_GPU_MemFree((void*)(m_pstWeights->values));
			m_pstWeights->values = NULL;
		}
		free(m_pstWeights);
		m_pstWeights = NULL;
	}
}

int IConvolutionLayer::forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut)
{
	cudnnHandle_t handle;
	cudnnCreate(&handle);

	_stOutPut.nbDims = _stInPut.nbDims;
	_stOutPut.d[0] = _stInPut.d[0];//n
	_stOutPut.d[1] = m_stKernel.d[0];	// c 
	int iFeatMap_h = _stOutPut.d[2] = 1 + (_stInPut.d[2] + 2 * m_stPadding.d[0] - m_stKernel.d[2]) / m_stStride.d[0]; //h
	int iFeatMap_w = _stOutPut.d[3] = 1 + (_stInPut.d[3] + 2 * m_stPadding.d[1] - m_stKernel.d[3]) / m_stStride.d[1]; //w

	cudnnTensorDescriptor_t input_descriptor;
	cudnnCreateTensorDescriptor(&input_descriptor);
	cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, _stInPut.d[0], _stInPut.d[1], _stInPut.d[2], _stInPut.d[3]);

	cudnnTensorDescriptor_t output_descriptor;
	cudnnCreateTensorDescriptor(&output_descriptor);
	cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, _stOutPut.d[0], _stOutPut.d[1], _stOutPut.d[2], _stOutPut.d[3]);

	cudnnFilterDescriptor_t kernel_descriptor;
	cudnnCreateFilterDescriptor(&kernel_descriptor);
	cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, m_stKernel.d[0], m_stKernel.d[1], m_stKernel.d[2], m_stKernel.d[3]);
	// convolution descriptor

	cudnnConvolutionDescriptor_t conv_descriptor;
	cudnnCreateConvolutionDescriptor(&conv_descriptor);
	cudnnSetConvolution2dDescriptor(conv_descriptor,
		m_stPadding.d[0], m_stPadding.d[1], // zero-padding
		m_stStride.d[0], m_stStride.d[1], // stride
		1, 1,
		CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

	cudnnStatus_t eStatus = CUDNN_STATUS_SUCCESS;
	cudnnConvolutionFwdAlgo_t algo;
	// algorithm
#if CUDNN_VERSION_MIN(8, 0, 0)
	int returnedAlgoCount = 0;
	int requestedAlgoCount = 1;
	cudnnConvolutionFwdAlgoPerf_t fwd_algoPer;

	eStatus = cudnnGetConvolutionForwardAlgorithm_v7(handle, input_descriptor,
		kernel_descriptor, conv_descriptor, output_descriptor, requestedAlgoCount,
		&returnedAlgoCount, &fwd_algoPer);
	algo = fwd_algoPer.algo;

	algo = fwd_algoPer.algo;

	if (CUDNN_STATUS_SUCCESS != eStatus)
	{
		printf("cudnnGetConvolutionForwardAlgorithm_v7 error code:%d\n", eStatus);
		return -1;
	}

#else
	// choose forward and backward algorithms + workspace(s)
	eStatus = cudnnGetConvolutionForwardAlgorithm(*((cudnnHandle_t*)pstGlobalInfos->pCuDNNHandles), pInputDesc,
		pstCudnnConv->pFilterDesc, pConvDesc, pOutputDesc, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
		workspace_limit_bytes, &pstCudnnConv->fwd_algo_[i]);
#endif

	// workspace size && allocate memory
	size_t workspace_size = 0;
	eStatus = cudnnGetConvolutionForwardWorkspaceSize(handle, input_descriptor, kernel_descriptor, conv_descriptor, output_descriptor, algo, &workspace_size);

	if (CUDNN_STATUS_SUCCESS != eStatus)
	{
		printf("cudnnGetConvolutionForwardWorkspaceSize error code:%d\n!", eStatus);
		return -2;
	}

	void * workspace = nullptr;
	cudaMalloc(&workspace, workspace_size);

	// convolution
	auto alpha = 1.0f, beta = 0.0f;
	eStatus = cudnnConvolutionForward(handle, &alpha, input_descriptor, _pInData, kernel_descriptor, m_pstWeights->values,
		conv_descriptor, algo, workspace, workspace_size, &beta, output_descriptor, _pOutData);

	if (CUDNN_STATUS_SUCCESS != eStatus)
	{
		printf("cudnnConvolutionForward error code:%d\n!", eStatus);
		return -3;
	}

	if (true == m_bHasBias)
	{
		cudnnTensorDescriptor_t			bias_descriptor;
		cudnnCreateTensorDescriptor(&bias_descriptor);
		//cudnnSetTensor4dDescriptorEx(bias_descriptor, CUDNN_DATA_FLOAT, _stInPut.d[0], _stInPut.d[1], 1, 1, _stInPut.d[0], _stInPut.d[1], 1, 1);
		eStatus = setTensor4dDesc<float>(&bias_descriptor, 1, m_pstBias->count, 1, 1);
		auto alpha = 1.0f, beta = 1.0f;
		eStatus = cudnnAddTensor(handle, &alpha, bias_descriptor, m_pstBias->values, &beta, output_descriptor, _pOutData);

		if (CUDNN_STATUS_SUCCESS != eStatus)
		{
			printf("cudnnAddTensor error code:%d\n!", eStatus);
			return -4;
		}

		cudnnDestroyTensorDescriptor(bias_descriptor);
	}

	cudaFree(workspace);

	cudnnDestroyTensorDescriptor(input_descriptor);
	cudnnDestroyTensorDescriptor(output_descriptor);
	cudnnDestroyConvolutionDescriptor(conv_descriptor);
	cudnnDestroyFilterDescriptor(kernel_descriptor);
	cudnnDestroy(handle);

	return 0;
}




cublasStatus_t CNN_Util_Math_Gemm_GPU(const CBLAS_TRANSPOSE TransA,
	const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
	const float alpha, const float* A, const float* B, const float beta, float* C, cublasHandle_t _hCuBLAS)
{
	// Note that cublas follows fortran order.
	int lda = (TransA == CblasNoTrans) ? K : M;
	int ldb = (TransB == CblasNoTrans) ? N : K;
	cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	return cublasSgemm(_hCuBLAS, cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N);
}


template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y)
{
	CUDA_KERNEL_LOOP(index, n)
	{
		y[index] = alpha;
	}
}

template <typename Dtype>
void CNN_Util_Math_Set_GPU(const int N, const Dtype alpha, Dtype* Y)
{
	if (alpha == 0)
	{
		cudaError_t eErr = cudaMemset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
		return;
	}
	// NOLINT_NEXT_LINE(whitespace/operators)
	set_kernel<Dtype> << <CNN_GET_BLOCKS(N), CNN_CUDA_NUM_THREADS >> >(N, alpha, Y);

	cudaDeviceSynchronize();
}

template void CNN_Util_Math_Set_GPU<int>(const int N, const int alpha, int* Y);
template void CNN_Util_Math_Set_GPU<float>(const int N, const float alpha, float* Y);
template void CNN_Util_Math_Set_GPU<double>(const int N, const double alpha, double* Y);

int IConvolutionLayer::forwardGMM(void* _pInData, Dims _stInPut, void* _pOutData, 
	Dims &_stOutPut, void *_pBiasMultip, void *_pBuffer)
{
#if 1
	_stOutPut.nbDims = _stInPut.nbDims;
	_stOutPut.d[0] = _stInPut.d[0];//n
	_stOutPut.d[1] = m_stKernel.d[0];	// c 
	int iFeatMap_h = _stOutPut.d[2] = 1 + (_stInPut.d[2] + 2 * m_stPadding.d[0] - m_stKernel.d[2]) / m_stStride.d[0]; //h
	int iFeatMap_w = _stOutPut.d[3] = 1 + (_stInPut.d[3] + 2 * m_stPadding.d[1] - m_stKernel.d[3]) / m_stStride.d[1]; //w


	int M = m_stKernel.d[0];
	int N = iFeatMap_h * iFeatMap_w;
	int K = _stInPut.d[1] * m_stKernel.d[2] * m_stKernel.d[3];
	int iInputStep = _stInPut.d[1] * _stInPut.d[2] * _stInPut.d[3];
	int iOutputStep = M * N;
	bool b1x1 = (1 == m_stKernel.d[2]) && (1 == m_stKernel.d[3]);
	cublasHandle_t hCuBLAS = NULL;

	cublasCreate_v2(&hCuBLAS);

	const float* weight = (float*)(m_pstWeights->values);

	int iBuffSize = sizeof(float)* N * m_stKernel.d[2] * m_stKernel.d[3] * _stInPut.d[1] * _stInPut.d[0];

	float *gmm_buf = nullptr;

	for (int n = 0; n < _stInPut.d[0]; ++n)
	{
		float* col_buff = ((float*)_pInData) + n * iInputStep;
		if (!b1x1)
		{
			if (nullptr == gmm_buf) {
				cudaMalloc((void**)&gmm_buf, iBuffSize);//
			}
			if (nullptr == gmm_buf)
			{
				printf("forwardGMM cudaMalloc gmm_buf error\n");
				continue;
			}

			CNN_Im2Col_GPU<float>(col_buff, _stInPut.d[1], _stInPut.d[2], _stInPut.d[3],
				m_stKernel.d[2], m_stKernel.d[3],
				m_stPadding.d[0], m_stPadding.d[1],
				m_stStride.d[0], m_stStride.d[1],
				m_stDilation.d[0], m_stDilation.d[1],
				(float*)gmm_buf);
			col_buff = (float*)gmm_buf;
		}
		cublasStatus_t eStatus = CNN_Util_Math_Gemm_GPU(CblasNoTrans, CblasNoTrans, M, N, K,
			1., weight, col_buff, 0., (float*)_pOutData + n * iOutputStep, hCuBLAS);

		if (CUBLAS_STATUS_SUCCESS != eStatus)
		{
			printf("CNN_Util_Math_Gemm_GPU error:%d\n", eStatus);
		}

		if (m_bHasBias)
		{
			//int iOutSize = pstBasicInfos->iOutputW * pstBasicInfos->iOutputH;
			void * tmp_buf = nullptr;

			cudaMalloc((void**)&tmp_buf, N * sizeof(float));//

			CNN_Util_Math_Set_GPU(N, 1.0f, (float*)tmp_buf);

			eStatus = CNN_Util_Math_Gemm_GPU(CblasNoTrans, CblasNoTrans, M, N, 1,
				1., (float*)(m_pstBias->values), (float*)tmp_buf, 1., (float*)_pOutData + n * iOutputStep, hCuBLAS);

			if (CUBLAS_STATUS_SUCCESS != eStatus)
			{
				printf("CNN_Util_Math_Gemm_GPU error:%d\n", eStatus);
			}

			cudaFree(tmp_buf);
		}
	}
	if (gmm_buf != nullptr)
	{
		cudaFree(gmm_buf);
	}
	cublasDestroy_v2(hCuBLAS);
	return 0;
#else
	_stOutPut.nbDims = _stInPut.nbDims;
	_stOutPut.d[0] = _stInPut.d[0];//n
	_stOutPut.d[1] = m_stKernel.d[0];	// c 
	int iFeatMap_h = _stOutPut.d[2] = 1 + (_stInPut.d[2] + 2 * m_stPadding.d[0] - m_stKernel.d[2]) / m_stStride.d[0]; //h
	int iFeatMap_w = _stOutPut.d[3] = 1 + (_stInPut.d[3] + 2 * m_stPadding.d[1] - m_stKernel.d[3]) / m_stStride.d[1]; //w

	int M = m_stKernel.d[0];
	int N = iFeatMap_h * iFeatMap_w;
	int K = _stInPut.d[1] * m_stKernel.d[2] * m_stKernel.d[3];
	fecnn::StorageT* weights = (fecnn::StorageT*)(m_pstWeights->values);

	fecnn::StorageT* col_buff = nullptr;
	int iBuffSize = sizeof(fecnn::StorageT)* N * m_stKernel.d[2] * m_stKernel.d[3] * _stInPut.d[1] * _stInPut.d[0];
	//float *gmm_buf = nullptr;
	
	if (nullptr == col_buff) {
		cudaMalloc((void**)&col_buff, iBuffSize);//
	}
	if (nullptr == col_buff)
	{
		printf("forwardGMM cudaMalloc gmm_buf error\n");
		return -2;
	}
	fecnn::im2col_gpu((fecnn::StorageT*)_pInData, _stInPut.d[1], _stInPut.d[2], _stInPut.d[3],
		m_stKernel.d[2], m_stKernel.d[3],
		m_stPadding.d[0], m_stPadding.d[1],
		m_stStride.d[0], m_stStride.d[1],
		1, 1, col_buff);

	cublasHandle_t hCuBLAS = NULL;

	cublasCreate_v2(&hCuBLAS);

	fecnn::fecnn_gpu_gemm(hCuBLAS, CUBLAS_OP_N, CUBLAS_OP_T,
		M, N, K,
		(fecnn::StorageT)1., (fecnn::StorageT*)_pOutData,
		col_buff,
		(fecnn::StorageT)1., weights);

	void * bias_multGPU = nullptr;

	cudaMalloc((void**)&bias_multGPU, N * sizeof(fecnn::StorageT));//

	CNN_Util_Math_Set_GPU(N, 1.0f, (fecnn::StorageT*)bias_multGPU);

	fecnn::fecnn_gpu_gemm(hCuBLAS, CUBLAS_OP_N, CUBLAS_OP_N,
		M, N, 1,
		(fecnn::StorageT)1., (fecnn::StorageT*)(m_pstBias->values), (fecnn::StorageT*)bias_multGPU,
		(fecnn::StorageT)1., (fecnn::StorageT*)_pOutData);

	cudaFree(bias_multGPU);

	if (col_buff != nullptr)
	{
		cudaFree(col_buff);
	}
	cublasDestroy_v2(hCuBLAS);
	return 0;
#endif // 0
}

