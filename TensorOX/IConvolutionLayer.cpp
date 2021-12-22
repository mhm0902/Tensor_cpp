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


IConvolutionLayer::IConvolutionLayer()
{
	m_pstBias = NULL;
	m_pstWeights = NULL;
		//卷积参数
	m_iGroup = 1;
	m_stDilation	= DimsNCHW(1, 1, 1, 1);	//核宽度方向填充像素数
	m_stStride		= DimsNCHW(1, 1, 1, 1);	//宽度方向步长
	m_stPadding		= DimsNCHW(0, 0, 0, 0);	//宽度方向填充像素数

								//m_iChannelAxis	= 0;			//通道数所在维度（后续扩展用）
}
IConvolutionLayer::IConvolutionLayer(int _nbOutputMaps, Dims _kernelSize, Weights _kernelWeights, Weights _biasWeights)
{
	m_pstBias = NULL;
	m_pstWeights = NULL;

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

		int iTypeSize = 0;

		if (_biasWeights.type == DataType::kFLOAT)
		{
			iTypeSize = sizeof(float);
		}
		else if (DataType::kINT32 == _biasWeights.type)
		{
			iTypeSize = sizeof(int);
		}
		else if (DataType::kHALF == _biasWeights.type)
		{
			iTypeSize = sizeof(short);
		}
		else if (DataType::kINT8 == _biasWeights.type)
		{
			iTypeSize = sizeof(char);
		}
		else {
			printf("CNN_ConvLayer  biasWeights Type is not support! \n!");
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

	int iTypeSize = 0;

	if (_kernelWeights.type == DataType::kFLOAT)
	{
		iTypeSize = sizeof(float);
	}
	else if (DataType::kINT32 == _kernelWeights.type)
	{
		iTypeSize = sizeof(int);
	}
	else if (DataType::kHALF == _kernelWeights.type)
	{
		iTypeSize = sizeof(short);
	}
	else if (DataType::kINT8 == _kernelWeights.type)
	{
		iTypeSize = sizeof(char);
	}
	else {
		printf("CNN_ConvLayer  kernelWeights Type is not support! \n!");
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
	
	m_nbOutputMaps	= _nbOutputMaps;
	m_stKernel		= _kernelSize;
	

	//卷积参数
	m_iGroup		= 0;
	m_stDilation	= DimsNCHW(1, 1, 1, 1);	//核宽度方向填充像素数
	m_stStride		= DimsNCHW(1, 1, 1, 1);	//宽度方向步长
	m_stPadding		= DimsNCHW(0, 0, 0, 0);	//宽度方向填充像素数
	
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

int IConvolutionLayer::forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims _stOutPut)
{
	cudnnHandle_t handle;
	cudnnCreate(&handle);

	cudnnTensorDescriptor_t input_descriptor;
	cudnnCreateTensorDescriptor(&input_descriptor);
	cudnnSetTensor4dDescriptor(input_descriptor,CUDNN_TENSOR_NHWC,CUDNN_DATA_FLOAT,_stInPut.d[0], _stInPut.d[1], _stInPut.d[2], _stInPut.d[3]);

	cudnnTensorDescriptor_t output_descriptor;
	cudnnCreateTensorDescriptor(&output_descriptor);
	cudnnSetTensor4dDescriptor(output_descriptor,CUDNN_TENSOR_NHWC,CUDNN_DATA_FLOAT,_stOutPut.d[0], _stOutPut.d[1], _stOutPut.d[2], _stOutPut.d[3]);

	cudnnFilterDescriptor_t kernel_descriptor;
	cudnnCreateFilterDescriptor(&kernel_descriptor);
	cudnnSetFilter4dDescriptor(kernel_descriptor,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,m_stKernel.d[0], m_stKernel.d[1], m_stKernel.d[2], m_stKernel.d[3]);
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

#else
	// choose forward and backward algorithms + workspace(s)
	eStatus = cudnnGetConvolutionForwardAlgorithm(*((cudnnHandle_t*)pstGlobalInfos->pCuDNNHandles), pInputDesc,
		pstCudnnConv->pFilterDesc, pConvDesc, pOutputDesc, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
		workspace_limit_bytes, &pstCudnnConv->fwd_algo_[i]);
#endif

	// workspace size && allocate memory
	size_t workspace_size = 0;
	cudnnGetConvolutionForwardWorkspaceSize(handle, input_descriptor, kernel_descriptor, conv_descriptor, output_descriptor, algo, &workspace_size);

	void * workspace = nullptr;
	cudaMalloc(&workspace, workspace_size);

	// convolution
	auto alpha = 1.0f, beta = 0.0f;
	cudnnConvolutionForward(handle, &alpha, input_descriptor, _pInData, kernel_descriptor, m_pstWeights->values,
		conv_descriptor, algo,workspace, workspace_size,&beta, output_descriptor, _pOutData);

	if (true == m_bHasBias)
	{
		cudnnTensorDescriptor_t			bias_descriptor;
		cudnnCreateTensorDescriptor(&bias_descriptor);
		cudnnSetTensor4dDescriptorEx(bias_descriptor, CUDNN_DATA_FLOAT, _stInPut.d[0], _stInPut.d[1], 1, 1, _stInPut.d[0], _stInPut.d[1], 1, 1);
		eStatus = cudnnAddTensor(handle,&alpha, bias_descriptor, m_pstBias->values, &beta, output_descriptor, _pOutData);

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

Dims IConvolutionLayer::init(Dims _stInPut)
{
	Dims stOutPut;

	cudnnCreate(&m_handle);

	cudnnCreateTensorDescriptor(&m_input_descriptor);
	cudnnSetTensor4dDescriptor(m_input_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
		_stInPut.d[0], _stInPut.d[1], _stInPut.d[2], _stInPut.d[3]);

	stOutPut.nbDims = _stInPut.nbDims;
	stOutPut.d[0] = _stInPut.d[0];//n
	stOutPut.d[1] = m_stKernel.d[0];	// c 
	int iFeatMap_h = stOutPut.d[2] = 1 + (_stInPut.d[2] + 2 * m_stPadding.d[0] - m_stKernel.d[2]) / m_stStride.d[0]; //h
	int iFeatMap_w = stOutPut.d[3] = 1 + (_stInPut.d[3] + 2 * m_stPadding.d[1] - m_stKernel.d[3]) / m_stStride.d[1]; //w

	cudnnCreateTensorDescriptor(&m_output_descriptor);
	cudnnSetTensor4dDescriptor(m_output_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
		stOutPut.d[0], stOutPut.d[1], stOutPut.d[2], stOutPut.d[3]);

	cudnnCreateFilterDescriptor(&m_kernel_descriptor);
	cudnnSetFilter4dDescriptor(m_kernel_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
		m_stKernel.d[0], m_stKernel.d[1], m_stKernel.d[2], m_stKernel.d[3]);
	// convolution descriptor

	cudnnCreateConvolutionDescriptor(&m_conv_descriptor);
	cudnnSetConvolution2dDescriptor(m_conv_descriptor,
		m_stPadding.d[0], m_stPadding.d[1], // zero-padding
		m_stStride.d[0], m_stStride.d[1], // stride
		1, 1,
		CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

	cudnnStatus_t eStatus = CUDNN_STATUS_SUCCESS;
	// algorithm
#if CUDNN_VERSION_MIN(8, 0, 0)
	int returnedAlgoCount = 0;
	int requestedAlgoCount = 1;
	cudnnConvolutionFwdAlgoPerf_t fwd_algoPer;

	eStatus = cudnnGetConvolutionForwardAlgorithm_v7(m_handle, m_input_descriptor,
		m_kernel_descriptor, m_conv_descriptor, m_output_descriptor, requestedAlgoCount,
		&returnedAlgoCount, &fwd_algoPer);
	m_algo = fwd_algoPer.algo;

	if (CUDNN_STATUS_SUCCESS != eStatus)
	{
		printf("cudnnGetConvolutionForwardAlgorithm_v7 error code:%d\n", eStatus);
		return stOutPut;
	}

#else
	// choose forward and backward algorithms + workspace(s)
	eStatus = cudnnGetConvolutionForwardAlgorithm(*((cudnnHandle_t*)pstGlobalInfos->pCuDNNHandles), pInputDesc,
		pstCudnnConv->pFilterDesc, pConvDesc, pOutputDesc, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
		workspace_limit_bytes, &pstCudnnConv->fwd_algo_[i]);
#endif

	// workspace size && allocate memory
	eStatus = cudnnGetConvolutionForwardWorkspaceSize(m_handle, m_input_descriptor,
		m_kernel_descriptor, m_conv_descriptor, m_output_descriptor, m_algo, &m_workspace_size);

	if (CUDNN_STATUS_SUCCESS != eStatus)
	{
		printf("cudnnGetConvolutionForwardWorkspaceSize error code:%d\n!", eStatus);
		return stOutPut;
	}

	cudaMalloc(&m_workspace, m_workspace_size);

	cudnnCreateTensorDescriptor(&m_bias_descriptor);

	//cudnnSetTensor4dDescriptorEx(m_bias_descriptor, CUDNN_DATA_FLOAT, _stInPut.d[0], _stInPut.d[1], 1, 1, _stInPut.d[0], _stInPut.d[1], 1, 1);
	eStatus = setTensor4dDesc<float>(&m_bias_descriptor, 1, m_pstBias->count, 1, 1);

	if (CUDNN_STATUS_SUCCESS != eStatus)
	{
		printf("cudnnAddTensor error code:%d\n!", eStatus);
		return stOutPut;
	}

	return stOutPut;
}
int IConvolutionLayer::forwardEx(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut)
{
	_stOutPut.nbDims = _stInPut.nbDims;
	_stOutPut.d[0] = _stInPut.d[0];//n
	_stOutPut.d[1] = m_stKernel.d[0];	// c 
	int iFeatMap_h = _stOutPut.d[2] = 1 + (_stInPut.d[2] + 2 * m_stPadding.d[0] - m_stKernel.d[2]) / m_stStride.d[0]; //h
	int iFeatMap_w = _stOutPut.d[3] = 1 + (_stInPut.d[3] + 2 * m_stPadding.d[1] - m_stKernel.d[3]) / m_stStride.d[1]; //w

	auto alpha = 1.0f, beta = 0.0f;
	cudnnStatus_t eStatus = cudnnConvolutionForward(m_handle, &alpha, m_input_descriptor, _pInData,
		m_kernel_descriptor, m_pstWeights->values,
		m_conv_descriptor, m_algo, m_workspace, m_workspace_size,
		&beta, m_output_descriptor, _pOutData);

	if (CUDNN_STATUS_SUCCESS != eStatus)
	{
		printf("cudnnConvolutionForward error code:%d\n!", eStatus);
		return -3;
	}

	if (true == m_bHasBias)
	{
		eStatus = cudnnAddTensor(m_handle, &alpha, m_bias_descriptor, m_pstBias->values,
			&beta, m_output_descriptor, _pOutData);

		if (CUDNN_STATUS_SUCCESS != eStatus)
		{
			printf("cudnnAddTensor error code:%d\n!", eStatus);
			return -4;
		}
	}
	return 0;
}
