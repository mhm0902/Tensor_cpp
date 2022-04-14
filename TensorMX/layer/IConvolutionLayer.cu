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


IConvolutionLayer::IConvolutionLayer()
{
	m_pstBias = NULL;
	m_pstWeights = NULL;
	m_bHasBias = false;
	m_eAct = Invalid;
	//��������
	m_iGroup = 1;
	m_stDilation = DimsHW(1, 1);	//�˿��ȷ������������
	m_stStride = DimsHW(1, 1);	//���ȷ��򲽳�
	m_stPadding = DimsHW(0, 0);	//���ȷ������������

								//m_iChannelAxis	= 0;			//ͨ��������ά�ȣ�������չ�ã�
}
IConvolutionLayer::IConvolutionLayer(int _nbOutputMaps, Dims _kernelSize, Weights _kernelWeights, Weights _biasWeights)
{
	m_pstBias = NULL;
	m_pstWeights = NULL;
	m_bHasBias = false;

	if (_biasWeights.count > 0 && _biasWeights.values != NULL)
	{
		m_bHasBias = true;
		m_pstBias = (Weights*)malloc(sizeof(Weights));		//Bias����

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
	m_pstWeights = (Weights*)malloc(sizeof(Weights));		//����

	if (NULL == m_pstWeights)
	{
		printf("CNN_ConvLayer m_pstWeights malloc error\n!");
		return;
	}
	m_pstWeights->type = _kernelWeights.type;
	m_pstWeights->count = _kernelWeights.count;

	int iTypeSize = Get_Type_Szie(_kernelWeights.type);

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

	//��������
	m_iGroup = 0;
	m_stDilation = DimsHW(1, 1);	//�˿��ȷ������������
	m_stStride = DimsHW(1, 1);	//���ȷ��򲽳�
	m_stPadding = DimsHW(0, 0);	//���ȷ������������

	//m_iChannelAxis	= 0;			//ͨ��������ά�ȣ�������չ�ã�
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
	//int64 iStart = cv::getTickCount();
	cudnnHandle_t handle;
	cudnnCreate(&handle);

	cudnnTensorDescriptor_t input_descriptor;
	cudnnCreateTensorDescriptor(&input_descriptor);
	cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, _stInPut.d[0], _stInPut.d[1], _stInPut.d[2], _stInPut.d[3]);

	_stOutPut.nbDims = _stInPut.nbDims;
	_stOutPut.d[0] = _stInPut.d[0];//n
	_stOutPut.d[1] = m_stKernel.d[0];	// c 
	int iFeatMap_h = _stOutPut.d[2] = 1 + (_stInPut.d[2] + 2 * m_stPadding.d[0] - m_stKernel.d[2]) / m_stStride.d[0]; //h
	int iFeatMap_w = _stOutPut.d[3] = 1 + (_stInPut.d[3] + 2 * m_stPadding.d[1] - m_stKernel.d[3]) / m_stStride.d[1]; //w


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
	//int64 iEnd = cv::getTickCount();

	//printf("__build param %f\n", (iEnd - iStart) * 1000.0 / cv::getTickFrequency());
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
#if 1

	cudnnTensorDescriptor_t			bias_descriptor;
	cudnnCreateTensorDescriptor(&bias_descriptor);
	eStatus = setTensor4dDesc<float>(&bias_descriptor, 1, m_pstBias->count, 1, 1);

	cudnnActivationDescriptor_t act_descriptor;
	eStatus = cudnnCreateActivationDescriptor(&act_descriptor);

	eStatus = cudnnSetActivationDescriptor(act_descriptor, CUDNN_ACTIVATION_IDENTITY, CUDNN_PROPAGATE_NAN, 0.);

	auto alpha1 = 1.0f, beta = 0.0f;
	eStatus = cudnnConvolutionBiasActivationForward(handle,&alpha1, input_descriptor, _pInData, kernel_descriptor, m_pstWeights->values,
		conv_descriptor, algo, workspace, workspace_size,&beta, output_descriptor, _pOutData, bias_descriptor, m_pstBias->values, act_descriptor,
		output_descriptor, _pOutData
	);

	cudnnDestroyTensorDescriptor(bias_descriptor);
	cudnnDestroyActivationDescriptor(act_descriptor);
#else


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
		//eStatus = setTensor4dDesc<double>(&bias_descriptor, 1, m_pstBias->count, 1, 1);
		auto alpha = 1.0f, beta = 1.0f;
		eStatus = cudnnAddTensor(handle, &alpha, bias_descriptor, m_pstBias->values, &beta, output_descriptor, _pOutData);

		if (CUDNN_STATUS_SUCCESS != eStatus)
		{
			printf("cudnnAddTensor error code:%d\n!", eStatus);
			return -4;
		}

		cudnnDestroyTensorDescriptor(bias_descriptor);
	}
#endif // 1
	cudaFree(workspace);

	cudnnDestroyTensorDescriptor(input_descriptor);
	cudnnDestroyTensorDescriptor(output_descriptor);
	cudnnDestroyConvolutionDescriptor(conv_descriptor);
	cudnnDestroyFilterDescriptor(kernel_descriptor);
	cudnnDestroy(handle);

	return 0;
//	//int64 iStart = cv::getTickCount();
//
//	cudnnStatus_t eStatus = CUDNN_STATUS_SUCCESS;
//	cudnnHandle_t handle;
//	eStatus = cudnnCreate(&handle);
//
//	if (CUDNN_STATUS_SUCCESS != eStatus)
//	{
//		printf("cudnnCreate error code:%d\n", eStatus);
//		return -1;
//	}
//
//	_stOutPut.nbDims = _stInPut.nbDims;
//	_stOutPut.d[0] = _stInPut.d[0];//n
//	_stOutPut.d[1] = m_stKernel.d[0];	// c 
//	int iFeatMap_h = _stOutPut.d[2] = 1 + (_stInPut.d[2] + 2 * m_stPadding.d[0] - m_stKernel.d[2]) / m_stStride.d[0]; //h
//	int iFeatMap_w = _stOutPut.d[3] = 1 + (_stInPut.d[3] + 2 * m_stPadding.d[1] - m_stKernel.d[3]) / m_stStride.d[1]; //w
//
//
//	//cudnnTensorDescriptor_t input_descriptor;
//	//eStatus = cudnnCreateTensorDescriptor(&input_descriptor);
//	//eStatus = cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, _stInPut.d[0], _stInPut.d[1], _stInPut.d[2], _stInPut.d[3]);
//
//	//if (CUDNN_STATUS_SUCCESS != eStatus)
//	//{
//	//	printf("cudnnSetTensor4dDescriptor error code:%d\n", eStatus);
//	//	return -1;
//	//}
//
//	//cudnnTensorDescriptor_t output_descriptor;
//	//eStatus = cudnnCreateTensorDescriptor(&output_descriptor);
//	//eStatus = cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, _stOutPut.d[0], _stOutPut.d[1], _stOutPut.d[2], _stOutPut.d[3]);
//
//	//if (CUDNN_STATUS_SUCCESS != eStatus)
//	//{
//	//	printf("cudnnSetTensor4dDescriptor error code:%d\n", eStatus);
//	//	return -1;
//	//}
//
//	//cudnnFilterDescriptor_t kernel_descriptor;
//	//eStatus = cudnnCreateFilterDescriptor(&kernel_descriptor);
//	//eStatus = cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, m_stKernel.d[0], m_stKernel.d[1], m_stKernel.d[2], m_stKernel.d[3]);
//	//// convolution descriptor
//
//	//if (CUDNN_STATUS_SUCCESS != eStatus)
//	//{
//	//	printf("cudnnSetFilter4dDescriptor error code:%d\n", eStatus);
//	//	return -1;
//	//}
//
//	//cudnnConvolutionDescriptor_t conv_descriptor;
//	//eStatus = cudnnCreateConvolutionDescriptor(&conv_descriptor);
//	//eStatus = cudnnSetConvolution2dDescriptor(conv_descriptor,
//	//			m_stPadding.d[0], m_stPadding.d[1], // zero-padding
//	//			m_stStride.d[0], m_stStride.d[1], // stride
//	//			1, 1,
//	//			CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
//
//	//if (CUDNN_STATUS_SUCCESS != eStatus)
//	//{
//	//	printf("cudnnSetFilter4dDescriptor error code:%d\n", eStatus);
//	//	return -1;
//																													  //}
//	cudnnTensorDescriptor_t input_descriptor;
//	cudnnTensorDescriptor_t output_descriptor;
//	cudnnFilterDescriptor_t kernel_descriptor;
//	cudnnConvolutionDescriptor_t conv_descriptor;
//
//	if (cudnnCreateTensorDescriptor(&input_descriptor) != CUDNN_STATUS_SUCCESS ||
//		cudnnCreateTensorDescriptor(&output_descriptor) != CUDNN_STATUS_SUCCESS ||
//		cudnnCreateFilterDescriptor(&kernel_descriptor) != CUDNN_STATUS_SUCCESS ||
//		cudnnCreateConvolutionDescriptor(&conv_descriptor) != CUDNN_STATUS_SUCCESS
//		)
//	{
//		printf("cudnnCreate Descriptor error\n");
//		return -2;
//	}
//	int iBatchSize = _stInPut.d[0];
//	int iInChnNum = _stInPut.d[1];
//	int height = _stInPut.d[2];
//	int width = _stInPut.d[3];
//	cudnn::setTensor4dDesc<float>(&input_descriptor, iBatchSize, iInChnNum, height, width, iInChnNum * height * width, height * width, width, 1);
//
//	int iOutChnNum = _stOutPut.d[1];
//	int height_out = _stOutPut.d[2];
//	int width_out = _stOutPut.d[3];
//	int iOutputSize = height_out*width_out;
//	cudnn::setTensor4dDesc<float>(&output_descriptor, iBatchSize, iOutChnNum, height_out, width_out, iOutChnNum * iOutputSize, iOutputSize, width_out, 1);
//
//	eStatus = cudnn::createFilterDesc<float>(&kernel_descriptor, iOutChnNum, iInChnNum, m_stKernel.d[2], m_stKernel.d[3]);
//
//	cudnn::setConvolutionDesc<float>(&conv_descriptor, input_descriptor, kernel_descriptor, m_stPadding.d[0], m_stPadding.d[1], m_stStride.d[0], m_stStride.d[1]);
//
//
//	cudnnConvolutionFwdAlgo_t algo;
//	// algorithm
//#if CUDNN_VERSION_MIN(8, 0, 0)
//	int returnedAlgoCount = 0;
//	int requestedAlgoCount = 1;
//	cudnnConvolutionFwdAlgoPerf_t fwd_algoPer;
//
//	eStatus = cudnnGetConvolutionForwardAlgorithm_v7(handle, input_descriptor,
//		kernel_descriptor, conv_descriptor, output_descriptor, requestedAlgoCount,
//		&returnedAlgoCount, &fwd_algoPer);
//	algo = fwd_algoPer.algo;
//
//	if (CUDNN_STATUS_SUCCESS != eStatus)
//	{
//		printf("cudnnGetConvolutionForwardAlgorithm_v7 error code:%d\n", eStatus);
//		return -1;
//	}
//
//#else
//	// choose forward and backward algorithms + workspace(s)
//	eStatus = cudnnGetConvolutionForwardAlgorithm(*((cudnnHandle_t*)pstGlobalInfos->pCuDNNHandles), pInputDesc,
//		pstCudnnConv->pFilterDesc, pConvDesc, pOutputDesc, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
//		workspace_limit_bytes, &pstCudnnConv->fwd_algo_[i]);
//#endif
//	//int64 iEnd = cv::getTickCount();
//
//	//printf("__build param %f\n", (iEnd - iStart) * 1000.0 / cv::getTickFrequency());
//	// workspace size && allocate memory
//	size_t workspace_size = 0;
//	eStatus = cudnnGetConvolutionForwardWorkspaceSize(handle, input_descriptor, kernel_descriptor, conv_descriptor, output_descriptor, algo, &workspace_size);
//
//	if (CUDNN_STATUS_SUCCESS != eStatus)
//	{
//		printf("cudnnGetConvolutionForwardWorkspaceSize error code:%d\n!", eStatus);
//		return -2;
//	}
//	void * workspace = nullptr;
//	cudaMalloc(&workspace, workspace_size);
//
//	// convolution
//	auto alpha = 1.0f, beta = 0.0f;
//	eStatus = cudnnConvolutionForward(handle, &alpha, input_descriptor, _pInData, kernel_descriptor, m_pstWeights->values,
//		conv_descriptor, algo, workspace, workspace_size, &beta, output_descriptor, _pOutData);
//
//	if (CUDNN_STATUS_SUCCESS != eStatus)
//	{
//		printf("cudnnConvolutionForward error code:%d\n!", eStatus);
//		return -3;
//	}
//
//	if (true == m_bHasBias)
//	{
//		cudnnTensorDescriptor_t			bias_descriptor;
//		cudnnCreateTensorDescriptor(&bias_descriptor);
//		eStatus = cudnn::setTensor4dDesc<float>(&bias_descriptor, 1, iOutChnNum, 1, 1);
//		auto alpha = 1.0f, beta = 1.0f;
//		eStatus = cudnnAddTensor(handle, &alpha, bias_descriptor, m_pstBias->values, &beta, output_descriptor, _pOutData);
//
//		if (CUDNN_STATUS_SUCCESS != eStatus)
//		{
//			printf("cudnnAddTensor error code:%d\n!", eStatus);
//			return -4;
//		}
//
//		cudnnDestroyTensorDescriptor(bias_descriptor);
//	}
//
//	//sync_conv_groups << <1, 1 >> >();
//
//	cudaFree(workspace);
//
//	cudnnDestroyTensorDescriptor(input_descriptor);
//	cudnnDestroyTensorDescriptor(output_descriptor);
//	cudnnDestroyConvolutionDescriptor(conv_descriptor);
//	cudnnDestroyFilterDescriptor(kernel_descriptor);
//	cudnnDestroy(handle);
//
//
//
//	return 0;
}


template <typename Dtype>
__global__ void convolution(const int nthreads, 
	const Dtype* const _pInData,const int in_c, 
	const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int stride_h, const int stride_w, 
	const int pad_h, const int pad_w,
	const int bias_term, const Dtype* const bias_data,
	const Dtype* const weight_data,
	ActivateMode eAct,
	Dtype* const _pOutData, const int out_c, const int out_h, const int out_w
	)
{

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= nthreads)
	{
		return;
	}
	else
	{
		const int pw = index % out_w;
		const int ph = (index / out_w) % out_h;
		const int c = (index / out_w / out_h) % out_c;
		const int n = index / out_w / out_h / out_c;

		int hstart = ph * stride_h - pad_h;
		int wstart = pw * stride_w - pad_w;

		const int hend = min(hstart + kernel_h, height);
		const int wend = min(wstart + kernel_w, width);

		int k_off_h = 0;
		int k_off_w = 0;
		if (hstart < 0)
		{
			k_off_h = -hstart;
			hstart = 0;
		}
		if (wstart < 0)
		{
			k_off_w = -wstart;
			wstart = 0;
		}

		const Dtype* const bottom_slice = _pInData + (n * out_c + c) * height * width;
		Dtype sum = 0.0;

		if (bias_term) {
			sum = bias_data[c];
		}			

		const Dtype* kptr = weight_data + kernel_h * kernel_w * in_c * c + k_off_h * kernel_w;

		for (int h = hstart; h < hend; ++h)
		{
			for (int w = wstart; w < wend; ++w)
			{
				Dtype m = bottom_slice[h * width + w];

				Dtype wt = kptr[ (h- hstart) * kernel_w + (w - wstart) + k_off_w];
				sum += m * wt;

			}
		}
		switch (eAct)
		{
		case Sigmoid:
			sum = 1. / (1. + exp(-sum));
			break;
		case ReLU:

			break;
		case Tanh:
			sum = tanh(sum);
			break;
		case Silu:
			sum = sum / (1. + exp(-sum));
			break;
		default:break;
		}
		_pOutData[index] = sum;
	}
}

int IConvolutionLayer::forwardGMM(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut)
{
	const int inw = _stInPut.d[3];
	const int inh = _stInPut.d[2];
	const int inch = _stInPut.d[1];

	_stOutPut.nbDims = _stInPut.nbDims;
	_stOutPut.d[0] = _stInPut.d[0];//n
	const int outch = _stOutPut.d[1] = m_stKernel.d[0];	// c 
	const int outh = _stOutPut.d[2] = 1 + (_stInPut.d[2] + 2 * m_stPadding.d[0] - m_stKernel.d[2]) / m_stStride.d[0]; //h
	const int outw = _stOutPut.d[3] = 1 + (_stInPut.d[3] + 2 * m_stPadding.d[1] - m_stKernel.d[3]) / m_stStride.d[1]; //w

	int count = 1;//
	for (int i = 0; i < _stOutPut.nbDims; i++)
	{
		count *= _stOutPut.d[i];
	}

	convolution << <CNN_GET_BLOCKS(count), CNN_CUDA_NUM_THREADS >> >(count,
	 (float*)_pInData, inch,inh, inw,
		m_stKernel.d[2], m_stKernel.d[3], 
		m_stStride.d[0], m_stStride.d[1], 
		m_stPadding.d[0], m_stPadding.d[1],
		m_bHasBias, (float*)(m_pstBias->values), 
		(float*)(m_pstWeights->values),m_eAct,
		(float*)_pOutData, outch, outh, outw);
	
	return 0;
}

