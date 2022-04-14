#include "IConvolutionLayer_BN.h"
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

#include "opencv2\opencv.hpp"
#include "MathFunctions.h"

//后期优化实现
IConvolutionLayer_BN::IConvolutionLayer_BN(Dims _kernelSize, Weights _kernelWeights, Weights _biasWeights,
	Weights _gammaWeights, Weights _betaWeights, Weights _meanWeights, Weights _varWeights, float _fEps)
{
	m_pstBias = NULL;
	m_pstWeights = NULL;
	m_bHasBias = true;

	if (_kernelWeights.count < 1 && _kernelWeights.values == NULL)
	{
		printf("CNN_ConvLayer kernelWeights error\n!");
		return;
	}

	if (_biasWeights.count > 0 && _biasWeights.values != NULL)//以后遇到再实现
	{		
		m_pstBias = (Weights*)malloc(sizeof(Weights));		//Bias数据

		if (NULL == m_pstBias)
		{
			printf("CNN_ConvLayer m_pstBias malloc error\n!");
			return;
		}
		memset(m_pstBias, 0, sizeof(Weights));

		m_pstBias->type = _betaWeights.type;
		m_pstBias->count = _betaWeights.count;

		int iTypeSize = Get_Type_Szie(_betaWeights.type);

		if (0 == iTypeSize)
		{
			free(m_pstBias);
			m_pstBias = NULL;
			return;
		}

		int iBiasNum = _betaWeights.count;

		float* pfBais = (float*)malloc(_betaWeights.count*iTypeSize);

		float* pfBeta = (float *)(_betaWeights.values);
		float* pfGamm = (float*)(_gammaWeights.values);
		float* pfMean = (float*)(_meanWeights.values);
		float* pfVarr = (float*)(_varWeights.values);
		float* pfSrcB = (float*)(_biasWeights.values);

		for (int j = 0; j < iBiasNum; j++)
		{
			double fGam = pfGamm[j];
			double fMean = pfMean[j];
			double fVarr = pfVarr[j];
			double fSrcB = pfSrcB[j];

			fVarr = sqrt(fVarr + _fEps);
			pfBais[j] = pfBeta[j] + fGam * (fSrcB - fMean) / fVarr;

		}

		m_pstBias->values = CNN_GPU_MemMaloc(0, _betaWeights.count * iTypeSize);

		if (NULL == m_pstBias->values)
		{
			free(m_pstBias);
			m_pstBias = NULL;
			free(pfBais);
			printf("CNN_ConvLayer m_pstBias->values malloc error\n!");
			return;
		}

		int iStatus = CNN_GPU_Memcpy(_betaWeights.count * iTypeSize, (void*)pfBais, (void*)(m_pstBias->values));

		free(pfBais);

		if (iStatus != 0)
		{
			free(m_pstBias);
			m_pstBias = NULL;
			printf("CNN_ConvLayer m_pstBias->values CNN_GPU_Memcpy error\n!");
			return;
		}
	}
	else
	{
		m_pstBias = (Weights*)malloc(sizeof(Weights));		//Bias数据

		if (NULL == m_pstBias)
		{
			printf("CNN_ConvLayer m_pstBias malloc error\n!");
			return;
		}
		memset(m_pstBias, 0, sizeof(Weights));

		m_pstBias->type = _betaWeights.type;
		m_pstBias->count = _betaWeights.count;

		int iTypeSize = Get_Type_Szie(_betaWeights.type);

		if (0 == iTypeSize)
		{
			free(m_pstBias);
			m_pstBias = NULL;
			return;
		}

		int iBiasNum = _betaWeights.count;

		float* pfBais = (float*)malloc(_betaWeights.count*iTypeSize);

		float* pfBeta = (float *)(_betaWeights.values);
		float* pfGamm = (float*)(_gammaWeights.values);
		float* pfMean = (float*)(_meanWeights.values);
		float* pfVarr = (float*)(_varWeights.values);

		for (int j = 0; j < iBiasNum; j++)
		{
			double fGam = pfGamm[j];
			double fMean = pfMean[j];
			double fVarr = pfVarr[j];

			//if (abs(fVarr) < _fEps)
			//{
				fVarr = sqrt(fVarr + _fEps);
			//}
			//else
			//{
			//	fVarr = sqrt(fVarr);
			//}
			//double fVar = sqrt(pfVarr[j] + _fEps);
			pfBais[j] = (pfBeta[j] - fGam * fMean / fVarr);
			//pfBais++;
		}

		m_pstBias->values = CNN_GPU_MemMaloc(0, _betaWeights.count * iTypeSize);

		if (NULL == m_pstBias->values)
		{
			free(m_pstBias);
			m_pstBias = NULL;
			free(pfBais);
			printf("CNN_ConvLayer m_pstBias->values malloc error\n!");
			return;
		}

		int iStatus = CNN_GPU_Memcpy(_betaWeights.count * iTypeSize, (void*)pfBais, (void*)(m_pstBias->values));

		free(pfBais);

		if (iStatus != 0)
		{
			free(m_pstBias);
			m_pstBias = NULL;
			printf("CNN_ConvLayer m_pstBias->values CNN_GPU_Memcpy error\n!");
			return;
		}
	}

	m_pstWeights = (Weights*)malloc(sizeof(Weights));		//数据

	if (NULL == m_pstWeights)
	{
		printf("CNN_ConvLayer m_pstWeights malloc error\n!");
		return;
	}
	memset(m_pstWeights, 0, sizeof(Weights));
	m_pstWeights->type = _kernelWeights.type;
	m_pstWeights->count = _kernelWeights.count;

	int iTypeSize = Get_Type_Szie(_kernelWeights.type);

	if (0 == iTypeSize)
	{
		free(m_pstWeights);
		m_pstWeights = NULL;
		return;
	}

	m_pstWeights->values = CNN_GPU_MemMaloc(0, _kernelWeights.count * iTypeSize);

	if (NULL == m_pstWeights->values)
	{
		free(m_pstWeights);
		m_pstWeights = NULL;
		printf("CNN_ConvLayer m_pstWeights->values malloc error\n!");
		return;
	}

	int iKernelSize = _kernelSize.d[1] * _kernelSize.d[2] * _kernelSize.d[3];//chw
	int iKernelNum = _kernelSize.d[0];//n					_kernelWeights.count / iKernelSize;
	float * pfKernel = (float *)(_kernelWeights.values);
	float* pfGamm = (float*)(_gammaWeights.values);
	float* pfVarr = (float*)(_varWeights.values);

	float* pfKWeights = (float*)malloc(_kernelWeights.count * iTypeSize);

	if (NULL == pfKWeights)
	{
		free(m_pstWeights);
		m_pstWeights = NULL;
		printf("CNN_ConvLayer pfKWeights malloc error\n!");
		return;
	}
	memset(pfKWeights, 0, _kernelWeights.count * iTypeSize);

	for (int j = 0; j < iKernelNum; j++)
	{
		//权重更新
		double fGam = pfGamm[j];
		//double fVar = sqrt( pfVarr[j] + _fEps );
		double fVarr = pfVarr[j];

		//if (abs(fVarr) < _fEps)
		//{
			fVarr = sqrt(fVarr + _fEps);
		//}
		//else
		//{
		//	fVarr = sqrt(fVarr);
		//}
		double fW = fGam / fVarr;

		for (int k = 0; k < iKernelSize; k++)
		{
			pfKWeights[j*iKernelSize + k] = (pfKernel[j*iKernelSize + k] * fW);
		}
	}
	//std::ofstream outfile("outw.txt", std::ios::trunc);
	//float* pfRes = (float*)pfKWeights;
	//int chanle =  _kernelSize.d[1];
	//int height =  _kernelSize.d[2];
	//int weight =  _kernelSize.d[3];
	//char acBuf[50] = {0};
	//for (int n = 0; n < _kernelSize.d[0]; n++) {
	//	outfile << "n:" << n << std::endl;
	//	for (int c = 0; c < chanle; c++){
	//		outfile << "c:"<< c << std::endl;
	//		for (int h = 0; h < height; h++){
	//			for (int w = 0; w < weight; w++){
	//				sprintf(acBuf, " %f ", pfRes[n*chanle*height *weight + c*height *weight + h*weight + w]);
	//				outfile.write(acBuf, strlen(acBuf));
	//				//outfile << std::setiosflags(std::ios::left) << std::setw(20) << std::setfill(' ') << pfRes[n*chanle*height *weight +c*height *weight + h*weight + w] << " ";
	//			}
	//			outfile << std::endl;
	//		}
	//	}		
	//}
	//outfile.close();
	int iStatus = CNN_GPU_Memcpy(_kernelWeights.count * iTypeSize, (void*)pfKWeights, (void*)(m_pstWeights->values));

	free(pfKWeights);

	if (iStatus != 0)
	{
		free(m_pstWeights);
		m_pstWeights = NULL;
		printf("CNN_ConvLayer m_pstWeights->values CNN_GPU_Memcpy error\n!");
		return;
	}

	m_stKernel = _kernelSize;

	//卷积参数
	m_iGroup = 1;
	m_stDilation	= DimsHW(1, 1);	//核宽度方向填充像素数
	m_stStride		= DimsHW(1, 1);	//宽度方向步长
	m_stPadding		= DimsHW(0, 0);	//宽度方向填充像素数
}


IConvolutionLayer_BN::~IConvolutionLayer_BN()
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
template <typename Dtype>
inline cudnnStatus_t setTensor4dDesc(cudnnTensorDescriptor_t* desc, int n, int c, int h, int w,
	int stride_n, int stride_c, int stride_h, int stride_w)
{
	return cudnnSetTensor4dDescriptorEx(*desc, CUDNN_DATA_FLOAT, n, c, h, w, stride_n, stride_c, stride_h, stride_w);
}

template cudnnStatus_t setTensor4dDesc<float>(cudnnTensorDescriptor_t* desc, int n, int c, int h, int w,
	int stride_n, int stride_c, int stride_h, int stride_w);
template cudnnStatus_t setTensor4dDesc<double>(cudnnTensorDescriptor_t* desc, int n, int c, int h, int w,
	int stride_n, int stride_c, int stride_h, int stride_w);

template <typename Dtype>
inline cudnnStatus_t setTensor4dDesc(cudnnTensorDescriptor_t* desc, int n, int c, int h, int w)
{
	const int stride_w = 1;
	const int stride_h = w * stride_w;
	const int stride_c = h * stride_h;
	const int stride_n = c * stride_c;
	return setTensor4dDesc<Dtype>(desc, n, c, h, w, stride_n, stride_c, stride_h, stride_w);
}

template cudnnStatus_t setTensor4dDesc<float>(cudnnTensorDescriptor_t* desc, int n, int c, int h, int w);
template cudnnStatus_t setTensor4dDesc<double>(cudnnTensorDescriptor_t* desc, int n, int c, int h, int w);

int IConvolutionLayer_BN::forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut)
{
	int64 iStart = cv::getTickCount();
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
	int64 iEnd = cv::getTickCount();

	printf("__build param %f\n", (iEnd - iStart) * 1000.0 / cv::getTickFrequency());
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
	eStatus = cudnnConvolutionBiasActivationForward(handle, &alpha1, input_descriptor, _pInData, kernel_descriptor, m_pstWeights->values,
		conv_descriptor, algo, workspace, workspace_size, &beta, output_descriptor, _pOutData, bias_descriptor, m_pstBias->values, act_descriptor,
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
}