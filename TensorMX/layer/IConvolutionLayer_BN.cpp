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

#include "opencv2/opencv.hpp"

//�����Ż�ʵ��
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

	if (_biasWeights.count > 0 && _biasWeights.values != NULL)//�Ժ�������ʵ��
	{		
		
	}
	else
	{
		m_pstBias = (Weights*)malloc(sizeof(Weights));		//Bias����

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

			if (abs(fVarr) < _fEps)
			{
				fVarr = sqrt(fVarr + _fEps);
			}
			else
			{
				fVarr = sqrt(fVarr);
			}
			//double fVar = sqrt(pfVarr[j] + _fEps);
			pfBais[j] = pfBeta[j] - fGam * fMean / fVarr;
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

	m_pstWeights = (Weights*)malloc(sizeof(Weights));		//����

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
		//Ȩ�ظ���
		double fGam = pfGamm[j];
		//double fVar = sqrt( pfVarr[j] + _fEps );
		double fVarr = pfVarr[j];

		if (abs(fVarr) < _fEps)
		{
			fVarr = sqrt(fVarr + _fEps);
		}
		else
		{
			fVarr = sqrt(fVarr);
		}
		double fW = fGam / fVarr;

		for (int k = 0; k < iKernelSize; k++)
		{
			pfKWeights[j*iKernelSize + k] = pfKernel[j*iKernelSize + k] * fW;
		}
	}

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

	//��������
	m_iGroup = 1;
	m_stDilation	= DimsHW(1, 1);	//�˿��ȷ������������
	m_stStride		= DimsHW(1, 1);	//���ȷ��򲽳�
	m_stPadding		= DimsHW(0, 0);	//���ȷ������������
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

	if (CUDNN_STATUS_SUCCESS != eStatus )
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

	cudaFree(workspace);

	cudnnDestroyTensorDescriptor(input_descriptor);
	cudnnDestroyTensorDescriptor(output_descriptor);
	cudnnDestroyConvolutionDescriptor(conv_descriptor);
	cudnnDestroyFilterDescriptor(kernel_descriptor);
	cudnnDestroy(handle);

	return 0;
}


int IConvolutionLayer_BN::forwardGMM(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut, void *_pBiasMultip, void *_pBuffer)
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
