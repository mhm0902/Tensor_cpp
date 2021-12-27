#ifndef __ICONVOLUTION_LAYER_H__
#define __ICONVOLUTION_LAYER_H__

#include "algo_common.h"

typedef enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113, CblasConjNoTrans = 114 } CBLAS_TRANSPOSE;
namespace fecnn {
	// 需要与处理的数据类型保持一致
	typedef float StorageT;
	typedef float ComputeT;
	void im2col_gpu(const StorageT* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		StorageT* data_col);
	void fecnn_gpu_gemm(cublasHandle_t cublasHandle, const cublasOperation_t TransA,
		const cublasOperation_t TransB, const int M, const int N, const int K,
		const StorageT alpha, const StorageT* A, const StorageT* B, const StorageT beta,
		StorageT* C);
}
class IConvolutionLayer
{
public:
	IConvolutionLayer();
	IConvolutionLayer(int _nbOutputMaps, Dims _kernelSize, Weights _kernelWeights, Weights _biasWeights);
	~IConvolutionLayer();
	//传入数据地址是显存地址
	int forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut);
	int forwardGMM(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut, void *_pBiasMultip = NULL, void *_pBuffer = NULL);
public:
	int setPadding(Dims _iPadding)
	{
		m_stPadding = _iPadding;
		return 0;
	};

	int setKernel(Dims _iKernel)
	{
		m_stKernel = _iKernel;
		return 0;
	};

	int setStride(Dims _iStride)
	{
		m_stStride = _iStride;
		return 0;
	};

	int setDilation(Dims _iDilation)
	{
		m_stDilation = _iDilation;
		return 0;
	};

	int setGroup(int _iGroup)
	{
		m_iGroup = _iGroup;
		return 0;
	};
	
private:
	Weights	*m_pstBias;		//Bias数据
	Weights	*m_pstWeights;	//卷积权重
	int m_nbOutputMaps;

	//卷积参数
	int		m_iGroup;		//可变卷积预留
	Dims	m_stKernel;		//卷积核宽度
	Dims	m_stStride;		//宽度方向步长
	Dims	m_stPadding;	//宽度方向填充像素数
	Dims	m_stDilation;	//核宽度方向填充像素数 空洞卷积预留
	bool	m_bHasBias;

private:
	//int Init_Conv_param();
};

template <typename Dtype>
void CNN_Im2Col_GPU(const Dtype* data_im, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w, Dtype* data_col);

cublasStatus_t CNN_Util_Math_Gemm_GPU(const CBLAS_TRANSPOSE TransA,
	const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
	const float alpha, const float* A, const float* B, const float beta, float* C, cublasHandle_t _hCuBLAS);

template <typename Dtype>
void CNN_Util_Math_Set_GPU(const int N, const Dtype alpha, Dtype *X);
#endif//__ICONVOLUTION_LAYER_H__