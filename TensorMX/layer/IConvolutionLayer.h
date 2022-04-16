#ifndef __ICONVOLUTION_LAYER_H__
#define __ICONVOLUTION_LAYER_H__

#include "algo_common.h"
#include "IActivationLayer.h"

typedef enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113, CblasConjNoTrans = 114 } CBLAS_TRANSPOSE;

template <typename Dtype>
inline cudnnStatus_t setTensor4dDesc(cudnnTensorDescriptor_t* desc, int n, int c, int h, int w);

class IConvolutionLayer
{
public:
	IConvolutionLayer();
	IConvolutionLayer(int _nbOutputMaps, Dims _kernelSize, Weights _kernelWeights, Weights _biasWeights);
	IConvolutionLayer(Dims _kernelSize, Weights _kernelWeights, Weights _biasWeights, Weights _gammaWeights,
		Weights _betaWeights, Weights _meanWeights, Weights _varWeights, float _fEps);
	~IConvolutionLayer();
	//传入数据地址是显存地址
	int forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut);
	int forwardGMM(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut);
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
	int setActivateMode(ActivateMode _eAct)
	{
		m_eAct = _eAct;
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
	Dims	m_stDilation;	//核宽度方向填充像素数 膨胀卷积预留
	bool	m_bHasBias;
	ActivateMode m_eAct;	//激活

private:
	//int Init_Conv_param();
};
#endif//__ICONVOLUTION_LAYER_H__