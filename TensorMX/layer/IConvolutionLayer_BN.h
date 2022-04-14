#ifndef __ICONVOLUTION_LAYER_BN_H__
#define __ICONVOLUTION_LAYER_BN_H__



#include "algo_common.h"
class IConvolutionLayer_BN //: public IConvolutionLayer
{
public:
	IConvolutionLayer_BN(Dims _kernelSize, Weights _kernelWeights, Weights _biasWeights, Weights _gammaWeights, 
		Weights _betaWeights, Weights _meanWeights, Weights _varWeights, float _fEps);
	virtual ~IConvolutionLayer_BN();

	//传入数据地址是显存地址
	int forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut);

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
	

	//卷积参数
	int		m_iGroup;		//可变卷积预留
	Dims	m_stKernel;		//卷积核宽度
	Dims	m_stStride;		//宽度方向步长
	Dims	m_stPadding;	//宽度方向填充像素数
	Dims	m_stDilation;	//核宽度方向填充像素数 空洞卷积预留
	bool	m_bHasBias;

	//int		m_iKernelSize;
	//int		m_iKernelNum;
};


#endif//__ICONVOLUTION_LAYER_BN_H__

