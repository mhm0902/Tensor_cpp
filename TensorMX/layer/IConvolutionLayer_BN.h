#ifndef __ICONVOLUTION_LAYER_BN_H__
#define __ICONVOLUTION_LAYER_BN_H__



#include "algo_common.h"
class IConvolutionLayer_BN //: public IConvolutionLayer
{
public:
	IConvolutionLayer_BN(Dims _kernelSize, Weights _kernelWeights, Weights _biasWeights, Weights _gammaWeights, 
		Weights _betaWeights, Weights _meanWeights, Weights _varWeights, float _fEps);
	virtual ~IConvolutionLayer_BN();

	//�������ݵ�ַ���Դ��ַ
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
	Weights	*m_pstBias;		//Bias����
	Weights	*m_pstWeights;	//���Ȩ��
	

	//�������
	int		m_iGroup;		//�ɱ���Ԥ��
	Dims	m_stKernel;		//����˿��
	Dims	m_stStride;		//��ȷ��򲽳�
	Dims	m_stPadding;	//��ȷ������������
	Dims	m_stDilation;	//�˿�ȷ������������ �ն����Ԥ��
	bool	m_bHasBias;

	//int		m_iKernelSize;
	//int		m_iKernelNum;
};


#endif//__ICONVOLUTION_LAYER_BN_H__

