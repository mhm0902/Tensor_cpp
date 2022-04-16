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
	//�������ݵ�ַ���Դ��ַ
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
	Weights	*m_pstBias;		//Bias����
	Weights	*m_pstWeights;	//���Ȩ��
	int m_nbOutputMaps;

	//�������
	int		m_iGroup;		//�ɱ���Ԥ��
	Dims	m_stKernel;		//����˿��
	Dims	m_stStride;		//��ȷ��򲽳�
	Dims	m_stPadding;	//��ȷ������������
	Dims	m_stDilation;	//�˿�ȷ������������ ���;��Ԥ��
	bool	m_bHasBias;
	ActivateMode m_eAct;	//����

private:
	//int Init_Conv_param();
};
#endif//__ICONVOLUTION_LAYER_H__