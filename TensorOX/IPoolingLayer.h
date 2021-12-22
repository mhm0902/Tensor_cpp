#ifndef __IPOOLING_LAYER_H__
#define __IPOOLING_LAYER_H__
#include "algo_common.h"


//Pooling����
typedef enum
{
	POOL_MAX = 0,
	POOL_AVE = 1,
	POOL_MAX_AVERAGE_BLEND = 2,
} POOL_TYPE_E;

class IPoolingLayer
{
public:
	IPoolingLayer(POOL_TYPE_E _eMode, DimsHW _stKernel, DimsHW _stStride, DimsHW _stPadding)
	{
		m_eMode = _eMode;
		m_stKernel = _stKernel;
		m_stStride = _stStride;
		m_stPadding = _stPadding;
	};
	~IPoolingLayer() {};

	int forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims _stOutPut);
private:
	POOL_TYPE_E m_eMode;

	DimsHW	m_stKernel;		//����˿��
	DimsHW	m_stStride;		//��ȷ��򲽳�
	DimsHW	m_stPadding;	//��ȷ������������
};

#endif //__IPOOLING_LAYER_H__

