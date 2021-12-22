#ifndef __IUPSAMPLE_LAYER_H__
#define __IUPSAMPLE_LAYER_H__

#include "algo_common.h"

enum ResizeMode { kNEAREST, kLINEAR };

class IUpsampleLayer
{
public:
	IUpsampleLayer(ResizeMode _mode) {
		m_eMode = _mode;
	};
	~IUpsampleLayer() {};

	int forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims _stOutPut );
private:
	ResizeMode m_eMode;
};



#endif//__IUPSAMPLE_LAYER_H__
