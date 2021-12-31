#ifndef __ISCALE_LAYER_H__
#define __ISCALE_LAYER_H__
#include "algo_common.h"

class IScaleLayer
{
public:
	IScaleLayer(ScaleMode mode, Weights shift, Weights scale, Weights powe);
	~IScaleLayer();

	int forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims _stOutPut);
};

#endif//__ISCALE_LAYER_H__

