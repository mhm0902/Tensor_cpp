#ifndef _CONCATENATIONG_LAYER_H__
#define _CONCATENATIONG_LAYER_H__


#include "algo_common.h"
class IConcatenationLayer
{
public:
	IConcatenationLayer();
	~IConcatenationLayer();
	int forward(void* _pInA, Dims _stInA, void* _pInB, Dims _stInB, void* _pOutY, Dims &_stOutY);
private:

};

#endif // !_CONCATENATIONG_LAYER_H__