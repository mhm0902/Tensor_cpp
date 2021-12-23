#ifndef __U_ELEMENT_LAYER_H__
#define __U_ELEMENT_LAYER_H__
#include "algo_common.h"

//Eltwise¿‡–Õ
typedef enum
{
	CNN_ELTWISE_PROD = 0,
	CNN_ELTWISE_SUM = 1,
	CNN_ELTWISE_MAX = 2,
} ELTWISE_OPT_TYPE_E;

class IElementWiseLayer
{
public:
	IElementWiseLayer() {};
	~IElementWiseLayer() {};

	int forward(void* _pInA, void* _pInB, Dims _stInPut, ELTWISE_OPT_TYPE_E _eOptTyoe, void* _pOutY);
};
#endif //__U_ELEMENT_LAYER_H__
