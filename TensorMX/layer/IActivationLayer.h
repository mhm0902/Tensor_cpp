#ifndef  _U_I_SILU_H__
#define _U_I_SILU_H__

#include "algo_common.h"

enum ActivateMode { Sigmoid, ReLU, Tanh, Silu, Invalid};

class IActivationLayer
{
public:
	IActivationLayer() {};
	~IActivationLayer() {};

	int forward(void* _pInData, Dims _stInPut, void* _pOutData, ActivateMode _eMode = Silu);
	//int forward_cudnn(void* _pInData, Dims _stInPut, void* _pOutData, ActivateMode _eMode = Silu);
};
#endif // ! _U_I_SILU_H__
