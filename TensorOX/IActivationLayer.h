#ifndef  _U_I_SILU_H__
#define _U_I_SILU_H__

#include "algo_common.h"

enum ActivateMode { Sigmoid, ReLU, Tanh, Silu };

class IActivationLayer
{
public:
	IActivationLayer(ActivateMode _mode) {
		m_eMode = _mode;
	};
	~IActivationLayer() {};

	int forward(void* _pInData, Dims _stInPut, void* _pOutData);
private:
	ActivateMode m_eMode;

};
#endif // ! _U_I_SILU_H__
