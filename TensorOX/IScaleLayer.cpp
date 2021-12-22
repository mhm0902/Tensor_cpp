#include "IScaleLayer.h"
#include"UScaleLayer.h"


IScaleLayer::IScaleLayer(ScaleMode mode, Weights shift, Weights scale, Weights powe)
{

}


IScaleLayer::~IScaleLayer()
{
}

int IScaleLayer::forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims _stOutPut)
{
	//template <typename Dtype> int IScaleLayer_forward_GPU(const Dtype* _pfIn, const Dtype* _pfScale, const Dtype* _pfBias,
	//	const int _iScaleDim, const int _iInnerDim, int _iTheads, Dtype* _pdOut);

	//int iStatus = IScaleLayer_forward_GPU(_pInData, );


	return 0;
}
