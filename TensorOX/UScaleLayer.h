#ifndef __U_SCALE_LAYER_H__
#define __U_SCALE_LAYER_H__

//#ifdef __cplusplus
//extern "C"
//{
//#endif


template <typename Dtype> int IScaleLayer_forward_GPU(const Dtype* _pfIn, const Dtype* _pfScale, const Dtype* _pfBias,
	const int _iScaleDim, const int _iInnerDim, int _iTheads, Dtype* _pdOut);




//#ifdef __cplusplus
//}
//#endif

#endif//__U_SCALE_LAYER_H__
