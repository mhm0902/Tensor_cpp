#ifndef _YOLO_V5_6_H__
#define _YOLO_V5_6_H__

#include "IConvolutionLayer.h"
#include"IConvolutionLayer_BN.h"
#include "IActivationLayer.h"
#include "IElementWiseLayer.h"
#include "convBlock.h"
#include "bottleneck.h"
#include "IConcatenationLayer.h"
#include "C3.h"
#include "IPoolingLayer.h"
#include "IUpsampleLayer.h"
#include "SPPF.h"



class yolo_v5_6
{
public:
	yolo_v5_6(std::map<std::string, Weights> weightMap, float _gd, float _gw);

	~yolo_v5_6();

	int forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut, void *_pBuffer);
private:
	convBlock* conv0;
	convBlock* conv1;
	C3* bottleneck_CSP2;
	convBlock* conv3;
	C3* bottleneck_csp4;
	convBlock* conv5;
	C3* bottleneck_csp6;
	convBlock* conv7;
	C3* bottleneck_csp8;
	SPPF* spp9;
	convBlock* conv10;
	IUpsampleLayer* upsample11;
	IConcatenationLayer *cat12;
	C3* bottleneck_csp13;
	convBlock* conv14;
	IUpsampleLayer*upsample15;
	IConcatenationLayer *cat16;
	C3* bottleneck_csp17;
	IConvolutionLayer* det0;
	convBlock* conv18;
	IConcatenationLayer *cat19;
	C3* bottleneck_csp20;
	IConvolutionLayer* det1;
	convBlock* conv21;
	IConcatenationLayer *cat22;
	C3* bottleneck_csp23;
	IConvolutionLayer* det2;
};

#endif //_YOLO_V5_6_H__

