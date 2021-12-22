#ifndef _ISPPF_H__
#define _ISPPF_H__

#include "IConvolutionLayer.h"
#include"IConvolutionLayer_BN.h"
#include "IActivationLayer.h"
#include "IElementWiseLayer.h"
#include "convBlock.h"
#include "bottleneck.h"
#include "IConcatenationLayer.h"
#include "C3.h"
#include "IPoolingLayer.h"
class SPPF
{
public:
	SPPF(std::map<std::string, Weights>& weightMap, int inch, int c1, int c2, int k, std::string lname);
	~SPPF();
	int forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut, void *_pBuffer);

	/////////////////固定输入输出加速方法////////////////////////////////////////////
	Dims init(Dims _stInPut);
	int forwardEx(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut, void *_pBuffer);
private:
	convBlock *cv1;
	convBlock *cv2;
	IConcatenationLayer *cat;
	IPoolingLayer *pool;
	/*	IPoolingLayer *pool2;
	IPoolingLayer *pool3;*/
	//uint8_t* buffer1;
	//uint8_t* buffer2;
	//uint8_t* buffer3;
	//uint8_t* buffer4;

	//uint8_t* buffer5;
	//uint8_t* buffer6;
	//uint8_t* buffer7;
};
#endif //_ISPPF_H__

