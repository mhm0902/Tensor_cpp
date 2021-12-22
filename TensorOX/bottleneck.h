#ifndef _IBOTTLENNCKK_H__
#define _IBOTTLENNCKK_H__

#include "IConvolutionLayer.h"
#include"IConvolutionLayer_BN.h"
#include "IActivationLayer.h"
#include "IElementWiseLayer.h"
#include "convBlock.h"
class bottleneck
{
public:
	bottleneck(std::map<std::string, Weights>& weightMap, int inch, int c1, int c2, bool shortcut, int g, float e, std::string lname);
	~bottleneck();

	int forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut, void *_pBuffer);

	/////////////////固定输入输出加速方法////////////////////////////////////////////
	Dims init(Dims _stInPut);
	int forwardEx(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut, void *_pBuffer);
private:
	convBlock* conv1;
	convBlock* conv2;
	IElementWiseLayer* ew;
	//uint8_t* buffer1;
	//uint8_t* buffer2;
};

#endif //_IBOTTLENNCKK_H__

