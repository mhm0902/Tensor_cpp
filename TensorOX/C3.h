#ifndef _IC_3_H__
#define _IC_3_H__

#include "IConvolutionLayer.h"
#include"IConvolutionLayer_BN.h"
#include "IActivationLayer.h"
#include "IElementWiseLayer.h"
#include "convBlock.h"
#include "bottleneck.h"
#include "IConcatenationLayer.h"
#include <vector>
class C3
{
public:
	C3(std::map<std::string, Weights>& weightMap, int inch, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname);
	~C3();
	int forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut, void *_pBuffer);

	/////////////////固定输入输出加速方法////////////////////////////////////////////
	Dims init(Dims _stInPut);
	int forwardEx(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut, void *_pBuffer);
private:
	convBlock* cv1;
	convBlock* cv2;
	convBlock* cv3;
	IConcatenationLayer*cat;
	std::vector<bottleneck*> bot;

	//uint8_t* buffer1;
	//uint8_t* buffer2;
	//uint8_t* buffer3;
};

#endif //_IC_3_H__

