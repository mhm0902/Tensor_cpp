#ifndef _ICONV_BLOCK_H__
#define _ICONV_BLOCK_H__

#include "IConvolutionLayer.h"
#include"IConvolutionLayer_BN.h"
#include "IActivationLayer.h"
#include "IScaleLayer.h"

class convBlock
{
public:
	convBlock(std::map<std::string, Weights>& weightMap, int inch, int outch, int ksize, int s, int g, std::string lname);
	~convBlock();

	int forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut, void *_pBuffer);

private:
	//IConvolutionLayer_BN* conv;
	IConvolutionLayer* conv;
	IScaleLayer* bn1;
	IActivationLayer* silu;

	void *buffer;

};
#endif // !_ICONV_BLOCK_H__



