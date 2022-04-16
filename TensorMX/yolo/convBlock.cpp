#include "convBlock.h"

#include <iostream>
#include<iomanip>


IScaleLayer* addBatchNorm2d(std::map<std::string, Weights>& weightMap, std::string lname, float eps) {
	float *gamma = (float*)weightMap[lname + ".weight"].values;
	float *beta = (float*)weightMap[lname + ".bias"].values;
	float *mean = (float*)weightMap[lname + ".running_mean"].values;
	float *var = (float*)weightMap[lname + ".running_var"].values;
	int len = weightMap[lname + ".running_var"].count;

	float *scval = (float*)(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		scval[i] = gamma[i] / sqrt(var[i] + eps);
	}
	Weights scale{ DataType::kFLOAT, scval, len };

	float *shval = (float*)(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
	}
	Weights shift{ DataType::kFLOAT, shval, len };

	float *pval = (float*)(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		pval[i] = 1.0;
	}
	Weights power{ DataType::kFLOAT, pval, len };

	IScaleLayer* scale_1 = new IScaleLayer( ScaleMode::kCHANNEL, shift, scale, power);

	free(scval);
	free(shval);
	free(pval);

	assert(scale_1);
	return scale_1;
}

convBlock::convBlock(std::map<std::string, Weights>& weightMap, int inch, int outch, int ksize, int s, int g, std::string lname) {
	conv = NULL;
	silu = NULL;
	bn1 = NULL;
	buffer = NULL;
	int p = ksize / 3;
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	//conv = new IConvolutionLayer_BN(DimsNCHW{ outch, inch, ksize, ksize }, weightMap[lname + ".conv.weight"], emptywts,
	//	weightMap[lname + ".bn.weight"], weightMap[lname + ".bn.bias"], weightMap[lname + ".bn.running_mean"],
	//	weightMap[lname + ".bn.running_var"], 1e-3 /*FLT_EPSILON*/);

	conv = new IConvolutionLayer(DimsNCHW{ outch, inch, ksize, ksize }, weightMap[lname + ".conv.weight"], emptywts,
		weightMap[lname + ".bn.weight"], weightMap[lname + ".bn.bias"], weightMap[lname + ".bn.running_mean"],
		weightMap[lname + ".bn.running_var"], 1e-3 /*FLT_EPSILON*/);

	conv->setStride(DimsHW{ s, s });
	conv->setPadding(DimsHW{ p, p });

	conv->setActivateMode(Silu);

	if(1 != g)
	{
		printf("-----g:%d\n", g);
	}

	//bn1 = addBatchNorm2d( weightMap, lname + ".bn", 1e-3);

	//silu = new IActivationLayer();

	//cudaMalloc((void**)&buffer, 8 * INPUT_H * INPUT_W * sizeof(float));
};

convBlock::~convBlock()
{
	if (NULL != silu)
	{
		delete silu;
		silu = NULL;
	}
	if (NULL != conv)
	{
		delete conv;
		conv = NULL;
	}
	if (NULL != bn1)
	{
		delete bn1;
		bn1 = NULL;
	}
	if (NULL != buffer)
	{
		cudaFree(buffer);
		buffer = NULL;
	}
}

int convBlock::forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut, void *_pBuffer)
{
	void *buffer = _pBuffer;
	Dims stDim1;
	conv->forwardGMM(_pInData, _stInPut, _pOutData, _stOutPut);

	//silu->forward(_pBuffer, _stOutPut, _pOutData);

	return 0;
};
