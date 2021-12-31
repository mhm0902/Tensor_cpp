#include "convBlock.h"



convBlock::convBlock(std::map<std::string, Weights>& weightMap, int inch, int outch, int ksize, int s, int g, std::string lname) {
	conv = NULL;
	silu = NULL;
	int p = ksize / 3;
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	conv = new IConvolutionLayer_BN(DimsNCHW{ outch, inch, ksize, ksize }, weightMap[lname + ".conv.weight"], emptywts,
		weightMap[lname + ".bn.weight"], weightMap[lname + ".bn.bias"], weightMap[lname + ".bn.running_mean"],
		weightMap[lname + ".bn.running_var"], 1e-3 /*FLT_EPSILON*/);

	//conv = new IConvolutionLayer(outch, DimsNCHW{ outch, inch, ksize, ksize }, weightMap[lname + ".conv.weight"], emptywts);

	conv->setStride(DimsHW{ s, s });
	conv->setPadding(DimsHW{ p, p });

	if(1 != g)
	{
		printf("-----g:%d\n", g);
	}

	silu = new IActivationLayer(Silu);

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
	//if (NULL != buffer)
	//{
	//	cudaFree(buffer);
	//	buffer = NULL;
	//}
}

int convBlock::forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut, void *_pBuffer)
{
	void *buffer = _pBuffer;

	//conv->forward(_pInData, _stInPut, buffer, _stOutPut);

	conv->forwardGMM(_pInData, _stInPut, buffer, _stOutPut, NULL, NULL);	

	silu->forward(buffer, _stOutPut, _pOutData);

	return 0;
};
