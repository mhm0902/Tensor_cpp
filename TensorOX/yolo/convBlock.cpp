#include "convBlock.h"



convBlock::convBlock(std::map<std::string, Weights>& weightMap, int inch, int outch, int ksize, int s, int g, std::string lname) {
	conv = NULL;
	silu = NULL;
	int p = ksize / 3;
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	conv = new IConvolutionLayer_BN(DimsNCHW{ outch, inch, ksize, ksize }, weightMap[lname + ".conv.weight"], emptywts,
		weightMap[lname + ".bn.weight"], weightMap[lname + ".bn.bias"], weightMap[lname + ".bn.running_mean"],
		weightMap[lname + ".bn.running_var"], 1e-3);

	//conv = new IConvolutionLayer(32, DimsNCHW{ outch, inch, ksize, ksize }, weightMap[lname + ".conv.weight"], emptywts);

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
	float *prob1;
	float *prob2;
	//if (_stInPut.d[1] == 64) {
	//	cudaStream_t stream;
	//	cudaStreamCreate(&stream);

	//	prob1 = (float*)malloc(1638400 * 4);
	//	cudaMemcpyAsync(prob1, _pInData, 1638400 * 4, cudaMemcpyDeviceToHost, stream);

	//	printf("-%f\n", prob1[0]);
	//}
	conv->forward(_pInData, _stInPut, buffer, _stOutPut);

	//if (_stInPut.d[1] == 64) {
	//	cudaStream_t stream;
	//	cudaStreamCreate(&stream);

	//	prob2 = (float*)malloc(1638400 * 4);
	//	cudaMemcpyAsync(prob2, buffer, 1638400 * 4, cudaMemcpyDeviceToHost, stream);

	//	printf("-%f\n", prob2[0]);
	//}

	silu->forward(buffer, _stOutPut, _pOutData);

	return 0;
};
