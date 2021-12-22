#include "bottleneck.h"



bottleneck::bottleneck(std::map<std::string, Weights>& weightMap, int inch, int c1, int c2, bool shortcut, int g, float e, std::string lname)
{
	conv1 = NULL;
	conv2 = NULL;
	ew = NULL;
	//buffer1 = buffer2 = NULL;
	int ch = (int)((float)c2 * e);
	conv1 = new convBlock(weightMap, inch, ch, 1, 1, 1, lname + ".cv1");
	conv2 = new convBlock(weightMap, ch, c2, 3, 1, g, lname + ".cv2");

	//cudaMalloc((void**)&buffer1, 8 * INPUT_H * INPUT_W * sizeof(float));
	if (shortcut && c1 == c2)
	{
		//cudaMalloc((void**)&buffer2, 8 * INPUT_H * INPUT_W * sizeof(float));
		ew = new IElementWiseLayer();
	}
}

bottleneck::~bottleneck()
{
	if (NULL != conv1)
	{
		delete conv1;
		conv1 = NULL;
	}
	if (NULL != conv2)
	{
		delete conv2;
		conv2 = NULL;
	}
	if (NULL != ew)
	{
		delete ew;
		ew = NULL;
	}
	//if (NULL != buffer1)
	//{
	//	cudaFree(buffer1);
	//	buffer1 = NULL;
	//}
	//if (NULL != buffer2)
	//{
	//	cudaFree(buffer2);
	//	buffer2 = NULL;
	//}
}

int bottleneck::forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut, void *_pBuffer)
{
	if (ew)
	{
		Dims cv1_o_dim;
		void* buffer1 = _pBuffer;
		void *tmp_buf = (char*)_pBuffer + get_bolck_size(_stInPut) * sizeof(float);
		conv1->forward(_pInData, _stInPut, buffer1, cv1_o_dim, tmp_buf);
		void* buffer2 = (char*)_pBuffer + get_bolck_size(cv1_o_dim) * sizeof(float);

		tmp_buf = (char*)buffer2 + get_bolck_size(cv1_o_dim) * sizeof(float);
		conv2->forward(buffer1, cv1_o_dim, buffer2, _stOutPut, tmp_buf);
		ew->forward(_pInData, buffer2, _stInPut, CNN_ELTWISE_SUM, _pOutData);
	}
	else
	{
		Dims cv1_o_dim;
		void* buffer1 = _pBuffer;
		void *tmp_buf = (char*)_pBuffer + get_bolck_size(_stInPut) * sizeof(float);
		conv1->forward(_pInData, _stInPut, buffer1, cv1_o_dim, tmp_buf);
		conv2->forward(buffer1, cv1_o_dim, _pOutData, _stOutPut, tmp_buf);
	}
	return 0;
};

Dims bottleneck::init(Dims _stInPut)
{
	Dims dim_out = conv1->init(_stInPut);
	dim_out = conv2->init(dim_out);

	return dim_out;
}

int bottleneck::forwardEx(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut, void *_pBuffer)
{
	if (ew)
	{
		Dims cv1_o_dim;
		void* buffer1 = _pBuffer;
		void *tmp_buf = (char*)_pBuffer + get_bolck_size(_stInPut) * sizeof(float);
		conv1->forwardEx(_pInData, _stInPut, buffer1, cv1_o_dim, tmp_buf);
		void* buffer2 = (char*)_pBuffer + get_bolck_size(cv1_o_dim) * sizeof(float);

		tmp_buf = (char*)buffer2 + get_bolck_size(cv1_o_dim) * sizeof(float);
		conv2->forwardEx(buffer1, cv1_o_dim, buffer2, _stOutPut, tmp_buf);
		ew->forward(_pInData, buffer2, _stInPut, CNN_ELTWISE_SUM, _pOutData);
	}
	else
	{
		Dims cv1_o_dim;
		void* buffer1 = _pBuffer;
		void *tmp_buf = (char*)_pBuffer + get_bolck_size(_stInPut) * sizeof(float);
		conv1->forwardEx(_pInData, _stInPut, buffer1, cv1_o_dim, tmp_buf);
		conv2->forwardEx(buffer1, cv1_o_dim, _pOutData, _stOutPut, tmp_buf);
	}
	return 0;
}

