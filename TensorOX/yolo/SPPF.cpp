#include "SPPF.h"



SPPF::SPPF(std::map<std::string, Weights>& weightMap, int inch, int c1, int c2, int k, std::string lname)
{
	cv1 = cv2 = NULL;
	cat = NULL;
	pool = NULL;
	//buffer1 = buffer2 = buffer3 = buffer4 = buffer5 = buffer6 = buffer7 = NULL;
	int c_ = c1 / 2;
	cv1 = new convBlock(weightMap, inch, c_, 1, 1, 1, lname + ".cv1");

	pool = new IPoolingLayer(POOL_MAX, DimsHW{ k, k }, DimsHW{ 1, 1 }, DimsHW{ k / 2, k / 2 });
	//pool2 = new IPoolingLayer(POOL_MAX, DimsHW{ k, k }, DimsHW{ 1, 1 }, DimsHW{ k / 2, k / 2 });
	//pool3 = new IPoolingLayer(POOL_MAX, DimsHW{ k, k }, DimsHW{ 1, 1 }, DimsHW{ k / 2, k / 2 });

	cat = new IConcatenationLayer();
	cv2 = new convBlock(weightMap, c_ * 4, c2, 1, 1, 1, lname + ".cv2");

	//cudaMalloc((void**)&buffer1, 8 * INPUT_H * INPUT_W * sizeof(float));

	//cudaMalloc((void**)&buffer2, 8 * INPUT_H * INPUT_W * sizeof(float));

	//cudaMalloc((void**)&buffer3, 8 * INPUT_H * INPUT_W * sizeof(float));

	//cudaMalloc((void**)&buffer4, 8 * INPUT_H * INPUT_W * sizeof(float));

	//cudaMalloc((void**)&buffer5, 8 * INPUT_H * INPUT_W * sizeof(float));

	//cudaMalloc((void**)&buffer6, 8 * INPUT_H * INPUT_W * sizeof(float));

	//cudaMalloc((void**)&buffer7, 8 * INPUT_H * INPUT_W * sizeof(float));

}

SPPF::~SPPF()
{
	if (NULL != cv1)
	{
		delete cv1;
		cv1 = NULL;
	}
	if (NULL != cv2)
	{
		delete cv2;
		cv2 = NULL;
	}
	if (NULL != cat)
	{
		delete cat;
		cat = NULL;
	}
	if (NULL != pool)
	{
		delete pool;
		pool = NULL;
	}
	/*if (NULL != buffer1)
	{
	cudaFree(buffer1);
	buffer1 = NULL;
	}
	if (NULL != buffer2)
	{
	cudaFree(buffer2);
	buffer2 = NULL;
	}
	if (NULL != buffer3)
	{
	cudaFree(buffer3);
	buffer3 = NULL;
	}

	if (NULL != buffer4)
	{
	cudaFree(buffer4);
	buffer4 = NULL;
	}
	if (NULL != buffer5)
	{
	cudaFree(buffer5);
	buffer5 = NULL;
	}
	if (NULL != buffer6)
	{
	cudaFree(buffer6);
	buffer6 = NULL;
	}
	if (NULL != buffer7)
	{
	cudaFree(buffer7);
	buffer7 = NULL;
	}*/
}

int SPPF::forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut, void *_pBuffer)
{
	Dims stDimCv1, stDimCv2, stDimCv3, stDimCv4;
	void *buffer1 = _pBuffer;
	void *tmp_buf = (char*)_pBuffer + get_bolck_size(_stInPut) * sizeof(float);
	cv1->forward(_pInData, _stInPut, buffer1, stDimCv1, tmp_buf);
#if 0
	tmp_buf = (char*)buffer1 + get_bolck_size(stDimCv1) * sizeof(float);


	void *buffer2 = (char*)_pBuffer + get_bolck_size(stDimCv1) * sizeof(float);
	stDimCv2 = stDimCv1;
	pool->forward(buffer1, stDimCv1, buffer2, stDimCv2);

	stDimCv1.d[0] = 1;
	stDimCv1.d[1] = 1024;
	stDimCv1.d[2] = 20;
	stDimCv1.d[3] = 20;
	cv2->forward(buffer1, stDimCv1, _pOutData, _stOutPut, tmp_buf);
#else
	void *buffer2 = (char*)_pBuffer + get_bolck_size(stDimCv1) * sizeof(float);
	stDimCv2 = stDimCv1;
	pool->forward(buffer1, stDimCv1, buffer2, stDimCv2);

	void *buffer3 = (char*)buffer2 + get_bolck_size(stDimCv2) * sizeof(float);
	stDimCv3 = stDimCv2;
	pool->forward(buffer2, stDimCv2, buffer3, stDimCv3);

	void *buffer4 = (char*)buffer3 + get_bolck_size(stDimCv3) * sizeof(float);
	stDimCv4 = stDimCv2;
	pool->forward(buffer3, stDimCv3, buffer4, stDimCv4);

	Dims stDimCat1, stDimCat2, stDimCat3;
	void *buffer5 = (char*)buffer4 + get_bolck_size(stDimCv4) * sizeof(float);
	cat->forward(buffer1, stDimCv1, buffer2, stDimCv2, buffer5, stDimCat1);

	void *buffer6 = (char*)buffer5 + get_bolck_size(stDimCat1) * sizeof(float);
	cat->forward(buffer5, stDimCat1, buffer3, stDimCv3, buffer6, stDimCat2);

	void *buffer7 = (char*)buffer6 + get_bolck_size(stDimCat2) * sizeof(float);
	cat->forward(buffer6, stDimCat2, buffer4, stDimCv4, buffer7, stDimCat3);

	tmp_buf = (char*)buffer7 + get_bolck_size(stDimCat3) * sizeof(float);

	cv2->forward(buffer7, stDimCat3, _pOutData, _stOutPut, tmp_buf);
#endif
	return 0;
}