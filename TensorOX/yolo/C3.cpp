#include "C3.h"
#include<iomanip>


C3::C3(std::map<std::string, Weights>& weightMap, int inch, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname)
{
	cv1 = cv2 = NULL;
	cv3 = NULL;
	cat = NULL;
	//buffer1 = buffer2 = buffer3 = NULL;
	bot.clear();
	int c_ = (int)((float)c2 * e);
	cv1 = new convBlock(weightMap, inch, c_, 1, 1, 1, lname + ".cv1");
	cv2 = new convBlock(weightMap, inch, c_, 1, 1, 1, lname + ".cv2");


	for (int i = 0; i < n; i++)
	{
		bot.push_back(new bottleneck(weightMap, c_, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i)));
	}

	cat = new IConcatenationLayer();

	cv3 = new convBlock(weightMap, c_*2, c2, 1, 1, 1, lname + ".cv3");

	//Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

	//cv3 = new IConvolutionLayer(64, DimsNCHW{ c_ * 2, c2, 1, 1 }, weightMap[lname + ".cv3.conv.weight"], emptywts);

	//cudaMalloc((void**)&buffer1, 8 * INPUT_H * INPUT_W * sizeof(float));

	//cudaMalloc((void**)&buffer2, 8 * INPUT_H * INPUT_W * sizeof(float));

	//cudaMalloc((void**)&buffer3, 8 * INPUT_H * INPUT_W * sizeof(float));

}


C3::~C3()
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
	if (NULL != cv3)
	{
		delete cv3;
		cv3 = NULL;
	}
	if (NULL != cat)
	{
		delete cat;
		cat = NULL;
	}
	for (size_t i = 0; i < bot.size(); i++)
	{
		if (NULL != bot[i])
		{
			delete bot[i];
			bot[i] = NULL;
		}
	}
	bot.clear();

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
	//if (NULL != buffer3)
	//{
	//	cudaFree(buffer3);
	//	buffer3 = NULL;
	//}
}
int C3::forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut, void *_pBuffer)
{
	Dims stDimCv1, stDimCv2, stDimCv3;
	void* buffer1 = _pBuffer;
	void *tmp_buf = (char*)buffer1 + get_bolck_size(_stInPut) * sizeof(float);

	cv1->forward(_pInData, _stInPut, buffer1, stDimCv1, tmp_buf);

	size_t iOffset = get_bolck_size(stDimCv1) * sizeof(float);

	void* buffer2 = (char*)_pBuffer + iOffset;

	tmp_buf = (char*)buffer2 + iOffset;

	void *y1 = buffer1;
	void *y2 = buffer2;
	for (size_t i = 0; i < bot.size(); i++)
	{
		bot[i]->forward(y1, stDimCv1, y2, stDimCv2, tmp_buf);

		void *tmp = y1;
		y1 = y2;
		y2 = tmp;
		stDimCv1 = stDimCv2;
	}

	cv2->forward(_pInData, _stInPut, y2, stDimCv2, tmp_buf);

	void *buffer3 = (char*)buffer2 + get_bolck_size(stDimCv1) * sizeof(float);

	cat->forward(y1, stDimCv1, y2, stDimCv2, buffer3, stDimCv3);

	cv3->forward(buffer3, stDimCv3, _pOutData, _stOutPut, tmp_buf);

	//cv3->forward(buffer3, stDimCv3, _pOutData, _stOutPut);


	return 0;
};
