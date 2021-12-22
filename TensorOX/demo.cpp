
#include<stdio.h>
#include<string.h>
#include "opencv2\opencv.hpp"
#include "preprocess.h"
#include "yolo_v5_6.h"

std::map<std::string, Weights> loadWeights(const std::string file) {
	std::cout << "Loading weights: " << file << std::endl;
	std::map<std::string, Weights> weightMap;

	// Open weights file
	std::ifstream input(file);
	assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

	// Read number of weight blobs
	int32_t count;
	input >> count;
	assert(count > 0 && "Invalid weight map file.");

	while (count--)
	{
		Weights wt{ DataType::kFLOAT, nullptr, 0 };
		uint32_t size;

		// Read name and type of blob
		std::string name;
		input >> name >> std::dec >> size;
		wt.type = DataType::kFLOAT;

		// Load blob
		uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
		for (uint32_t x = 0, y = size; x < y; ++x)
		{
			input >> std::hex >> val[x];
		}
		wt.values = val;

		wt.count = size;
		weightMap[name] = wt;
	}

	return weightMap;
}
int main()
{
#if 0
	IPoolingLayer *pooling = new IPoolingLayer(POOL_MAX, DimsHW{ 5, 5 }, DimsHW{ 1, 1 }, DimsHW{ 5 / 2, 5 / 2 });

	int size_image = 1 * 256 * 20 * 20 * sizeof(float);
	uint8_t* img_device = nullptr;
	uint8_t* buffer = nullptr;
	cudaMalloc((void**)&img_device, size_image);//
	cudaMalloc((void**)&buffer, size_image);

	Dims dim_in, dim_out;
	dim_in.nbDims = 4;
	dim_in.d[0] = 1;
	dim_in.d[1] = 256;
	dim_in.d[2] = 20;
	dim_in.d[3] = 20;
	dim_out = dim_in;
	pooling->forward(buffer, dim_in, img_device, dim_out);
#else


	float gd = 0.33;
	float gw = 0.50;
	std::map<std::string, Weights> weightMap = loadWeights("D:/Workspace/TensorRT/YoloV5/x64/Release/yolov5s.wts");

	yolo_v5_6 *pstYolo = new yolo_v5_6(weightMap, gd, gw);
	cv::Mat img = cv::imread("D:/20210111100406.jpg");
	int size_image = img.cols*img.rows * 3;
	// Create stream
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	uint8_t* img_device = nullptr;
	uint8_t* buffer = nullptr;
	cudaMalloc((void**)&img_device, size_image);//
	cudaMalloc((void**)&buffer, 8 * INPUT_H * INPUT_W * sizeof(float));

	cudaMemcpyAsync(img_device, img.data, size_image, cudaMemcpyHostToDevice, stream);
	preprocess_kernel_img(img_device, img.cols, img.rows, (float*)buffer, INPUT_W, INPUT_H, stream);

	Dims dim_out = DimsNCHW(1, 3, INPUT_H, INPUT_W);

	void * tmp_buf = nullptr;
	int iBuf_size = 32 * INPUT_H * INPUT_W * sizeof(float);
	cudaMalloc((void**)&tmp_buf, iBuf_size);//

	
	for (int i = 0; i < 1000; i++)
	{
		int64 iStart = cv::getTickCount();
		pstYolo->forward(buffer, DimsNCHW(1, 3, INPUT_H, INPUT_W), img_device, dim_out, tmp_buf);
		int64 iEnd = cv::getTickCount();

		printf("__---------%f\n", (iEnd - iStart) * 1000.0 / cv::getTickFrequency());
	}


	//upsample11->forward(img_device, dim_out1, );
	//assert(upsample11);
	//upsample11->setResizeMode(ResizeMode::kNEAREST);
	//upsample11->setOutputDimensions(bottleneck_csp6->getOutput(0)->getDimensions());
#endif // 1
	return 0;
}