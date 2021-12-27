
#include<stdio.h>
#include<string.h>
#include "opencv2\opencv.hpp"
#include "common/preprocess.h"
#include "yolo/yolo_v5_6.h"

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

float iou(float lbox[4], float rbox[4]) {
	float interBox[] = {
		(std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
		(std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
		(std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
		(std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
	};

	if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
		return 0.0f;

	float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
	return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

bool cmp(const Detection& a, const Detection& b) {
	return a.conf > b.conf;
}
void nms(std::vector<Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5) {
	int det_size = sizeof(Detection) / sizeof(float);
	std::map<float, std::vector<Detection>> m;
	for (int i = 0; i < output[0] && i < MAX_OUTPUT_BBOX_COUNT; i++) {
		if (output[1 + det_size * i + 4] <= conf_thresh) continue;
		Detection det;
		memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
		if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
		m[det.class_id].push_back(det);
	}
	for (auto it = m.begin(); it != m.end(); it++) {
		//std::cout << it->second[0].class_id << " --- " << std::endl;
		auto& dets = it->second;
		std::sort(dets.begin(), dets.end(), cmp);
		for (size_t m = 0; m < dets.size(); ++m) {
			auto& item = dets[m];
			res.push_back(item);
			for (size_t n = m + 1; n < dets.size(); ++n) {
				if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
					dets.erase(dets.begin() + n);
					--n;
				}
			}
		}
	}
}
cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
	float l, r, t, b;
	float r_w = INPUT_W / (img.cols * 1.0);
	float r_h = INPUT_H / (img.rows * 1.0);
	if (r_h > r_w) {
		l = bbox[0] - bbox[2] / 2.f;
		r = bbox[0] + bbox[2] / 2.f;
		t = bbox[1] - bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
		b = bbox[1] + bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
		l = l / r_w;
		r = r / r_w;
		t = t / r_w;
		b = b / r_w;
	}
	else {
		l = bbox[0] - bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
		r = bbox[0] + bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
		t = bbox[1] - bbox[3] / 2.f;
		b = bbox[1] + bbox[3] / 2.f;
		l = l / r_h;
		r = r / r_h;
		t = t / r_h;
		b = b / r_h;
}
	return cv::Rect(round(l), round(t), round(r - l), round(b - t));
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
	preprocess_kernel_img<float>(img_device, img.cols, img.rows, (float*)buffer, INPUT_W, INPUT_H, stream);

	Dims dim_out = DimsNCHW(1, 3, INPUT_H, INPUT_W);

	void * tmp_buf = nullptr;
	int iBuf_size = 32 * 640 * 640 * sizeof(float)*3;
	cudaMalloc((void**)&tmp_buf, iBuf_size);//


	float prob[ 6001];
	for (int i = 0; i < 1000; i++)
	{
		int64 iStart = cv::getTickCount();
		pstYolo->forward(buffer, DimsNCHW(1, 3, INPUT_H, INPUT_W), img_device, dim_out, tmp_buf, stream);
		int64 iEnd = cv::getTickCount();

		cudaMemcpyAsync(prob, img_device, 6001 * sizeof(float), cudaMemcpyDeviceToHost, stream);

		std::vector<Detection>res;

		nms(res, prob, 0.5, 0.4);

		for (size_t j = 0; j < res.size(); j++) {
			cv::Rect r = get_rect(img, res[j].bbox);
			cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
			cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
		}
		cv::imwrite("_res.jpg", img);

		printf("__---------%f\n", (iEnd - iStart) * 1000.0 / cv::getTickFrequency());
	}

#endif // 1
	return 0;
}