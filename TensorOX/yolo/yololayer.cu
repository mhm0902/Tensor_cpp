#include "yololayer.h"
#include <algorithm>

__device__ float Logist(float data) { return 1.0f / (1.0f + expf(-data)); };

__global__ void CalDetection(const float *input, float *output, int noElements,
	const int netwidth, const int netheight, int maxoutobject, int yoloWidth, int yoloHeight, 
	const float anchors[CHECK_COUNT * 2], int classes, int outputElem)
{

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= noElements) return;

	int total_grid = yoloWidth * yoloHeight;
	int bnIdx = idx / total_grid;
	idx = idx - total_grid * bnIdx;
	int info_len_i = 5 + classes;
	const float* curInput = input + bnIdx * (info_len_i * total_grid * CHECK_COUNT);

	for (int k = 0; k < CHECK_COUNT; ++k) {
		float box_prob = Logist(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
		if (box_prob < IGNORE_THRESH) continue;
		int class_id = 0;
		float max_cls_prob = 0.0;
		for (int i = 5; i < info_len_i; ++i) {
			float p = Logist(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
			if (p > max_cls_prob) {
				max_cls_prob = p;
				class_id = i - 5;
			}
		}
		float *res_count = output + bnIdx * outputElem;
		int count = (int)atomicAdd(res_count, 1);
		if (count >= maxoutobject) return;
		char *data = (char*)res_count + sizeof(float) + count * sizeof(Detection);
		Detection *det = (Detection*)(data);

		int row = idx / yoloWidth;
		int col = idx % yoloWidth;

		//Location
		// pytorch:
		//  y = x[i].sigmoid()
		//  y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
		//  y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
		//  X: (sigmoid(tx) + cx)/FeaturemapW *  netwidth
		det->bbox[0] = (col - 0.5f + 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * netwidth / yoloWidth;
		det->bbox[1] = (row - 0.5f + 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * netheight / yoloHeight;

		// W: (Pw * e^tw) / FeaturemapW * netwidth
		// v5: https://github.com/ultralytics/yolov5/issues/471
		det->bbox[2] = 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]);
		det->bbox[2] = det->bbox[2] * det->bbox[2] * anchors[2 * k];
		det->bbox[3] = 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]);
		det->bbox[3] = det->bbox[3] * det->bbox[3] * anchors[2 * k + 1];
		det->conf = box_prob * max_cls_prob;
		det->class_id = class_id;
	}
}

std::vector<std::vector<float>> getAnchors(std::map<std::string, Weights>& weightMap, std::string lname) {
	std::vector<std::vector<float>> anchors;
	Weights wts = weightMap[lname + ".anchor_grid"];
	int anchor_len = CHECK_COUNT * 2;
	for (int i = 0; i < wts.count / anchor_len; i++) {
		auto *p = (const float*)wts.values + i * anchor_len;
		std::vector<float> anchor(p, p + anchor_len);
		anchors.push_back(anchor);
	}
	return anchors;
}


yoloyaler::yoloyaler(std::map<std::string, Weights>& weightMap, std::string lname)
{
	auto anchors = getAnchors(weightMap, lname);

	int scale = 8;
	mYoloKernel.clear();
	for (size_t i = 0; i < anchors.size(); i++) {
		YoloKernel kernel;
		kernel.width = INPUT_W / scale;
		kernel.height = INPUT_H / scale;
		memcpy(kernel.anchors, &anchors[i][0], anchors[i].size() * sizeof(float));
		mYoloKernel.push_back(kernel);
		scale *= 2;
	}

	mClassCount = 80;
	mYoloV5NetWidth = INPUT_W;
	mYoloV5NetHeight = INPUT_H;
	mMaxOutObject = MAX_OUTPUT_BBOX_COUNT;
	mKernelCount = mYoloKernel.size();

	cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*));
	size_t AnchorLen = sizeof(float)* CHECK_COUNT * 2;
	for (int ii = 0; ii < mKernelCount; ii++)
	{
		cudaMalloc(&mAnchor[ii], AnchorLen);
		const auto& yolo = mYoloKernel[ii];
		cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice);
	}
}

yoloyaler::~yoloyaler()
{
	for (int ii = 0; ii < mKernelCount; ii++)
	{
		cudaFree(&mAnchor[ii]);
	}
	cudaFree(mAnchor);
}

int yoloyaler::forward(const float* const* inputs, float *output, cudaStream_t stream, int batchSize)
{
	int outputElem = 1 + mMaxOutObject * sizeof(Detection) / sizeof(float);

	for (int idx = 0; idx < batchSize; ++idx) {
		cudaMemsetAsync(output + idx * outputElem, 0, sizeof(float), stream);
	}
	int numElem = 0;
	for (unsigned int i = 0; i < mYoloKernel.size(); ++i) {
		const auto& yolo = mYoloKernel[i];
		numElem = yolo.width * yolo.height * batchSize;
		if (numElem < mThreadCount) mThreadCount = numElem;

		//printf("Net: %d  %d \n", mYoloV5NetWidth, mYoloV5NetHeight);
		CalDetection << < (numElem + mThreadCount - 1) / mThreadCount, mThreadCount, 0, stream >> >
			(inputs[i], output, numElem, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, 
				yolo.width, yolo.height, (float*)mAnchor[i], mClassCount, outputElem);
	}
	return 0;
}

