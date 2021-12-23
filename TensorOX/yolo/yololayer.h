#ifndef _YOLO_LAYER_H__
#define _YOLO_LAYER_H__
#include "algo_common.h"


static constexpr int CHECK_COUNT = 3;
static constexpr float IGNORE_THRESH = 0.1f;
static constexpr int LOCATIONS = 4;
static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;

struct alignas(float)Detection {
	//center_x center_y w h
	float bbox[LOCATIONS];
	float conf;  // bbox_conf * cls_conf
	float class_id;
};
struct YoloKernel
{
	int width;
	int height;
	float anchors[CHECK_COUNT * 2];
};

class yoloyaler {
public:
	yoloyaler(std::map<std::string, Weights>& weightMap, std::string lname);
	
	~yoloyaler();

	int forward(const float* const* inputs, float *output, cudaStream_t stream, int batchSize = 1);

private:
	int mThreadCount = 256;
	int mKernelCount;
	int mClassCount;
	int mYoloV5NetWidth;
	int mYoloV5NetHeight;
	int mMaxOutObject;
	std::vector<YoloKernel> mYoloKernel;
	void** mAnchor;
};


#endif //_YOLO_LAYER_H__
