#ifndef _DECODE_H__
#define _DECODE_H__

#include "IConvolutionLayer.h"
#include"IConvolutionLayer_BN.h"
#include "IActivationLayer.h"
#include "IElementWiseLayer.h"
#include "convBlock.h"
#include "bottleneck.h"
#include "IConcatenationLayer.h"
#include <vector>

#define CHECK_COUNT 3

struct YoloKernel
{
	int width;
	int height;
	float anchors[CHECK_COUNT * 2];
};

static constexpr int LOCATIONS = 4;
struct alignas(float)Detection {
	//center_x center_y w h
	float bbox[LOCATIONS];
	float conf;  // bbox_conf * cls_conf
	float class_id;
};

class yolo_decode
{
public:
	yolo_decode(std::map<std::string, Weights>& weightMap, std::string lname, std::vector<IConvolutionLayer*> dets);
	~yolo_decode();
private:
	std::vector<YoloKernel> m_kernels;
};


#endif //_DECODE_H__
