#include "decode.h"


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


yolo_decode::yolo_decode(std::map<std::string, Weights>& weightMap, std::string lname, 
	std::vector<IConvolutionLayer*> dets)
{
	auto anchors = getAnchors(weightMap, lname);

	int scale = 8;
	m_kernels.clear();
	for (size_t i = 0; i < anchors.size(); i++) {
		YoloKernel kernel;
		kernel.width = INPUT_W / scale;
		kernel.height = INPUT_H / scale;
		memcpy(kernel.anchors, &anchors[i][0], anchors[i].size() * sizeof(float));
		m_kernels.push_back(kernel);
		scale *= 2;
	}
}
yolo_decode::~yolo_decode()
{

}