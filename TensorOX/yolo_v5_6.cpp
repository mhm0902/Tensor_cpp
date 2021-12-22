#include "yolo_v5_6.h"
#include<stdio.h>
#include<string.h>
#include <stdlib.h>
#include <algorithm>

int get_width(int x, float gw, int divisor = 8) {
	return int(ceil((x * gw) / divisor)) * divisor;
}

int get_depth(int x, float gd) {
	if (x == 1) return 1;
	int r = round(x * gd);
	if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
		--r;
	}
	return std::max<int>(r, 1);
}

yolo_v5_6::yolo_v5_6(std::map<std::string, Weights> weightMap, float _gd, float _gw)
{
	conv0 = conv1 = conv3 = conv5 = conv7 = conv10 = conv14 = conv18 = conv21 = NULL;
	bottleneck_CSP2 = bottleneck_csp4 = bottleneck_csp6 = bottleneck_csp8 = bottleneck_csp13
		= bottleneck_csp17 = bottleneck_csp20 = bottleneck_csp23 = NULL;

	spp9 = NULL;
	upsample11 = upsample15 = NULL;
	cat12 = cat16 = cat19 = cat22 = NULL;
	det0 = det1 = det2;

	float gw = _gw;
	float gd = _gd;
	int conv0_inch = 3;
	int conv0_outch = get_width(64, gw);
	conv0 = new convBlock(weightMap, conv0_inch, conv0_outch, 6, 2, 1, "model.0");

	Dims dim_cv0 = conv0->init(DimsNCHW(1, 3, INPUT_H, INPUT_W));

	int conv1_inch = conv0_outch;
	int conv1_outch = get_width(128, gw);
	conv1 = new convBlock(weightMap, conv1_inch, conv1_outch, 3, 2, 1, "model.1");
	Dims dim_cv1 = conv1->init(dim_cv0);

	int csp2_inch = conv1_outch;
	int csp2_c1 = get_width(128, gw);
	int csp2_c2 = get_width(128, gw);
	int csp2_n = get_depth(3, gd);
	bottleneck_CSP2 = new C3(weightMap, csp2_inch, csp2_c1, csp2_c2, csp2_n, true, 1, 0.5, "model.2");
	Dims dim_csp2 = bottleneck_CSP2->init(dim_cv1);

	int conv3_inch = csp2_c2;
	int conv3_outch = get_width(256, gw);
	conv3 = new convBlock(weightMap, conv3_inch, conv3_outch, 3, 2, 1, "model.3");
	Dims dim_cv3 = conv3->init(dim_csp2);

	int csp4_inch = conv3_outch;
	int csp4_c1 = get_width(256, gw);
	int csp4_c2 = get_width(256, gw);
	int csp4_n = get_depth(6, gd);
	bottleneck_csp4 = new C3(weightMap, csp4_inch, csp4_c1, csp4_c2, csp4_n, true, 1, 0.5, "model.4");
	Dims dim_csp4 = bottleneck_csp4->init(dim_cv3);

	int conv5_inch = csp4_c2;
	int conv5_outch = get_width(512, gw);
	conv5 = new convBlock(weightMap, conv5_inch, conv5_outch, 3, 2, 1, "model.5");
	Dims dim_cv5 = conv5->init(dim_csp4);

	int csp6_inch = conv5_outch;
	int csp6_c1 = get_width(512, gw);
	int csp6_c2 = get_width(512, gw);
	int csp6_n = get_depth(9, gd);
	bottleneck_csp6 = new C3(weightMap, csp6_inch, csp6_c1, csp6_c2, csp6_n, true, 1, 0.5, "model.6");
	Dims dim_csp6 = bottleneck_csp6->init(dim_cv5);

	int conv7_inch = csp6_c2;
	int conv7_outch = get_width(1024, gw);
	conv7 = new convBlock(weightMap, conv7_inch, conv7_outch, 3, 2, 1, "model.7");
	Dims dim_cv7 = conv7->init(dim_csp6);

	int csp8_inch = conv7_outch;
	int csp8_c1 = get_width(1024, gw);
	int csp8_c2 = get_width(1024, gw);
	int csp8_n = get_depth(3, gd);
	bottleneck_csp8 = new C3(weightMap, csp8_inch, csp8_c1, csp8_c2, csp8_n, true, 1, 0.5, "model.8");
	Dims dim_csp8 = bottleneck_csp8->init(dim_cv7);

	int spp9_inch = csp8_c2;
	int spp9_c1 = get_width(1024, gw);
	int spp9_c2 = get_width(1024, gw);
	spp9 = new SPPF(weightMap, spp9_inch, spp9_c1, spp9_c2, 5, "model.9");
	Dims dim_spp9 = spp9->init(dim_csp8);
	/* ------ yolov5 head ------ */

	int conv10_inch = spp9_c2;
	int conv10_outch = get_width(512, gw);
	conv10 = new convBlock(weightMap, conv10_inch, conv10_outch, 1, 1, 1, "model.10");
	Dims dim_cv10 = conv10->init(dim_spp9);

	upsample11 = new IUpsampleLayer(kNEAREST);//Êä³ö csp6_c2

											  //ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
	cat12 = new IConcatenationLayer();//Êä³ö csp6_c2*2

	int csp13_inch = csp6_c2 * 2;
	int csp13_c1 = get_width(1024, gw);
	int csp13_c2 = get_width(512, gw);
	int csp13_n = get_depth(3, gd);
	bottleneck_csp13 = new C3(weightMap, csp13_inch, csp13_c1, csp13_c2, csp13_n, false, 1, 0.5, "model.13");
	Dims dim_csp13 = bottleneck_csp13->init(DimsNCHW{ dim_csp6.d[0], csp13_inch, dim_csp6.d[2], dim_csp6.d[3] });

	int conv14_inch = csp13_c2;
	int conv14_outch = get_width(256, gw);
	conv14 = new convBlock(weightMap, conv14_inch, conv14_outch, 1, 1, 1, "model.14");
	Dims dim_cv14 = conv14->init(dim_csp13);

	upsample15 = new IUpsampleLayer(kNEAREST);//Êä³ö csp4_c2
	assert(upsample15);

	//ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
	cat16 = new IConcatenationLayer();//Êä³ö csp4_c2*2

	int csp17_inch = csp4_c2 * 2;
	int csp17_c1 = get_width(512, gw);
	int csp17_c2 = get_width(256, gw);
	int csp17_n = get_depth(3, gd);
	bottleneck_csp17 = new C3(weightMap, csp17_inch, csp17_c1, csp17_c2, csp17_n, false, 1, 0.5, "model.17");
	Dims dim_csp17 = bottleneck_csp17->init(DimsNCHW{ dim_csp4.d[0], csp17_inch, dim_csp4.d[2], dim_csp4.d[3] });

	/* ------ detect ------ */
	int det0_inch = csp17_c2;
	int det0_outch = 3 * (CLASS_NUM + 5);
	det0 = new IConvolutionLayer(det0_outch, DimsNCHW{ det0_outch, det0_inch, 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
	Dims dim_det0 = det0->init(dim_csp17);

	int conv18_outch = get_width(256, gw);
	conv18 = new convBlock(weightMap, csp17_c2, conv18_outch, 3, 2, 1, "model.18");
	Dims dim_cv18 = conv18->init(dim_csp17);

	//ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
	cat19 = new IConcatenationLayer();//Êä³ö conv18_outch + conv14_outch

	int csp20_inch = conv18_outch + conv14_outch;
	int csp20_c1 = get_width(512, gw);
	int csp20_c2 = get_width(512, gw);
	int csp20_n = get_depth(3, gd);
	bottleneck_csp20 = new C3(weightMap, csp20_inch, csp20_c1, csp20_c2, csp20_n, false, 1, 0.5, "model.20");
	Dims dim_csp20 = bottleneck_csp20->init(DimsNCHW{ dim_cv18.d[0], csp20_inch, dim_cv18.d[2], dim_cv18.d[3] });

	int det1_inch = csp20_c2;
	int det1_outch = 3 * (CLASS_NUM + 5);
	det1 = new IConvolutionLayer(det1_outch, DimsNCHW{ det1_outch, det1_inch, 1, 1 }, 
		weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
	Dims dim_det1 = det1->init(dim_csp20);

	int conv21_inch = csp20_c2;
	int conv21_outch = get_width(512, gw);
	conv21 = new convBlock(weightMap, conv21_inch, conv21_outch, 3, 2, 1, "model.21");
	Dims dim_cv21 = conv21->init(dim_csp20);

	//ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
	cat22 = new IConcatenationLayer();//Êä³ö conv21_outch + conv10_outch

	int csp23_inch = conv21_outch + conv10_outch;
	int csp23_c1 = get_width(1024, gw);
	int csp23_c2 = get_width(1024, gw);
	int csp23_n = get_depth(3, gd);
	bottleneck_csp23 = new C3(weightMap, csp23_inch, csp23_c1, csp23_c2, csp23_n, false, 1, 0.5, "model.23");
	Dims dim_csp23 = bottleneck_csp23->init(DimsNCHW{ dim_cv21.d[0], csp23_inch, dim_cv21.d[2], dim_cv21.d[3] });

	int det2_inch = csp23_c2;
	int det2_outch = 3 * (CLASS_NUM + 5);
	det2 = new IConvolutionLayer(det2_outch, DimsNCHW{ det2_outch, det2_inch, 1, 1 },
		weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);
	Dims dim_det2 = det2->init(dim_csp23);
}


yolo_v5_6::~yolo_v5_6()
{
	if (NULL != conv0)
	{
		delete conv0;
		conv0 = NULL;
	}
	if (NULL != conv1)
	{
		delete conv1;
		conv1 = NULL;
	}
	if (NULL != conv3)
	{
		delete conv3;
		conv3 = NULL;
	}
	if (NULL != conv5)
	{
		delete conv5;
		conv5 = NULL;
	}
	if (NULL != conv7)
	{
		delete conv7;
		conv7 = NULL;
	}
	if (NULL != conv10)
	{
		delete conv10;
		conv10 = NULL;
	}
	if (NULL != conv14)
	{
		delete conv14;
		conv14 = NULL;
	}
	if (NULL != conv18)
	{
		delete conv18;
		conv18 = NULL;
	}
	if (NULL != conv21)
	{
		delete conv21;
		conv21 = NULL;
	}
	if (NULL != bottleneck_CSP2)
	{
		delete bottleneck_CSP2;
		bottleneck_CSP2 = NULL;
	}
	if (NULL != bottleneck_csp4)
	{
		delete bottleneck_csp4;
		bottleneck_csp4 = NULL;
	}
	if (NULL != bottleneck_csp6)
	{
		delete bottleneck_csp6;
		bottleneck_csp6 = NULL;
	}
	if (NULL != bottleneck_csp8)
	{
		delete bottleneck_csp8;
		bottleneck_csp8 = NULL;
	}
	if (NULL != bottleneck_csp13)
	{
		delete bottleneck_csp13;
		bottleneck_csp13 = NULL;
	}
	if (NULL != bottleneck_csp17)
	{
		delete bottleneck_csp17;
		bottleneck_csp17 = NULL;
	}
	if (NULL != bottleneck_csp20)
	{
		delete bottleneck_csp20;
		bottleneck_csp20 = NULL;
	}
	if (NULL != bottleneck_csp23)
	{
		delete bottleneck_csp23;
		bottleneck_csp23 = NULL;
	}
	if (NULL != spp9)
	{
		delete spp9;
		spp9 = NULL;
	}
	if (NULL != upsample11)
	{
		delete upsample11;
		upsample11 = NULL;
	}
	if (NULL != upsample15)
	{
		delete upsample15;
		upsample15 = NULL;
	}
	if (NULL != cat12)
	{
		delete cat12;
		cat12 = NULL;
	}
	if (NULL != cat16)
	{
		delete cat16;
		cat16 = NULL;
	}
	if (NULL != cat19)
	{
		delete cat19;
		cat19 = NULL;
	}
	if (NULL != cat22)
	{
		delete cat22;
		cat22 = NULL;
	}
	if (NULL != det0)
	{
		delete det0;
		det0 = NULL;
	}
	if (NULL != det1)
	{
		delete det1;
		det1 = NULL;
	}
	if (NULL != det2)
	{
		delete det2;
		det2 = NULL;
	}
}

int yolo_v5_6::forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut, void *_tmp_buf)
{
	int block_size = 1024 * 1024 * 2;
	void *buffer1 = _tmp_buf;
	void *buffer2		= (char*)_tmp_buf + block_size ;		//Æ«ÒÆ1M
	void *csp4_buf		= (char*)_tmp_buf + block_size * 2;	//Æ«ÒÆ1M
	void *csp6_buf		= (char*)_tmp_buf + block_size * 3;	//Æ«ÒÆ1M
	void *conv10_buf	= (char*)_tmp_buf + block_size * 4;	//Æ«ÒÆ1M
	void *conv14_buf	= (char*)_tmp_buf + block_size * 5;	//Æ«ÒÆ1M
	void *csp17_buf		= (char*)_tmp_buf + block_size * 6;	//Æ«ÒÆ1M
	void *csp20_buf		= (char*)_tmp_buf + block_size * 7;	//Æ«ÒÆ1M
	void *csp23_buf		= (char*)_tmp_buf + block_size * 8;	//Æ«ÒÆ1M
	void *det0_buf		= (char*)_tmp_buf + block_size * 9;	//Æ«ÒÆ1M
	void *det1_buf		= (char*)_tmp_buf + block_size * 10;	//Æ«ÒÆ1M
	void *det2_buf		= (char*)_tmp_buf + block_size * 11;	//Æ«ÒÆ1M
	void *tmp_buf		= (char*)_tmp_buf + block_size * 12;	//Æ«ÒÆ1M
	
	Dims dim_tmp1 = DimsNCHW(1, 3, INPUT_H, INPUT_W);
	Dims dim_tmp2 = DimsNCHW(1, 3, INPUT_H, INPUT_W);

	conv0->forwardEx(_pInData, _stInPut, buffer1, dim_tmp1, tmp_buf);

	conv1->forwardEx(buffer1, dim_tmp1, buffer2, dim_tmp2, tmp_buf);

	bottleneck_CSP2->forwardEx(buffer2, dim_tmp2, buffer1, dim_tmp1, tmp_buf);

	conv3->forwardEx(buffer1, dim_tmp1, buffer2, dim_tmp2, tmp_buf);

	bottleneck_csp4->forwardEx(buffer2, dim_tmp2, csp4_buf, dim_tmp1, tmp_buf);
	Dims dim_csp4 = dim_tmp1;

	conv5->forwardEx(csp4_buf, dim_tmp1, buffer2, dim_tmp2, tmp_buf);

	bottleneck_csp6->forwardEx(buffer2, dim_tmp2, csp6_buf, dim_tmp1, tmp_buf);
	Dims dim_csp6 = dim_tmp1;

	conv7->forwardEx(csp6_buf, dim_tmp1, buffer2, dim_tmp2, tmp_buf);

	bottleneck_csp8->forwardEx(buffer2, dim_tmp2, buffer1, dim_tmp1, tmp_buf);

	spp9->forwardEx(buffer1, dim_tmp1, buffer2, dim_tmp2, tmp_buf);

	///* ------ yolov5 head ------ */
	conv10->forwardEx(buffer2, dim_tmp2, conv10_buf, dim_tmp1, tmp_buf);
	Dims dim_conv10 = dim_tmp1;

	upsample11->forward(conv10_buf, dim_tmp1, buffer2, dim_csp6);

	//ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
	cat12->forward(buffer2, dim_csp6, csp6_buf, dim_csp6, buffer1, dim_tmp1);

	bottleneck_csp13->forwardEx(buffer1, dim_tmp1, buffer2, dim_tmp2, tmp_buf);

	conv14->forwardEx(buffer2, dim_tmp2, conv14_buf, dim_tmp1, tmp_buf);
	Dims dim_conv14 = dim_tmp1;

	upsample15->forward(conv14_buf, dim_tmp1, buffer2, dim_csp4);

	cat16->forward(buffer2, dim_csp4, csp4_buf, dim_csp4, buffer1, dim_tmp1);

	bottleneck_csp17->forwardEx(buffer1, dim_tmp1, csp17_buf, dim_tmp2, tmp_buf);
	Dims dim_csp17 = dim_tmp2;

	/* ------ detect ------ */
	det0->forwardEx(csp17_buf, dim_csp17, det0_buf, dim_tmp1);
	Dims dim_det0 = dim_tmp1;
	conv18->forwardEx(csp17_buf, dim_csp17, buffer1, dim_tmp1, tmp_buf);

	cat19->forward(buffer1, dim_tmp1, conv14_buf, dim_conv14, buffer2, dim_tmp2);

	bottleneck_csp20->forwardEx(buffer2, dim_tmp2, csp20_buf, dim_tmp1, tmp_buf);
	Dims dim_csp20 = dim_tmp1;

	det1->forwardEx(csp20_buf, dim_tmp1, det1_buf, dim_tmp2);
	Dims dim_det1 = dim_tmp2;

	conv21->forwardEx(csp20_buf, dim_csp20, buffer1, dim_tmp1, tmp_buf);

	cat22->forward(buffer1, dim_tmp1, conv10_buf, dim_conv10, buffer2, dim_tmp2);

	bottleneck_csp23->forwardEx(buffer2, dim_tmp2, buffer1, dim_tmp1, tmp_buf);
	
	det2->forwardEx(buffer1, dim_tmp1, det2_buf, dim_tmp2);
	
	return 0;
}
