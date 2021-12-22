#ifndef __ICONVOLUTION_LAYER_H__
#define __ICONVOLUTION_LAYER_H__

#include "algo_common.h"


//typedef struct
//{
//	//algorithms for forward convolutions
//	cudnnConvolutionFwdAlgo_t*		fwd_algo_;
//	cudnnFilterDescriptor_t			pFilterDesc;
//	cudnnTensorDescriptor_t			pBiasDesc;
//	cudnnTensorDescriptor_t			pInputDesc;
//	cudnnTensorDescriptor_t			pOutputDesc;
//	cudnnConvolutionDescriptor_t	pConvDesc;
//
//	int			group_;
//	int			top_offset_;
//	int			bias_offset_;
//	int			bottom_offset_;
//	bool		has_bias_term_;
//
//	void**		workspace;				// aliases into workspaceData
//	void*		workspaceData;			// underlying storage
//	size_t*		workspace_fwd_sizes_;
//	size_t		workspaceSizeInBytes;  // size of underlying storage
//	size_t		total_workspace_fwd;
//} CNN_CONV_S;

class IConvolutionLayer
{
public:
	IConvolutionLayer();
	IConvolutionLayer(int _nbOutputMaps, Dims _kernelSize, Weights _kernelWeights, Weights _biasWeights);
	~IConvolutionLayer();
	//传入数据地址是显存地址
	int forward(void* _pInData, Dims _stInPut, void* _pOutData, Dims _stOutPut);

public:
	int setPadding(Dims _iPadding)
	{
		m_stPadding = _iPadding;
		return 0;
	};

	int setKernel(Dims _iKernel)
	{
		m_stKernel = _iKernel;
		return 0;
	};

	int setStride(Dims _iStride)
	{
		m_stStride = _iStride;
		return 0;
	};

	int setDilation(Dims _iDilation)
	{
		m_stDilation = _iDilation;
		return 0;
	};

	int setGroup(int _iGroup)
	{
		m_iGroup = _iGroup;
		return 0;
	};

	/////////////////固定输入输出加速方法////////////////////////////////////////////
	Dims init(Dims _stInPut);
	int forwardEx(void* _pInData, Dims _stInPut, void* _pOutData, Dims &_stOutPut);
	
private:
	Weights	*m_pstBias;		//Bias数据
	Weights	*m_pstWeights;	//卷积权重
	int m_nbOutputMaps;

	//卷积参数
	int		m_iGroup;		//可变卷积预留
	Dims	m_stKernel;		//卷积核宽度
	Dims	m_stStride;		//宽度方向步长
	Dims	m_stPadding;	//宽度方向填充像素数
	Dims	m_stDilation;	//核宽度方向填充像素数 空洞卷积预留
	bool	m_bHasBias;
	/////////////////////////////////////////////////////////////////
	cudnnHandle_t m_handle;
	cudnnTensorDescriptor_t m_input_descriptor;
	cudnnTensorDescriptor_t m_output_descriptor;
	cudnnFilterDescriptor_t m_kernel_descriptor;
	cudnnConvolutionDescriptor_t m_conv_descriptor;
	cudnnConvolutionFwdAlgo_t m_algo;
	cudnnTensorDescriptor_t m_bias_descriptor;
	void * m_workspace;
	size_t m_workspace_size;
};

#endif//__ICONVOLUTION_LAYER_H__