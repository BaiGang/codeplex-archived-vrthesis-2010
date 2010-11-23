/********************************************************************
	created:	2007/06/18
	created:	18:6:2007   21:27
	filename: 	J:\HuYong\SVBRDF\CommonTools\image\PFMImage.h
	file path:	J:\HuYong\SVBRDF\CommonTools\image
	file base:	PFMImage
	file ext:	h
	author:		HuYong
	
	purpose:	Portable Float Map 文件
*********************************************************************/


#pragma once
#include "image.h"
class PFMImage : public Image
{
public:
	//拷贝构造函数
	PFMImage(const PFMImage& pfmImage);
	//赋值函数
	PFMImage& operator=(const PFMImage& pfmImage);

	//pfmType为0时代表grayscale，其余代表color
	PFMImage(int width, int height, int pfmType, float* pixel);
	PFMImage(int type);//type为0代表单通道，为1代表3通道
	virtual ~PFMImage(void);
	//u代表横坐标，v代表纵坐标，图像左上角为(0,0)点，v从上到下，u从左到右，与HDRShop中相同
	virtual PixelFormat GetPixel(int u, int v) const;
	virtual bool ReadImage(const char *strFileName);
	virtual bool WriteImage(const char *strFileName);
	//传入的数据必须为RGB3通道值
	void SetPixelData(const float* pixel, int width, int height);
	float* GetPixelDataBuffer(void);
	virtual void SetPixel(int u, int v, PixelFormat pf);
	virtual void ClearImage(void);
	
private:
	bool WriteToGrayscalePFM(const char *strFileName);
	bool WriteToColorPFM(const char *strFileName);
private:
	float* m_pPixel;
	int m_type;//m_type为0时代表grayscale，其余代表color,

};
