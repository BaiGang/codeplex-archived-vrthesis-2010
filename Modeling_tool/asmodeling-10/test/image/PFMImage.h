/********************************************************************
	created:	2007/06/18
	created:	18:6:2007   21:27
	filename: 	J:\HuYong\SVBRDF\CommonTools\image\PFMImage.h
	file path:	J:\HuYong\SVBRDF\CommonTools\image
	file base:	PFMImage
	file ext:	h
	author:		HuYong
	
	purpose:	Portable Float Map �ļ�
*********************************************************************/


#pragma once
#include "image.h"
class PFMImage : public Image
{
public:
	//�������캯��
	PFMImage(const PFMImage& pfmImage);
	//��ֵ����
	PFMImage& operator=(const PFMImage& pfmImage);

	//pfmTypeΪ0ʱ����grayscale���������color
	PFMImage(int width, int height, int pfmType, float* pixel);
	PFMImage(int type);//typeΪ0����ͨ����Ϊ1����3ͨ��
	virtual ~PFMImage(void);
	//u��������꣬v���������꣬ͼ�����Ͻ�Ϊ(0,0)�㣬v���ϵ��£�u�����ң���HDRShop����ͬ
	virtual PixelFormat GetPixel(int u, int v) const;
	virtual bool ReadImage(const char *strFileName);
	virtual bool WriteImage(const char *strFileName);
	//��������ݱ���ΪRGB3ͨ��ֵ
	void SetPixelData(const float* pixel, int width, int height);
	float* GetPixelDataBuffer(void);
	virtual void SetPixel(int u, int v, PixelFormat pf);
	virtual void ClearImage(void);
	
private:
	bool WriteToGrayscalePFM(const char *strFileName);
	bool WriteToColorPFM(const char *strFileName);
private:
	float* m_pPixel;
	int m_type;//m_typeΪ0ʱ����grayscale���������color,

};
