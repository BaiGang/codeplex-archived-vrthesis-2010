/********************************************************************
	created:	2007/07/02
	created:	2:7:2007   16:37
	filename: 	I:\HuYong\SVBRDF\CommonTools\image\BMPImage.h
	file path:	I:\HuYong\SVBRDF\CommonTools\image
	file base:	BMPImage
	file ext:	h
	author:		HuYong
	
	purpose:	24λBMPͼ��Ĵ�ȡ����
*********************************************************************/

#pragma once
#include "image.h"
#include<windows.h>
class BMPImage : public Image
{
public:

	BMPImage(IMAGETYPE imageType, int width, int height, WORD bitCount, BYTE* pixel);

	BMPImage(IMAGETYPE imageType = BMP);
	virtual ~BMPImage(void);
	//�������캯��
	BMPImage(const BMPImage& bmpImage);
	//��ֵ����
	BMPImage& operator=(const BMPImage& bmpImage);


	//u��������꣬v���������꣬ͼ�����Ͻ�Ϊ(0,0)�㣬v���ϵ��£�u�����ң���HDRShop����ͬ
	virtual PixelFormat GetPixel(int u, int v) const;
	virtual void SetPixel(int u, int v, PixelFormat pf);
	virtual bool ReadImage(const char *strFileName);
	virtual bool WriteImage(const char *strFileName);
	virtual void ClearImage(void);
	//bytesNumΪ�ֽ����� bitCountΪÿ������ռ��λ��
	void SetPixelData(const BYTE* pixel, DWORD bytesNum, WORD bitCount);
private:
	bool Write24BitBMP(const char *strFileName);

private:
	BYTE* m_pPixel;
	WORD m_bitCount; 
	DWORD m_bytesNum;
	DWORD m_bpl;//ÿ���ֽ���

};
