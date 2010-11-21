/********************************************************************
	created:	2007/07/02
	created:	2:7:2007   16:37
	filename: 	I:\HuYong\SVBRDF\CommonTools\image\BMPImage.h
	file path:	I:\HuYong\SVBRDF\CommonTools\image
	file base:	BMPImage
	file ext:	h
	author:		HuYong
	
	purpose:	24位BMP图像的存取操作
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
	//拷贝构造函数
	BMPImage(const BMPImage& bmpImage);
	//赋值函数
	BMPImage& operator=(const BMPImage& bmpImage);


	//u代表横坐标，v代表纵坐标，图像左上角为(0,0)点，v从上到下，u从左到右，与HDRShop中相同
	virtual PixelFormat GetPixel(int u, int v) const;
	virtual void SetPixel(int u, int v, PixelFormat pf);
	virtual bool ReadImage(const char *strFileName);
	virtual bool WriteImage(const char *strFileName);
	virtual void ClearImage(void);
	//bytesNum为字节数， bitCount为每像素所占用位数
	void SetPixelData(const BYTE* pixel, DWORD bytesNum, WORD bitCount);
private:
	bool Write24BitBMP(const char *strFileName);

private:
	BYTE* m_pPixel;
	WORD m_bitCount; 
	DWORD m_bytesNum;
	DWORD m_bpl;//每行字节数

};
