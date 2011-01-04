#pragma once
#include <string>
using namespace std;
enum IMAGETYPE{HDR=0, JPG, TGA, BMP, PFM};
struct UVCoordinate 
{
	short u;
	short v;
};
struct PixelFormat
{
	float r;
	float g;
	float b;
	float alpha;//��ͼ������ΪHDRͼ���ǣ�alphaΪ�㣬��������
};

class Image
{
public:
	Image(void);
	Image(IMAGETYPE imageType, int width=0, int height=0);
	virtual ~Image(void);
	//u��������꣬v���������꣬ͼ�����Ͻ�Ϊ(0,0)�㣬v���ϵ��£�u�����ң���HDRShop����ͬ
	virtual PixelFormat GetPixel(int u, int v) const = 0;
	virtual void SetPixel(int u, int v, PixelFormat pf) = 0;
	virtual bool ReadImage(const char *strFileName) = 0;
	virtual bool WriteImage(const char *strFileName) = 0;
	virtual void ClearImage(void) = 0;
	int GetWidth(void) const;
	int GetHeight(void) const;
	IMAGETYPE GetImageType(void);

protected:
	IMAGETYPE m_imageType;
	int m_height;
	int m_width;
};
