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
	float alpha;//当图像数据为HDR图像是，alpha为零，即无意义
};

class Image
{
public:
	Image(void);
	Image(IMAGETYPE imageType, int width=0, int height=0);
	virtual ~Image(void);
	//u代表横坐标，v代表纵坐标，图像左上角为(0,0)点，v从上到下，u从左到右，与HDRShop中相同
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
