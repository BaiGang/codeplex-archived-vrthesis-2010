#include "BMPImage.h"
#include <assert.h>
#include <memory.h>

BMPImage::BMPImage(IMAGETYPE imageType):Image(imageType)
{
	m_pPixel = NULL;
	m_bitCount = 0; 
	m_bytesNum = 0;
}

BMPImage::BMPImage(IMAGETYPE imageType, int width, int height, WORD bitCount, BYTE* pixel):Image(imageType, width, height)
{
	assert(0 != bitCount);

	m_bitCount = bitCount;
	int bpp = m_bitCount/8;//bpp means byte per pixel
	assert(0 != bpp);

	//BMP图像每行行字节数必须是4的倍数
	if(0 == (m_bitCount * m_width) % 32)
	{
		m_bpl = (m_bitCount * m_width) / 8;
	}
	else
	{
		int fillByte = ((m_bitCount*m_width)/8)&0x03;
		fillByte = 4 - fillByte;
		m_bpl = ((m_bitCount * m_width) / 8) + fillByte;
	}	

	m_bytesNum = m_bpl*m_height;

	m_pPixel = new BYTE[m_bytesNum];
	if(NULL != pixel)
	{		
		memcpy(m_pPixel, pixel, m_bytesNum);
	}
	else
	{
		memset(m_pPixel, 0, m_bytesNum);
	}
}

BMPImage::~BMPImage(void)
{
	if(NULL != m_pPixel)
	{
		delete[] m_pPixel;
	}
}

BMPImage::BMPImage(const BMPImage& bmpImage)
{
	m_imageType = bmpImage.m_imageType;
	m_width = bmpImage.m_width;
	m_height = bmpImage.m_height;
	m_bitCount = bmpImage.m_bitCount;
	m_bytesNum = bmpImage.m_bytesNum;

	m_pPixel = new BYTE[m_bytesNum];
	memcpy(m_pPixel, bmpImage.m_pPixel, m_bytesNum);
}

BMPImage& BMPImage::operator=(const BMPImage& bmpImage)
{
	//检查自赋值
	if(this == &bmpImage)
	{
		return *this;
	}

	//释放原有内存资源
	if(NULL != m_pPixel)
	{
		delete[] m_pPixel;
	}

	//分配新资源，并赋值
	m_imageType = bmpImage.m_imageType;
	m_width = bmpImage.m_width;
	m_height = bmpImage.m_height;
	m_bitCount = bmpImage.m_bitCount;
	m_bytesNum = bmpImage.m_bytesNum;

	m_pPixel = new BYTE[m_bytesNum];
	memcpy(m_pPixel, bmpImage.m_pPixel, m_bytesNum);

	//返回引用
	return *this;
}

PixelFormat BMPImage::GetPixel(int u, int v) const
{
	PixelFormat pf;
	switch(m_bitCount) {
	case 1:
	case 4:
	case 8:
	case 32:
		{
		
			MessageBox(NULL, "Only support 24-bit bitmap!", "Error", MB_OK);
			break;
		}
	case 24:
		{
			int index =0;
			BYTE* tmpPtr = m_pPixel;		
			index = (m_height-1-v) * m_bpl + u * 3;
			tmpPtr += index;
			pf.b = (*tmpPtr)/255.0f;
			pf.g = (*(++tmpPtr))/255.0f;
			pf.r = (*(++tmpPtr))/255.0f;
			pf.alpha = 1.0f;
			break;			
		}
	default:
		{	
			MessageBox(NULL, "Error: Return Error!", "Error", MB_OK);		
			break;
		}
	}
	return pf;
}

void BMPImage::SetPixel(int u, int v, PixelFormat pf)
{

	switch(m_bitCount) {
	case 1:
	case 4:
	case 8:
	case 32:
		{

			MessageBox(NULL, "Only support 24-bit bitmap!", "Error", MB_OK);
			break;
		}
	case 24:
		{
			int index =0;
			BYTE* tmpPtr = m_pPixel;		
			index = (m_height-1-v) *m_bpl + u * 3;
			tmpPtr += index;
			(*tmpPtr) = (BYTE)(255 * pf.b+0.5f);
			(*(++tmpPtr)) = (BYTE)(255 * pf.g+0.5f);
			(*(++tmpPtr)) = (BYTE)(255 * pf.r+0.5f);
			break;			
		}
	default:
		{	
			MessageBox(NULL, "Error: Return Error!", "Error", MB_OK);		
			break;
		}
	}
}

bool BMPImage::ReadImage(const char *strFileName)
{
	FILE* in;
	if(!(in = fopen(strFileName, "rb")))
	{
		string strError("无法打开文件：");
		strError += strFileName;
		MessageBox(NULL, strError.c_str(), "error",MB_OK);
		return false;
	}

	BITMAPFILEHEADER bf;
	BITMAPINFOHEADER bi;
	DWORD bpl;	//bytes per line


	fread (&bf, 1, sizeof(BITMAPFILEHEADER), in);
	fread (&bi, 1, sizeof(BITMAPINFOHEADER), in);
	m_width = bi.biWidth;
	m_height = bi.biHeight;
	m_bitCount = bi.biBitCount;
	//BMP图像每行行字节数必须是4的倍数
	if(0 == (bi.biBitCount * m_width) % 32)
	{
		bpl = (bi.biBitCount * m_width) / 8;
	}
	else
	{
		int fillByte = ((bi.biBitCount*m_width)/8)&0x03;
		fillByte = 4 - fillByte;
		bpl = ((bi.biBitCount * m_width) / 8) + fillByte;
	}
	m_bpl = bpl;
	m_bytesNum = bpl * m_height;

	switch(m_bitCount)
	{
	case 1:
	case 4:
	case 8:
	case 32:
		{
			fclose(in);
			MessageBox(NULL, "Only support 24-bit bitmap!", "Error", MB_OK);
			break;
		}
	case 24:
		{
			m_pPixel = new BYTE[m_bytesNum];
			int num = fread(m_pPixel, 1, m_bytesNum, in) ;
			if(num != m_bytesNum)
			{
				fclose(in);
				MessageBox(NULL, "Error: Read BMP Image Error!", "Error", MB_OK);
				return false;
			}
 			fclose(in);
			return true;
		}
	default:
		{
			fclose(in);
			break;
		}
		

	}
	return false;
}

bool BMPImage::WriteImage(const char *strFileName)
{
	switch(m_bitCount) 
	{
		case 1:
		case 4:
		case 8:
		case 32:
			{
				MessageBox(NULL, "Only support 24-bit bitmap!", "Error", MB_OK);
				break;
			}
		case 24:
			{
				bool result = Write24BitBMP(strFileName);	
				return result;

			}
		default:
			{
				
				break;
			}


	}
	return false;
	
}

bool BMPImage::Write24BitBMP(const char *strFileName)
{
	if(strFileName == NULL || m_pPixel == NULL)
	{
		MessageBox(NULL, "Error:fileName or Image data is null!", "Error", MB_OK);
		return false;
	}
	FILE* fp=fopen(strFileName,"wb");
	if(!fp)
		return false;

	
	

	BITMAPFILEHEADER bitmapFileHeader;
	BITMAPINFOHEADER bmpInfoHeaderPtr ;

	bitmapFileHeader.bfType = 0x4D42;
	bitmapFileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER); // 54
	bitmapFileHeader.bfSize = m_width * m_height + bitmapFileHeader.bfOffBits; 
	bitmapFileHeader.bfReserved1 = 0;
	bitmapFileHeader.bfReserved2 = 0;

	bmpInfoHeaderPtr.biSize			 = 40;
	bmpInfoHeaderPtr.biWidth		 = m_width;
	bmpInfoHeaderPtr.biHeight	     = m_height;
	bmpInfoHeaderPtr.biPlanes        = 1;
	bmpInfoHeaderPtr.biBitCount      = 24;
	bmpInfoHeaderPtr.biCompression   = 0;
	bmpInfoHeaderPtr.biSizeImage     = 0;
	bmpInfoHeaderPtr.biXPelsPerMeter = 1024;//any digit maybe
	bmpInfoHeaderPtr.biYPelsPerMeter = 768; //any digit maybe
	bmpInfoHeaderPtr.biClrUsed       = 0;
	bmpInfoHeaderPtr.biClrImportant  = 0;

	fwrite((void*) &bitmapFileHeader, 1, sizeof(BITMAPFILEHEADER), fp);
	fwrite((void*) &bmpInfoHeaderPtr, 1, sizeof(BITMAPINFOHEADER), fp);

	fwrite(m_pPixel, 1,m_bytesNum,fp);
	fclose(fp);
	return true;
}

void BMPImage::SetPixelData(const BYTE* pixel, DWORD bytesNum, WORD bitCount)
{
	//m_bitCount = bitCount;
	//int bpp = m_bitCount/8;//bpp means byte per pixel
	//assert(0 != bpp);

	////BMP图像每行行字节数必须是4的倍数
	//if(0 == (m_bitCount * m_width) % 32)
	//{
	//	m_bpl = (m_bitCount * m_width) / 8;
	//}
	//else
	//{
	//	int fillByte = ((m_bitCount*m_width)/8)&0x03;
	//	fillByte = 4 - fillByte;
	//	m_bpl = ((m_bitCount * m_width) / 8) + fillByte;
	//}	

	//m_bytesNum = m_bpl*m_height;

	//m_bytesNum = bytesNum;
	//m_bitCount = bitCount;
	//if(NULL != m_pPixel)
	//{
	//	memcpy(m_pPixel, pixel, m_bytesNum);
	//}
	//else
	//{
	//	m_pPixel = new BYTE[m_bytesNum];
	//	memcpy(m_pPixel, pixel, m_bytesNum);
	//}

	MessageBox(NULL, "Did not surport!", "Error", MB_OK);
}

void BMPImage::ClearImage(void)
{
	assert(m_pPixel != NULL);
	memset(m_pPixel, 0, m_bytesNum);
}
