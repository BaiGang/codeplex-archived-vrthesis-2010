#include<windows.h>
#include<fstream>
#include<memory.h>
#include<string.h>
#include "pfmimage.h"
using namespace std;

PFMImage::PFMImage(const PFMImage& pfmImage)
{
	this->m_type = pfmImage.m_type;
	this->m_width = pfmImage.m_width;
	this->m_height = pfmImage.m_height;
	if(0 == m_type)
	{
		m_pPixel = new float[m_width*m_height];
		memcpy(m_pPixel, pfmImage.m_pPixel, m_width*m_height*sizeof(float));
	}
	else
	{
		m_pPixel = new float[3*m_width*m_height];
		memcpy(m_pPixel, pfmImage.m_pPixel, 3*m_width*m_height*sizeof(float));
	}

}

PFMImage& PFMImage::operator=(const PFMImage& pfmImage)
{
	//检查自付值
	if(this == &pfmImage)
	{
		return *this;
	}

	//释放原有内存资源
	if(NULL != this->m_pPixel)
	{
		delete[] m_pPixel;
	}

	//分配新资源，并赋值
	this->m_type = pfmImage.m_type;
	this->m_width = pfmImage.m_width;
	this->m_height = pfmImage.m_height;
	if(0 == m_type)
	{
		m_pPixel = new float[m_width*m_height];
		memcpy(m_pPixel, pfmImage.m_pPixel, m_width*m_height*sizeof(float));
	}
	else
	{
		m_pPixel = new float[3*m_width*m_height];
		memcpy(m_pPixel, pfmImage.m_pPixel, 3*m_width*m_height*sizeof(float));
	}

	//返回引用
	return *this;

}

PFMImage::PFMImage(int width, int height, int pfmType, float* pixel):Image(PFM, width, height)
{

	m_type = pfmType;
	if(0 == m_type)
	{
		m_pPixel = new float[m_width*m_height];
		if (pixel == NULL)
		{
			memset(m_pPixel, 0, m_width*m_height*sizeof(float));
		}
		else
		{
			memcpy(m_pPixel, pixel, m_width*m_height*sizeof(float));
		}
		
	}
	else
	{
		m_pPixel = new float[3*m_width*m_height];
		if (pixel == NULL)
		{
			memset(m_pPixel, 0, 3*m_width*m_height*sizeof(float));
		}
		else
		{
			memcpy(m_pPixel, pixel, 3*m_width*m_height*sizeof(float));
		}
		
	}

}

PFMImage::PFMImage(int type):Image(PFM)
{
	m_pPixel = NULL;
	m_type = type;
}

PFMImage::~PFMImage(void)
{
	if(m_pPixel != NULL)
	{
		delete[] m_pPixel;
		m_pPixel = NULL;
	}
	
}

void PFMImage::SetPixelData(const float* pixel, int width, int height)
{
	m_width = width;
	m_height = height;
	if(0 == m_type)
	{
		if(NULL == m_pPixel)
		{
			m_pPixel = new float[m_width*m_height];
		}
		memcpy(m_pPixel, pixel, m_width*m_height*sizeof(float));
	}
	else if(1 == m_type)
	{
		if(NULL == m_pPixel)
		{
			m_pPixel = new float[3*m_width*m_height];
		}
		memcpy(m_pPixel, pixel, 3*m_width*m_height*sizeof(float));
	}
	else
	{
		MessageBox(NULL, "Error: Unknown PFM image type!", "Error",MB_OK);
	}
	
}

bool PFMImage::WriteImage(const char *strFileName)
{
	if(m_type == 0)
	{
		return this->WriteToGrayscalePFM(strFileName);
	}
	else
	{
		return this->WriteToColorPFM(strFileName);
	}
	return true;
}

bool PFMImage::WriteToGrayscalePFM(const char *strFileName)
{
	if(m_pPixel == NULL)
	{
		MessageBox(NULL, "Error: There is no image data!", "Error",MB_OK);
		return false;
	}
	ofstream ofs(strFileName, ios::binary);
	ofs<<"Pf"<<"\n"<<m_width<<" "<<m_height<<"\n-1.000000\n";
	float *pTempPixel = m_pPixel;
	for(int i=0; i<m_width; i++)
	{
		for(int j=0; j<m_height; j++)
		{
			ofs.write((char*)pTempPixel, sizeof(float));
			pTempPixel+=1;
		}		

	}	
	ofs.close();
	return true;
}

bool PFMImage::WriteToColorPFM(const char *strFileName)
{
	if(m_pPixel == NULL)
	{
		MessageBox(NULL, "Error: There is no image data!", "Error",MB_OK);
		return false;
	}

	ofstream ofs(strFileName, ios::binary);

	ofs<<"PF"<<"\n"<<m_width<<" "<<m_height<<"\n-1.000000\n";
	float *pTempPixel = m_pPixel;
	for(int i=0; i<m_width; i++)
	{
		for(int j=0; j<m_height; j++)
		{
			ofs.write((char*)pTempPixel, 3*sizeof(float));
			pTempPixel+=3;
		}
	}
	ofs.close();
	return true;
}

bool PFMImage::ReadImage(const char *strFileName)
{
	ifstream ifs(strFileName, ios::binary);
	if(ifs)
	{
		string PFMType;
		string tmpString;

		ifs>>PFMType;
		ifs>>m_width;
		ifs>>m_height;
		ifs>>tmpString;
		//除去一个换行符
		char a;
		ifs.read(&a,1);
		if(0 == PFMType.compare("PF"))
		{
			if(NULL == m_pPixel)
			{
				m_pPixel = new float[3*m_width*m_height];
			}
			//PF代表为3通道color data
			m_type = 1;
			float* tmpPtr = m_pPixel;
			for(int i=0; i<m_width; i++)
				for(int j=0; j<m_height; j++)
				{
					ifs.read((char*)tmpPtr, 3*sizeof(float));
					tmpPtr += 3;
				}
		}
		else if(0 == PFMType.compare("Pf"))
		{
			if(NULL == m_pPixel)
			{
				m_pPixel = new float[m_width*m_height];
			}
			m_type = 0;
			//Pf代表为单通道数据
			float* tmpPtr = m_pPixel;
			for(int i=0; i<m_width; i++)
				for(int j=0; j<m_height; j++)
				{
					ifs.read((char*)tmpPtr, sizeof(float));
					tmpPtr++;
					
				}
		}
		else
		{
			ifs.close();
			return false;
		}
	}
	else
	{
		ifs.close();
		MessageBox(NULL, "Error: Can not open this PFM file!!", "Error",MB_OK);
		return false;
	}
	ifs.close();
	return true;
}
PixelFormat  PFMImage::GetPixel(int u, int v) const
{
	PixelFormat tmpPF;
	if((u<0) || (v<0)|| (u>=m_width) || (v>=m_height))
	{
		tmpPF.r = tmpPF.g = tmpPF.b = 0.0f;
		return tmpPF;
	}

	int index =0;
	float* tmpPtr = m_pPixel;
	if(0 == m_type)
	{
		 index = (m_height-1-v) * m_width + u;
		 tmpPtr += index;
		 tmpPF.r = *tmpPtr;
		 tmpPF.g = *tmpPtr;
		 tmpPF.b = *tmpPtr;
		 tmpPF.alpha = 1.0f;
	}
	else
	{
		index = (m_height-1-v) * m_width * 3 + u * 3;
		tmpPtr += index;
		tmpPF.r = *tmpPtr;
		tmpPF.g = *(++tmpPtr);
		tmpPF.b = *(++tmpPtr);
		tmpPF.alpha = 1.0f;
	}
	
	return tmpPF;
}

void PFMImage::SetPixel(int u, int v, PixelFormat pf)
{
	int index =0;
	float* tmpPtr = m_pPixel;
	if(0 == m_type)
	{
		index = (m_height-1-v) * m_width + u;
		tmpPtr += index;
		*tmpPtr = pf.r;		
	}
	else
	{
		index = (m_height-1-v) * m_width * 3 + u * 3;
		tmpPtr += index;
		*tmpPtr = pf.r;
		*(++tmpPtr) = pf.g;
		*(++tmpPtr) = pf.b;
		
	}

}

void PFMImage::ClearImage(void)
{
	
}

float* PFMImage::GetPixelDataBuffer(void)
{
	return m_pPixel;
}

















