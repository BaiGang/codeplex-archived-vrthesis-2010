#include "image.h"

Image::Image(void)
{

}

Image::Image(IMAGETYPE imageType, int width, int height)
{
	m_imageType = imageType;
	m_width = width;
	m_height = height;

}

Image::~Image(void)
{

}

int Image::GetWidth() const
{
	return m_width;
}

int Image::GetHeight() const
{
	return m_height;
}

IMAGETYPE Image::GetImageType()
{
	return m_imageType;
}
