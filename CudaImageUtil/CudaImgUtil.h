//Copyright (c) 2010 BAI Gang.
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in
//all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//THE SOFTWARE.

#ifndef _CUDA_IMAGE_UTIL_H_
#define _CUDA_IMAGE_UTIL_H_

//====================================================================//
//
//    ImageUtil Lib by Baigang.
//
//====================================================================//

#include <cassert>
#include <cstdlib>
#include "CImgUtiltTpes.h"

namespace cuda_imageutil
{
	template < class ElemType = uint8, uint Channels = 3 >
	class CudaImageUtil
	{
	public:
		typedef ElemType  PixelType[Channels];

    ElemType dummy[Channels];

	public:
		// Constructor/Destructor
		CudaImageUtil() : m_width(0), m_height(0), m_pixelsP(NULL)
		{};
		~CudaImageUtil()
		{
			if (m_pixelsP != NULL)
			{
				delete m_pixelsP;
			}
		};

	public:
		// Accessors
		inline uint GetWidth() { return m_width; }
		inline uint GetHeight() { return m_height; }

		//// Access pixel
		inline ElemType * GetPixelAt(uint u, uint v)
		{
			//assert( u < m_width && v < m_height ); 
      if( u>= m_width || v >= m_height )
        return dummy;
			return & m_pixelsP [ (v * m_width + u) * Channels ];
		}

	public:
		inline bool SetSizes(uint width, uint height)
		{
			m_width = width;
			m_height = height;
			if (m_pixelsP != NULL)
			{
				delete m_pixelsP;
			}
			m_pixelsP = new ElemType[Channels * m_width * m_height];
			return true;
		}

	public:
		// I / O
		// NOTE: I/O issues should not be handle by this base class.
		// We just deal with in-core fairs.
		virtual bool LoadImage(const char * fname) { return false; };
		virtual bool SaveImage(const char * fname) { return false; };

	public:
		CudaImageUtil<ElemType, Channels>& operator = (CudaImageUtil<ElemType, Channels>& img)
		{
			this->SetSizes( img.GetWidth(), img.GetHeight() );
			memcpy(this->m_pixelsP, img.GetPixelAt(0, 0), sizeof(ElemType) * m_width * m_height * Channels);
			return *this;
		}
    CudaImageUtil<ElemType, Channels>& ClearImage()
    {
      memset(this->m_pixelsP, 0, sizeof(ElemType) * m_width * m_height * Channels);
      return *this;
    }

	private:
		ElemType *    m_pixelsP;
		uint          m_width;
		uint          m_height;
	};


	typedef CudaImageUtil <uint32,  3>   Image_3c32u;
	typedef CudaImageUtil <uint8,   3>   Image_3c8u;
	typedef CudaImageUtil <float,   3>   Image_3c32s;
	typedef CudaImageUtil <double,  3>   Image_3c64d;
	typedef CudaImageUtil <uint32,  4>   Image_4c32u;
	typedef CudaImageUtil <uint8,   4>   Image_4c8u;
	typedef CudaImageUtil <float,   4>   Image_4c32s;
	typedef CudaImageUtil <double,  4>   Image_4c64d;
	typedef CudaImageUtil <uint8,   1>   Image_1c8u;
	typedef CudaImageUtil <float,   1>   Image_1c32s;
	typedef CudaImageUtil <int8,    3>   Image_3c8i;	// Typically, a CIE-Lab image

}
#endif //_CUDA_IMAGE_UTIL_H_