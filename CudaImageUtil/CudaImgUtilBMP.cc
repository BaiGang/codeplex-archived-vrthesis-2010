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

#include <cstdio>

#include "CudaImgUtilBMP.h"

//======================================
//  Declarations of helper functions
//======================================
int            ReadInt(FILE *fp);
void           WriteInt(int x, FILE *fp);
unsigned short ReadUnsignedShort(FILE *fp);
void           WriteUnsignedShort(unsigned short int x, FILE *fp);
unsigned int   ReadUnsignedInt(FILE *fp);
void           WriteUnsignedInt(unsigned int x, FILE *fp);

namespace cuda_imageutil
{
	bool BMPImageUtil::LoadImage(char * imageFileName)
	{
		uint8 *img;
		uint8 *tmp;
		int   x, y;
		int   lineLength;

		FILE   *fp;
		uint16 bmpType, bmpReserved1, bmpReserved2;
		uint   bmpSize, bmpOffBits;

		uint   imgSize, imgSizeImage, imgCompression, imgClrUsed, imgClrImportant;
		int    imgWidth, imgHeight, imgXPelsPerMeter, imgYPelsPerMeter;
		uint16 imgPlanes, imgBitCount;

		bool invertY = false;

		fp = fopen( imageFileName, "rb" );
		if (!fp)
		{
			fprintf( stderr, "ReadBMP() unable to open file '%s'!\n\n", imageFileName );
			return false;
		}

		/* Read file header */
		bmpType = ReadUnsignedShort(fp);
		bmpSize = ReadUnsignedInt(fp);
		bmpReserved1 = ReadUnsignedShort(fp);
		bmpReserved2 = ReadUnsignedShort(fp);
		bmpOffBits = ReadUnsignedInt(fp);

		/* Check file header */
		if (bmpType != type || bmpOffBits != off_bits)
		{
			fprintf( stderr, "ReadBMP() encountered bad header in file '%s'!\n\n", imageFileName);
			return false;
		}

		/* Read info header */
		imgSize = ReadUnsignedInt(fp);
		imgWidth = ReadInt(fp);
		imgHeight = ReadInt(fp);
		imgPlanes = ReadUnsignedShort(fp);
		imgBitCount = ReadUnsignedShort(fp);
		imgCompression = ReadUnsignedInt(fp);
		imgSizeImage = ReadUnsignedInt(fp);
		imgXPelsPerMeter = ReadInt(fp);
		imgYPelsPerMeter = ReadInt(fp);
		imgClrUsed = ReadUnsignedInt(fp);
		imgClrImportant = ReadUnsignedInt(fp);

		/* Check info header */
		if( imgSize != bi_size || imgWidth <= 0 || 
			imgHeight <= 0 || imgPlanes != 1 || 
			imgBitCount != 24 || imgCompression != RGB ||
			imgSizeImage == 0 )
		{
			fprintf( stderr, "ReadBMP() encountered unsupported bitmap type in '%s'!\n\n", imageFileName);
			return false;
		}

		/* compute the line length */
		lineLength = imgWidth * 3; 
		if ((lineLength % 4) != 0) 
			lineLength = (lineLength / 4 + 1) * 4;

		/* Creates the image */
		SetSizes( imgWidth, imgHeight );
		tmp = new uint8[ 3 * lineLength ];
		img = GetPixelAt(0, 0);

		if (!tmp || !img)
		{
			fprintf( stderr, "Unable to allocate memory in ReadBMP()!\n\n");
			return false;
		}

		/* Position the file after header.  Header should be 54 bytes long -- checked above */
		fseek(fp, (long) bmpOffBits, SEEK_SET);  

		/* Read the image */
		for (y = 0; y < imgHeight; y++) {
			int yy = invertY ? (imgHeight-1-y) : y;
			fread(tmp, 1, lineLength, fp);

			/* Copy into permanent structure */
			for (x = 0; x < imgWidth; x++)
			{
				*(img+(yy*3*imgWidth)+3*x+2) = tmp[3*x+0]; 
				*(img+(yy*3*imgWidth)+3*x+1) = tmp[3*x+1]; 
				*(img+(yy*3*imgWidth)+3*x+0) = tmp[3*x+2]; 
			}
		}

		/* cleanup */
		delete[] tmp;
		fclose( fp );

		return true;
	} // LoadImage(char *)

	bool BMPImageUtil::SaveImage(char * imageFileName)
	{
		FILE *fp;
		char buf[1024];
		int x, y;
		int lineLength;

		int height = GetHeight();
		int width  = GetWidth();

		uint8 * ptr = GetPixelAt(0, 0);

		fp = fopen( imageFileName, "wb" );
		if (!fp)
		{
			fprintf( stderr, "WriteBMP() unable to open file '%s' for writing!\n", imageFileName );
			return false;
		}

		lineLength = width * 3;  
		if ((lineLength % 4) != 0)
			lineLength = (lineLength / 4 + 1) * 4;

		/* Write file header */
		WriteUnsignedShort( (unsigned short int) type,								              fp);
		WriteUnsignedInt  ( (unsigned int)       (off_bits + lineLength * height),	fp);
		WriteUnsignedShort( (unsigned short int) 0,											                    fp);
		WriteUnsignedShort( (unsigned short int) 0,										                      fp);
		WriteUnsignedInt  ( (unsigned short)     off_bits,                         fp);

		/* Write info header */
		WriteUnsignedInt  ( (unsigned short int) bi_size,                             fp);
		WriteInt          ( (int)                width,                                     fp);
		WriteInt          ( (int)                height,                                    fp);
		WriteUnsignedShort( (unsigned short int) 1,                                         fp);
		WriteUnsignedShort( (unsigned short int) 24,                                        fp);
		WriteUnsignedInt  ( (unsigned int)       RGB,                              fp);
		WriteUnsignedInt  ( (unsigned int)      (lineLength * (unsigned int) height),       fp);
		WriteInt          ( (int)				         2925,                                      fp);
		WriteInt          ( (int)				         2925,                                      fp);
		WriteUnsignedInt  ( (int)				         0,                                         fp);
		WriteUnsignedInt  ( (int)                0,                                         fp);

		/* Write pixels */
		for (y = 0; y < height; y++) 
		{
			int nbytes = 0;
			for (x = 0; x < width; x++) 
			{
				putc( *(ptr+(y*3*width)+3*x+2), fp), nbytes++;
				putc( *(ptr+(y*3*width)+3*x+1), fp), nbytes++;
				putc( *(ptr+(y*3*width)+3*x+0), fp), nbytes++;
			}
			/* Padding for 32-bit boundary */
			while ((nbytes % 4) != 0) 
			{
				putc(0, fp);
				nbytes++;
			}
		}

		fclose( fp );

		return true;
	} // SaveImage(char *)

} // namespace cuda_imageutil


//======================================
//
//        Helper functions
//
//=======================================

// Reads an unsigned short from a file in little endian format
static unsigned short int ReadUnsignedShort(FILE *fp)
{
    unsigned short int lsb, msb;

    lsb = getc(fp);
    msb = getc(fp);
    return (msb << 8) | lsb;
}

// Writes as unsigned short to a file in little endian format 
static void WriteUnsignedShort(unsigned short int x, FILE *fp)
{
    unsigned char lsb, msb;

    lsb = (unsigned char) (x & 0x00FF);
    msb = (unsigned char) (x >> 8);
    putc(lsb, fp);
    putc(msb, fp);
}

// Reads as unsigned int word from a file in little endian format
static unsigned int ReadUnsignedInt(FILE *fp)
{
    unsigned int b1, b2, b3, b4;

    b1 = getc(fp);
    b2 = getc(fp);
    b3 = getc(fp);
    b4 = getc(fp);
    return (b4 << 24) | (b3 << 16) | (b2 << 8) | b1;
}



// Writes an unsigned int to a file in little endian format 
static void WriteUnsignedInt(unsigned int x, FILE *fp)
{
    unsigned char b1, b2, b3, b4;

    b1 = (unsigned char) (x & 0x000000FF);
    b2 = (unsigned char) ((x >> 8) & 0x000000FF);
    b3 = (unsigned char) ((x >> 16) & 0x000000FF);
    b4 = (unsigned char) ((x >> 24) & 0x000000FF);
    putc(b1, fp);
    putc(b2, fp);
    putc(b3, fp);
    putc(b4, fp);
}


// Reads an int word from a file in little endian format 
static int ReadInt(FILE *fp)
{
    int b1, b2, b3, b4;

    b1 = getc(fp);
    b2 = getc(fp);
    b3 = getc(fp);
    b4 = getc(fp);
    return (b4 << 24) | (b3 << 16) | (b2 << 8) | b1;
}


// Writes an int to a file in little endian format 
static void WriteInt(int x, FILE *fp)
{
    char b1, b2, b3, b4;

    b1 = (x & 0x000000FF);
    b2 = ((x >> 8) & 0x000000FF);
    b3 = ((x >> 16) & 0x000000FF);
    b4 = ((x >> 24) & 0x000000FF);
    putc(b1, fp);
    putc(b2, fp);
    putc(b3, fp);
    putc(b4, fp);
}