#ifndef _CUDA_IMG_UTIL_H_
#define _CUDA_IMG_UTIL_H_

#include "CudaImgUtil.h"

/* define return codes for WriteBMP() */
#ifndef GFXIO_ERRORS
#define GFXIO_ERRORS
    #define GFXIO_OK            0
    #define GFXIO_OPENERROR     1
    #define GFXIO_BADFILE       2
    #define GFXIO_UNSUPPORTED   3
#endif

/* some useful bitmap constants, prefixed with nonsense to not overlap with */
/*    potential MS Windows definitions...                                   */
#define MYBMP_BF_TYPE           0x4D42
#define MYBMP_BF_OFF_BITS       54
#define MYBMP_BI_SIZE           40
#define MYBMP_BI_RGB            0L
#define MYBMP_BI_RLE8           1L
#define MYBMP_BI_RLE4           2L
#define MYBMP_BI_BITFIELDS      3L

namespace cuda_imageutil
{

	class BMPImageUtil : public CudaImageUtil<uint8, 3>
	{
	public:
		// I / O
		virtual bool LoadImage(char * imageFileName);
		virtual bool SaveImage(char * imageFileName);
	};

} // namespace cuda_imageutil

#endif //_CUDA_IMG_UTIL_H_