#include <stdafx.h>
#include <cstdio>

#include "ASModeling.h"
#include "../Utils/image/PFMImage.h"
#include "../Utils/image/BMPImage.h"

////////////////////////////////////////////////////////
//
//  Load captured image for current frame
//
//   All the frame images are stored at
//     ../Data/Camera**/Frame****
//     Cameras are numbered from 0 to num_cameras_-1
//
////////////////////////////////////////////////////////

namespace as_modeling
{
	bool ASModeling::load_captured_images(int iframe)
	{
#ifndef __USE_PFM_IMAGE__ // if use BMP img
		char path_buf[200];
		//cuda_imageutil::BMPImageUtil tmpBMP;


		for (int i_camera = 0; i_camera < num_cameras_; ++i_camera)
		{
			int zbase = i_camera * height_;

			BMPImage tmpBMP;
			sprintf_s(path_buf, 200, "D:/BaiGang/NEWNEW/asmodeling-56327/Data/result%02d/result_smoke2_Scene4_%d-DVR Express CLSAS%d.bmp", i_camera, 1+i_camera, 2907+iframe);

			//sprintf_s(path_buf, 200, "../Data/Camera%02d/Frame%05d.bmp", i_camera, iframe);

			if (!tmpBMP.ReadImage(path_buf))
			{
				fprintf(stderr, "Error when loading %s \n", path_buf);
				return false;
			}

			for (int y = 0; y < height_; ++y)
			{
				for (int x = 0; x < width_; ++x)
				{
#if 0
					ground_truth_image_.GetPixelAt(x,zbase+height_-y)[0] = tmpBMP.GetPixelAt(x,y)[0];
					ground_truth_image_.GetPixelAt(x,zbase+height_-y)[1] = tmpBMP.GetPixelAt(x,y)[1];
					ground_truth_image_.GetPixelAt(x,zbase+height_-y)[2] = tmpBMP.GetPixelAt(x,y)[2];
#else
					//ground_truth_image_.GetPixelAt(x,zbase+y)[0] = tmpBMP.GetPixelAt(x,y)[0];
					//ground_truth_image_.GetPixelAt(x,zbase+y)[1] = tmpBMP.GetPixelAt(x,y)[1];
					//ground_truth_image_.GetPixelAt(x,zbase+y)[2] = tmpBMP.GetPixelAt(x,y)[2];

					ground_truth_image_.GetPixelAt(x,zbase+y)[0] = 255 * tmpBMP.GetPixel(x, y).r;
					ground_truth_image_.GetPixelAt(x,zbase+y)[1] = 255 * tmpBMP.GetPixel(x, y).r;
					ground_truth_image_.GetPixelAt(x,zbase+y)[2] = 255 * tmpBMP.GetPixel(x, y).r;

#endif
					ground_truth_image_.GetPixelAt(x,zbase+y)[3] = 255;
				}
			}
		} // for each camera
#else // use PFM image
		char path_buf[200];
		PFMImage tmpPFM(1);

		for (int i_camera = 0; i_camera < num_cameras_; ++i_camera)
		{
			int zbase = i_camera * height_;
			sprintf_s(path_buf, 200, "../Data/Camera%02d/image%d.pfm", i_camera, i_camera);
			if (!tmpPFM.ReadImage(path_buf))
			{
				fprintf(stderr, "Error when loading %s \n", path_buf);
				return false;
			}

			for (int y = 0; y < height_; ++y)
			{
				for (int x = 0; x < width_; ++x)
				{
#if 1
					ground_truth_image_.GetPixelAt(x,zbase+height_-y)[0] = __min(255.0, __max(0.0, 255.0*tmpPFM.GetPixel(x, y).r) );
					ground_truth_image_.GetPixelAt(x,zbase+height_-y)[1] = __min(255.0, __max(0.0, 255.0*tmpPFM.GetPixel(x, y).g) );
					ground_truth_image_.GetPixelAt(x,zbase+height_-y)[2] = __min(255.0, __max(0.0, 255.0*tmpPFM.GetPixel(x, y).b) );
#else
					ground_truth_image_.GetPixelAt(x,zbase+y)[0] = __min(255.0, __max(0.0, 255.0*tmpPFM.GetPixel(x, y).r) );
					ground_truth_image_.GetPixelAt(x,zbase+y)[1] = __min(255.0, __max(0.0, 255.0*tmpPFM.GetPixel(x, y).g) );
					ground_truth_image_.GetPixelAt(x,zbase+y)[2] = __min(255.0, __max(0.0, 255.0*tmpPFM.GetPixel(x, y).b) );
#endif
					ground_truth_image_.GetPixelAt(x,zbase+y)[3] = 255;
				}
			}

		}  // for each view
#endif // __USE_PFM_IMAGE__

		return true;
	}
} // namespace as_modeling