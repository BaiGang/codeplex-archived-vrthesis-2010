#include <list>

#include "ASModeling.h"
#include "../Utils/math/ConvexHull2D.h"

/////////////////////////////////////////////////////////////////////////////
//
//  set_density_indicator()
//    set an volume, so that if the density of a cell is not zero, the 
//  corresponding cell in indicator is the array index of the cell in x[].
//
//  param- level : current level of the density volume
//  param- ind_volume:  the incidator volume
//  param- density_vectorized :
//     vectorized effective voxels
//  param- is_init_vol:
//     if false, just set the indicator volume, without init density.
//     else set indicator and init density by sum of pixels...
//
//  preliminaries: 
//      - the corresponding images have been loaded
//      - the pointers are pointing to preallocated space
//
/////////////////////////////////////////////////////////////////////////////


namespace as_modeling
{
  bool ASModeling::set_density_indicator(int level, int * ind_volume, std::list<float> & density_vectorized, bool is_init_vol)
  {
    int length = (1<<level);

    float wc_x[8];
    float wc_y[8];
    float wc_z[8];

    PT2DVEC pts;
    pts.reserve(16);


    if (is_init_vol)
    {
      density_vectorized.clear();
      density_vectorized.push_back(0.0f); // zero density value
    }

    memset(ind_volume, 0, sizeof(int)*length*length*length);

    // for each cell (i,j,k) 
    for (int k = 0; k < length; ++k)
    {
      for (int j = 0; j < length; ++j)
      {
        for (int i = 0; i < length; ++i)
        {
          // calc the world coordinates of the 8 corners of the current cell
          // NOTE : currently NO ROTATION
          int wc_index = 0;
          for (int kk = 0; kk <= 1; ++kk)
          {
            for (int jj = 0; jj <= 1; ++jj)
            {
              for (int ii = 0; ii <= 1; ++ii)
              {
                wc_x[wc_index] = static_cast<float>(i+ii)/static_cast<float>(box_width_)*box_size_ + trans_x_;
                wc_y[wc_index] = static_cast<float>(j+jj)/static_cast<float>(box_height_)*box_size_ + trans_y_;
                wc_z[wc_index] = static_cast<float>(k+kk)/static_cast<float>(box_depth_)*box_size_ + trans_z_;
                ++ wc_index; // index increment
              }
            }
          }

          // number of pixels that are not zero valued
          int n_effective_pixels = 0;
          // sum of the values of all non-zero pixels
          float luminance_sum = 0.0f;

          // for each camera
          for (int i_camera = 0; i_camera < num_cameras_; ++i_camera)
          {
            pts.clear();

            // for each corner
            //  calc the projected position on each image/camera
            for (int corner_index = 0; corner_index < 8; ++corner_index)
            {
              // calc the projected pixel on the image of the current camera

              // Set world coordinates position
              Vector4 wc;
              wc.x = wc_x[corner_index];
              wc.y = wc_y[corner_index];
              wc.z = wc_z[corner_index];
              wc.w = 1.0;

              // eye coordinates
              //  [ R | T ] * M
              Vector4 ec;
              ec = camera_extr_paras_[i_camera] * wc;

              // Normalization
              ec.x /= ec.z;
              ec.y /= ec.z;
              ec.z /= ec.z;

              // Projection
              Point2D tmpPt;
              tmpPt.x = camera_intr_paras_[i_camera](0,0) * ec.x + camera_intr_paras_[i_camera](0,2);
              tmpPt.y = camera_intr_paras_[i_camera](1,1) * ec.y + camera_intr_paras_[i_camera](1,2);
              pts.push_back(tmpPt);
            }

            // Calc the effective pixels
            // construct the convex hull
            ConvexHull2D tmpConvexHull(pts);
            float x_min, x_max, y_min, y_max;
            tmpConvexHull.GetBoundingBox(x_min, x_max, y_min, y_max);

            for (int vv = static_cast<int>(y_min); vv < static_cast<int>(y_max+0.5f); ++vv)
            {
              for (int uu = static_cast<int>(x_min); uu < static_cast<int>(x_max+0.5f); ++uu)
              {
                // IMAGE TYPE NOT SPECIFIED YET
                //   USE BMP HERE
                float pix = ground_truth_images_(i_camera, uu, vv);
                if (pix > 0.0f && tmpConvexHull.IfInConvexHull(uu, vv))
                {
                  ++ n_effective_pixels;
                  luminance_sum += pix;
                }
              }
            }

          } // for i_camera

          if (n_effective_pixels != 0)
          {
            int vol_index = index3(i, j, k, length);
            ind_volume[vol_index] = density_vectorized.size();
            density_vectorized.push_back( luminance_sum / static_cast<float>(n_effective_pixels) );
          }
        } // for i
      } // for j
    } // for k

    return true;
  }

} // namespace as_modeling