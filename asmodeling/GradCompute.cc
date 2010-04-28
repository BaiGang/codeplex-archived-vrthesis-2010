#include <GL/glew.h>

#include "GradCompute.h"
#include "ASModeling.h"
#include "../Utils/math/geomath.h"
#include "../Utils/math/ConvexHull2D.h"

namespace as_modeling
{
  void ASMGradCompute::grad_compute(ap::real_1d_array x, double &f, ap::real_1d_array &g)
  {
    float * p_device_x = 0;
    float * p_device_g = 0;
    float * p_host_x = 0;
    float * p_host_g = 0;

    // length of x[] and g[]
    int n = x.gethighbound() - x.getlowbound() + 1;

    // alloc size
    size_t size = n * sizeof(float);

    cudaMalloc<float>(&p_device_g, size);
    cudaMalloc<float>(&p_device_x, size+1);
    p_host_g = new float [n];
    p_host_x = new float [n+1];

    // set host_x using x[]
    p_host_x[0] = 0.0f;
    for (int i = 1; i <=n; ++i)
    {
      p_host_x[i] = x(x.getlowbound()+i-1);
    }

    cudaMemcpy(p_device_x, p_host_x, size+1, cudaMemcpyHostToDevice);
    cudaMemset(p_device_g, 0, size);

    // construct volume using x[] and voxel tags
    //construct_volume_cuda(p_device_x);

    // render to image 1


    // calc f

    // perturb voxel

    // render to image 2

    // calc g[]

    // copy g[] from device to host 
    cudaMemcpy(p_host_g, p_device_g, size, cudaMemcpyDeviceToHost);

    // set g[]
    for (int i = 0; i < n; ++i)
    {
      g(i+g.getlowbound()) = p_host_g[i];
    }

    cudaFree(p_device_x);
    cudaFree(p_device_g);
    delete [] p_host_g;
    delete [] p_host_x;
  } // 

  
  void ASMGradCompute::init(int level, 
                            std::list<float>& guess_x,
                            std::list<int>& projection_center,
                            int * tag_vol)
  {
    // init gl texture for volume data storage
    glGenTextures(1, &(p_asmodeling_->volume_texture_id_));
    glBindTexture(GL_TEXTURE_3D, p_asmodeling_->volume_texture_id_);

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE32F_ARB,
      p_asmodeling_->box_width_,
      p_asmodeling_->box_height_,
      p_asmodeling_->box_depth_,
      0, GL_LUMINANCE, GL_FLOAT, NULL);

    // then init cuda graphics resources
    cudaGLSetGLDevice(0);

    cudaGraphicsGLRegisterImage(&resource_vol_,
      p_asmodeling_->volume_texture_id_, 
      GL_TEXTURE_3D,
      cudaGraphicsMapFlagsWriteDiscard);

    // calc an ininial guess of x
    int length = (1<<level);

    float wc_x[8];
    float wc_y[8];
    float wc_z[8];

    PT2DVEC pts;
    pts.reserve(16);

    guess_x.clear();
    guess_x.push_back(0.0f); // zero density value

    if (tag_vol)
    {
      delete [] tag_vol;
    }
    tag_vol = new int [length*length*length];
    memset(tag_vol, 0, sizeof(int)*length*length*length);

    // for each cell (i, j, k)
    for (int k = 0; k < length; ++k)
    {
      for (int j = 0; j < length; ++j)
      {
        for (int i = 0; i < length; ++i)
        {
          int wc_index = 0;
        } // for i
      } // for j
    } // for k
  }
  

  
  void ASMGradCompute::init_current_level(int level)
  {

  }

  void ASMGradCompute::release( )
  {
    cudaGraphicsUnregisterResource(resource_vol_);
    //cudaFree(
  }

  void ASMGradCompute::set_density_tags(int level, int *tag_volume, std::list<float> &density, bool is_init_density)
  {

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
                wc_x[wc_index] = static_cast<float>(i+ii)/static_cast<float>(p_asmodeling_->box_width_)*p_asmodeling_->box_size_ + p_asmodeling_->trans_x_;
                wc_y[wc_index] = static_cast<float>(j+jj)/static_cast<float>(p_asmodeling_->box_height_)*p_asmodeling_->box_size_ + p_asmodeling_->trans_y_;
                wc_z[wc_index] = static_cast<float>(k+kk)/static_cast<float>(p_asmodeling_->box_depth_)*p_asmodeling_->box_size_ + p_asmodeling_->trans_z_;
                ++ wc_index; // index increment
              }
            }
          }

          // number of pixels that are not zero valued
          int n_effective_pixels = 0;
          // sum of the values of all non-zero pixels
          float luminance_sum = 0.0f;

          // for each camera
          for (int i_camera = 0; i_camera < p_asmodeling_->num_cameras_; ++i_camera)
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
              ec = p_asmodeling_->camera_extr_paras_[i_camera] * wc;

              // Normalization
              ec.x /= ec.z;
              ec.y /= ec.z;
              ec.z /= ec.z;

              // Projection
              Point2D tmpPt;
              tmpPt.x = p_asmodeling_->camera_intr_paras_[i_camera](0,0) * ec.x + p_asmodeling_->camera_intr_paras_[i_camera](0,2);
              tmpPt.y = p_asmodeling_->camera_intr_paras_[i_camera](1,1) * ec.y + p_asmodeling_->camera_intr_paras_[i_camera](1,2);
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
                float pix = p_asmodeling_->ground_truth_images_(i_camera, uu, vv);
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
            int vol_index = p_asmodeling_->index3(i, j, k, length);
            tag_volume[vol_index] = density.size();
            if (is_init_density)
              density.push_back( luminance_sum / static_cast<float>(n_effective_pixels) );
          }
        } // for i
      } // for j
    } // for k

  }

} // namespace as_modeling