#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

#include "GradCompute.h"
#include "ASModeling.h"
#include <math/geomath.h>
#include <math/ConvexHull2D.h>

namespace 
{
  void construct_volume_cuda(
    int level,
    float * device_x,
    float * density_vol,
    int *   tag_vol
    );
} // unnamed namespace

namespace as_modeling
{

  ASMGradCompute* ASMGradCompute::instance_ = NULL;

  void ASMGradCompute::grad_compute(ap::real_1d_array& x, double &f, ap::real_1d_array &g)
  {
    float * p_device_x = 0;
    float * p_device_g = 0;
    float * p_host_x = 0;
    float * p_host_g = 0;

    // length of x[] and g[]
    int n = x.gethighbound() - x.getlowbound() + 1;

    // alloc size
    size_t size = n * sizeof(float);

    cutilSafeCall( cudaMalloc<float>(&p_device_g, size) );
    cutilSafeCall( cudaMalloc<float>(&p_device_x, size+1) );
    p_host_g = new float [n];
    p_host_x = new float [n+1];

    // set host_x using x[]
    p_host_x[0] = 0.0f;
    for (int i = 1; i <=n; ++i)
    {
      p_host_x[i] = x(x.getlowbound()+i-1);
    }

    // copy to device x
    cutilSafeCall( cudaMemcpy(p_device_x, p_host_x, size+1, cudaMemcpyHostToDevice) );
    cutilSafeCall( cudaMemset(p_device_g, 0, size) );

    // map gl graphics resource
    cutilSafeCall( cudaGraphicsMapResources(1, &(Instance()->resource_vol_)) );
    cutilSafeCall(
      cudaGraphicsResourceGetMappedPointer( (void**)&(Instance()->d_vol_bufferptr),
      &(Instance()->vol_buffer_num_bytes_), Instance()->resource_vol_)
      );

    // construct volume using x[] and voxel tags
    //construct_volume_cuda(p_device_x);
    //construct_volume_cuda(current_level_, p_device_x, d_tag_volume


    // unmap gl graphics resource after writing-to operation
    cutilSafeCall( cudaGraphicsUnmapResources(1, &Instance()->resource_vol_) );

    


    // copy buffer to texture


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


  bool ASMGradCompute::get_data(int & level, scoped_array<float>& data)
  {
    int length = 1<<level;
    int size = length * length * length;
    float * tmpptr = new float[size];
    data.reset(tmpptr);

    return true;
  }
  bool ASMGradCompute::init(void)
  {
    int max_length = 1<<(p_asmodeling_->MAX_VOL_LEVEL);
    int tex_buffer_size = max_length * max_length * max_length * sizeof(float);

    // OpenGL has been initialized in RenderGL...

    ///////////////////////////////////////////////////////
    //           Init CUDA
    ///////////////////////////////////////////////////////
    cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );

    // create the pixel buffer object
    glGenBuffers(1, &pbo_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, tex_buffer_size, NULL, GL_DYNAMIC_COPY);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); // unbind pbo_

    // register this buffer object with CUDA
    cutilSafeCall(cudaGraphicsGLRegisterBuffer(&resource_vol_, pbo_, cudaGraphicsMapFlagsWriteDiscard));	

    // create gl texture for volume data storage
    glGenTextures(1, &vol_tex_);
    glBindTexture(GL_TEXTURE_3D, vol_tex_);

    // set basic texture parameters
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE32F_ARB, max_length, max_length, max_length, 0, GL_LUMINANCE, GL_FLOAT, NULL);
    CUT_CHECK_ERROR_GL2();

    return true;
  }
  
  bool ASMGradCompute::frame_init(int level, std::list<float>& guess_x)
  {
    // ground truth images have already been loaded...

    current_level_ = level;

    int length = 1<<level;
    int size = length * length *length;

    // set initial guess of x using ground truth image
    set_density_tags(level, h_tag_volume, guess_x, projected_centers_, true);

    // copy data to CUDA
    h_projected_centers = new int [projected_centers_.size()];
    int i = 0;
    for (std::list<int>::const_iterator it = projected_centers_.begin();
      it != projected_centers_.end();
      ++i, ++it)
    {
      h_projected_centers[i] = *it;
    }
    cutilSafeCall( cudaMalloc<int>(&d_tag_volume, sizeof(int)*size) );
    cutilSafeCall( cudaMalloc<int>(&d_projected_centers, sizeof(int)*projected_centers_.size()) );
    cutilSafeCall( cudaMemcpy(d_projected_centers, h_projected_centers, sizeof(int)*projected_centers_.size(), cudaMemcpyHostToDevice));
    cutilSafeCall( cudaMemcpy(d_tag_volume, h_tag_volume, sizeof(int)*size, cudaMemcpyHostToDevice) );
    delete [] h_projected_centers;
    delete [] h_tag_volume;

    return true;
  }


  bool ASMGradCompute::succframe_init(int level)
  {
    // init using previous frame's result
    current_level_ = level;

    std::list<float> dummy_list;

    set_density_tags(level, h_tag_volume, dummy_list, projected_centers_, false);

    // copy data to CUDA

    return true;
  }

  bool ASMGradCompute::level_init(int level)
  {
    current_level_ = level;

    int length = 1<<level;
    int size = length * length *length;

    std::list<float> dummy_list;

    set_density_tags(level, h_tag_volume, dummy_list, projected_centers_, FALSE);

    // copy data to CUDA
    cutilSafeCall( cudaFree(d_tag_volume) );
    cutilSafeCall( cudaFree(d_projected_centers) );

    h_projected_centers = new int [projected_centers_.size()];
    int i = 0;
    for (std::list<int>::const_iterator it = projected_centers_.begin();
      it != projected_centers_.end();
      ++i, ++it)
    {
      h_projected_centers[i] = *it;
    }
    cutilSafeCall( cudaMalloc<int>(&d_tag_volume, sizeof(int)*size) );
    cutilSafeCall( cudaMalloc<int>(&d_projected_centers, sizeof(int)*projected_centers_.size()) );
    cutilSafeCall( cudaMemcpy(d_projected_centers, h_projected_centers, sizeof(int)*projected_centers_.size(), cudaMemcpyHostToDevice));
    cutilSafeCall( cudaMemcpy(d_tag_volume, h_tag_volume, sizeof(int)*size, cudaMemcpyHostToDevice) );
    delete [] h_projected_centers;
    delete [] h_tag_volume;

    return true;
  }

  bool ASMGradCompute::release(void)
  {
    cutilSafeCall(cudaGraphicsUnregisterResource(resource_vol_));
    
    return true;
  }

  void ASMGradCompute::set_density_tags(int level,
                                        int *tag_volume,
                                        std::list<float> &density,
                                        std::list<int> &centers,
                                        bool is_init_density)
  {
    int length = (1<<level);

    float wc_x[9];
    float wc_y[9];
    float wc_z[9];

    wc_x[8] = wc_y[8] = wc_z[8] = 0.0;

    PT2DVEC pts;
    pts.reserve(32);

    tag_volume = new int [length*length*length];

    if (is_init_density)
    {
      density.clear();
      density.push_back(0.0f); // zero density value
    }
    centers.clear();

    memset(tag_volume, 0, sizeof(int)*length*length*length);

    int tag_index = 0;
    // for each cell (i,j,k) 
    for (int k = 0; k < length; ++k)
    {
      for (int j = 0; j < length; ++j)
      {
        for (int i = 0; i < length; ++i)
        {
          // calc the world coordinates of the 8 corners of the current cell
          // NOTE : NO ROTATION
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
                
                wc_x[8] += wc_x[wc_index];
                wc_y[8] += wc_y[wc_index];
                wc_z[8] += wc_z[wc_index];

                ++ wc_index; // index increment
              }
            }
          }
          
          // calc the wc of the cell center
          wc_x[8] /= 8.0;
          wc_y[8] /= 8.0;
          wc_z[8] /= 8.0;

          // number of pixels that are not zero valued
          int n_effective_pixels = 0;
          // sum of the values of all non-zero pixels
          float luminance_sum = 0.0f;

          // for each camera
          for (int i_camera = 0; i_camera < p_asmodeling_->num_cameras_; ++i_camera)
          {
            pts.clear();

            // calc the projected pixel of the current cell 
            Vector4 cwc;
            cwc.x = wc_x[8];
            cwc.y = wc_y[8];
            cwc.z = wc_z[8];
            cwc.w = 1.0;
            Vector4 cec;
            cec = p_asmodeling_->camera_extr_paras_[i_camera] * cwc;
            cec.x /= cec.z;
            cec.y /= cec.z;
            cec.z /= cec.z;
            int px = p_asmodeling_->camera_intr_paras_[i_camera](0,0) * cec.x + p_asmodeling_->camera_intr_paras_[i_camera](0,2);
            int py = p_asmodeling_->camera_intr_paras_[i_camera](1,1) * cec.y + p_asmodeling_->camera_intr_paras_[i_camera](1,2);
            centers.push_back(px);
            centers.push_back(py);

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
            ++ tag_index;
            int vol_index = p_asmodeling_->index3(i, j, k, length);
            tag_volume[vol_index] = tag_index;
            if (is_init_density)
              density.push_back( luminance_sum / static_cast<float>(n_effective_pixels) );
          }
        } // for i
      } // for j
    } // for k

  }

  void ASMGradCompute::set_asmodeling(ASModeling *p)
  {
    p_asmodeling_ = p;
    num_views = p->num_cameras_;
  }


} // namespace as_modeling