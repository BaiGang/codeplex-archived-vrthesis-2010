#include <stdafx.h>
#include <GL/glew.h>

#include "../CudaImageUtil/CudaImgUtilBMP.h"

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

#include "../Utils/Timer/CPUTimer.h"

#include "devfun_def.h"


/////////////////////////////////////////////


namespace as_modeling
{

  ASMGradCompute* ASMGradCompute::instance_ = NULL;

  void ASMGradCompute::grad_compute(const ap::real_1d_array& x, double &f, ap::real_1d_array &g)
  {
    fprintf(stderr, "=== == === == Grad Computing ....\n");
    Timer tmer_1, tmer_2, tmer_3;
    tmer_1.start();

    tmer_2.start();

    f = 0.0;

    // length of x[] and g[]
    int n = x.gethighbound();

    // alloc size
    size_t size = n * sizeof(float);

    // set host_x using x[]
    Instance()->p_host_x[0] = 0.0f;
    for (int i = 1; i <=n; ++i)
    {
      Instance()->p_host_x[i] = static_cast<float>(x(i));
    }

    // copy to device x
    cutilSafeCall( cudaMemcpy(
      Instance()->p_device_x,
      Instance()->p_host_x,
      size+sizeof(float),  // {xi...} plus {0.0}
      cudaMemcpyHostToDevice) );

    // set g[] to zero
    cutilSafeCall( cudaMemset(Instance()->p_device_g, 0, size) );

    // map gl graphics resource
    cutilSafeCall( cudaGraphicsMapResources(1, &(Instance()->resource_vol_)) );

    cutilSafeCall( cudaGraphicsSubResourceGetMappedArray(
      &(Instance()->vol_tex_cudaArray),
      Instance()->resource_vol_,
      0,
      0) );


    // we need to upsample the low-resolution volume
    // to full resolution for rendering
    if (Instance()->current_level_ != ASModeling::MAX_VOL_LEVEL)
    {
      // construct volume using x[] and voxel tags
      construct_volume_cuda(
        Instance()->p_device_x,
        &(Instance()->d_vol_pitchedptr),
        Instance()->vol_extent,
        Instance()->d_tag_volume
        );

      // upsampling
      upsample_volume_cuda(
        Instance()->current_level_,
        ASModeling::MAX_VOL_LEVEL,
        &(Instance()->d_vol_pitchedptr),
        &(Instance()->d_full_vol_pptr)
        );
    }
    else
    {
      // construct volume using x[] and voxel tags
      construct_volume_cuda(
        Instance()->p_device_x,
        &(Instance()->d_full_vol_pptr),
        Instance()->full_vol_extent,
        Instance()->d_tag_volume
        );
    }

    // copy 
    cudaMemcpy3DParms param = {0};
    param.dstArray = Instance()->vol_tex_cudaArray;
    param.srcPtr   = Instance()->d_full_vol_pptr;
    param.extent   = Instance()->vol_cudaArray_extent;
    param.kind     = cudaMemcpyDeviceToDevice;

    cutilSafeCall( cudaMemcpy3D(&param) );

    fprintf(stderr, "--- === --- COPY to GPU used %lf secs.\n", tmer_2.stop());

    tmer_3.start();

    // unmap gl graphics resource after writing-to operation
    cutilSafeCall( cudaGraphicsUnmapResources(1, &Instance()->resource_vol_) );

    // reset the array for f[],
    // this is due to reduction only works for sizes that are power of 2
    int powtwo_length = nearest_pow2( n );
    cutilSafeCall( cudaMemset( Instance()->p_device_x, 0, powtwo_length * sizeof(float)));


    int length = 1 << Instance()->current_level_;

    // calc f and g[]
    for (int i_view = 0; i_view < Instance()->num_views; ++i_view)
    {
      // render to image 1
      Instance()->renderer_->render_unperturbed(i_view, Instance()->vol_tex_, length);

      cutilSafeCall( cudaGraphicsMapResources(1, &(Instance()->resource_rr_)) );

      // get mapped array
      cutilSafeCall( cudaGraphicsSubResourceGetMappedArray(
        &(Instance()->rr_tex_cudaArray),
        Instance()->resource_rr_,
        0, 0) );

      // bind array to cuda tex
      bind_rrtex_cuda(Instance()->rr_tex_cudaArray);

      // launch kernel
      f += calculate_f_cuda(
        Instance()->current_level_,
        i_view,
        Instance()->num_views,
        n,
        powtwo_length,
        Instance()->p_asmodeling_->render_interval_,
        Instance()->d_projected_centers,
        Instance()->d_tag_volume,
        Instance()->p_device_x,
        Instance()->d_temp_f);

      // calc g[]
      for (int pt_slice = 0; pt_slice < 1 << (Instance()->current_level_); ++pt_slice)
      {
        //fprintf(stderr, " --- --- perturbing slice %d...\n", pt_slice);

        for (int pv = 0; pv < Instance()->p_asmodeling_->volume_interval_; ++pv)
        {
          for (int pu = 0; pu < instance_->p_asmodeling_->volume_interval_; ++pu)
          {
            Instance()->renderer_->render_perturbed(i_view, Instance()->vol_tex_, length, pt_slice, pu, pv);

            // accumulate g[]
            cutilSafeCall( cudaGraphicsMapResources(1, &(Instance()->resource_pr_)) );

            // get mapped array
            cutilSafeCall( cudaGraphicsSubResourceGetMappedArray(
              &(Instance()->pr_tex_cudaArray),
              Instance()->resource_pr_,
              0, 0) );

            // bind 2 cuda tex
            bind_prtex_cuda(Instance()->pr_tex_cudaArray);

            //// launch kernel
            calculate_g_cuda(
              Instance()->current_level_,
              i_view,
              Instance()->num_views,
              n,
              Instance()->p_asmodeling_->render_interval_,
              Instance()->d_projected_centers,
              Instance()->d_tag_volume,
              Instance()->p_device_g );

            // unmap resource
            cutilSafeCall( cudaGraphicsUnmapResources(1, &(Instance()->resource_pr_)) );

          } // for pu
        } // for pv
      } // for each slice

      // unmap resource
      cutilSafeCall( cudaGraphicsUnmapResources(1, &(Instance()->resource_rr_)) );
    } // for i_view


    fprintf(stderr, "-- == == -- ++ Calc F and G in this interation used %lf secs.\n", tmer_3.stop());

    // copy g[] from device to host 
    cutilSafeCall( cudaMemcpy(
      Instance()->p_host_g, 
      Instance()->p_device_g,
      n*sizeof(float),
      cudaMemcpyDeviceToHost) );

    // set g[]
    for (int i = 0; i < n; ++i)
    {
      g(i+1) = Instance()->p_host_g[i];
    }

    fprintf(stderr, "=== == === == Grad Computing Used %lf secs.\n", tmer_1.stop());

  } // static grad_compute


  bool ASMGradCompute::get_data(int level, scoped_array<float>& data, ap::real_1d_array &x)
  {
    int length = 1<<level;
    int vol_size = length * length * length;
    float * tmpptr = new float[vol_size];
    data.reset(tmpptr);

    // construct volume
    // length of x[] and g[]
    int n = x.gethighbound();

    // alloc size
    size_t size = n * sizeof(float);

    // set host_x using x[]
    Instance()->p_host_x[0] = 0.0f;
    for (int i = 1; i <=n; ++i)
    {
      Instance()->p_host_x[i] = static_cast<float>(x(i));
    }

    // copy to device x
    cutilSafeCall( cudaMemcpy(
      Instance()->p_device_x,
      Instance()->p_host_x,
      size+sizeof(float),  // {xi...} plus {0.0}
      cudaMemcpyHostToDevice) );

    // construct on cuda
    construct_volume_linm_cuda(length, Instance()->p_device_x, Instance()->d_temp_f, Instance()->d_tag_volume);

    // copy back to HOST
    cutilSafeCall( cudaMemcpy(
      data.get(),
      Instance()->d_temp_f,
      sizeof(float) * vol_size,
      cudaMemcpyDeviceToHost) );

    // set

    return true;
  }


  bool ASMGradCompute::set_ground_truth_images(cuda_imageutil::Image_4c8u& gt_images)
  {
    cudaMemcpy3DParms param = {0};
    param.srcPtr   = make_cudaPitchedPtr(gt_images.GetPixelAt(0,0), 
      p_asmodeling_->width_*sizeof(uchar4),
      p_asmodeling_->width_,
      p_asmodeling_->height_ );

    param.dstArray = gt_tex_cudaArray;
    param.extent   = gt_cudaArray_extent;
    param.kind     = cudaMemcpyHostToDevice;

    cutilSafeCall( cudaMemcpy3D(&param) );
    bind_gttex_cuda( gt_tex_cudaArray );

    return true;
  }

  bool ASMGradCompute::init(void)
  {
    current_level_ = ASModeling::INITIAL_VOL_LEVEL;
    int max_length = ASModeling::MAX_VOL_SIZE;
    int max_size = max_length * max_length * max_length;
    int tex_buffer_size = max_size * sizeof(float);

    // OpenGL and renderer initiation in RenderGL...
    renderer_ = new RenderGL(p_asmodeling_);
    if (!renderer_->init())
      return false;

    ///////////////////////////////////////////////////////
    //           Init CUDA
    ///////////////////////////////////////////////////////
    cudaSetDevice( cutGetMaxGflopsDeviceId() );
    cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );

    // create gl texture for volume data storage
    glGenTextures(1, &vol_tex_);
    glBindTexture(GL_TEXTURE_3D, vol_tex_);

    // set basic texture parameters
    glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE32F_ARB, max_length, max_length, max_length, 0, GL_LUMINANCE, GL_FLOAT, NULL);
    CUT_CHECK_ERROR_GL2();

    // register 3D volume texture with CUDA
    cutilSafeCall( cudaGraphicsGLRegisterImage(
      &resource_vol_,
      vol_tex_,
      GL_TEXTURE_3D,
      cudaGraphicsMapFlagsWriteDiscard) );

    vol_cudaArray_extent = make_cudaExtent(max_length, max_length, max_length);

    // register render target textures with CUDA
    cutilSafeCall( cudaGraphicsGLRegisterImage(
      &resource_rr_,
      renderer_->get_render_result_tex(),
      GL_TEXTURE_2D,
      cudaGraphicsMapFlagsReadOnly) );

    cutilSafeCall( cudaGraphicsGLRegisterImage(
      &resource_pr_,
      renderer_->get_perturb_result_tex(),
      GL_TEXTURE_2D,
      cudaGraphicsMapFlagsReadOnly) );

    //////////////////////////////////////////////////////////
    // alloc memory on CUDA
    //////////////////////////////////////////////////////////

    // NOTE: due to pitching issues, 
    //  shall be allocated for distinct levels
    size_t vol_size = 1 << current_level_;
    vol_extent = make_cudaExtent(vol_size*sizeof(float), vol_size, vol_size);
    cutilSafeCall( cudaMalloc3D(&d_vol_pitchedptr, vol_extent) );

    // full resolution can be allocated once,
    // thanks to its fixed size...
    full_vol_extent = make_cudaExtent(max_length*sizeof(float), max_length, max_length);
    cutilSafeCall( cudaMalloc3D(&d_full_vol_pptr, full_vol_extent) );

    cutilSafeCall( cudaMalloc<int>(&d_projected_centers, 2 * p_asmodeling_->num_cameras_ * max_size) );
    cutilSafeCall( cudaMalloc<int>(&d_tag_volume, max_size) );

    cutilSafeCall( cudaMalloc<float>(&p_device_x, max_size) );
    cutilSafeCall( cudaMalloc<float>(&p_device_g, max_size) );
    cutilSafeCall( cudaMalloc<float>(&d_temp_f,   max_size) );

    // alloc array for ground truth image
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    gt_cudaArray_extent = make_cudaExtent(p_asmodeling_->width_, p_asmodeling_->height_, num_views);
    cutilSafeCall( cudaMalloc3DArray(&gt_tex_cudaArray, &channelDesc, gt_cudaArray_extent) );

    //////////////////////////////////////////////////////////
    // alloc memory on HOST
    //////////////////////////////////////////////////////////
    h_vol_data = new float [max_size];
    h_projected_centers = new int [2 * p_asmodeling_->num_cameras_ * max_size];
    h_tag_volume = new int [ max_size ];

    p_host_g = new float [max_size];
    p_host_x = new float [max_size];

    return true;
  }

  bool ASMGradCompute::release(void)
  {
    fprintf(stderr, " <========>  Releasing ASMGradCompute..\n");

    cutilSafeCall( cudaGraphicsUnregisterResource(resource_vol_));

    cutilSafeCall( cudaFree( d_projected_centers ) );
    cutilSafeCall( cudaFree( d_tag_volume ) );

    cutilSafeCall( cudaFree( d_full_vol_pptr.ptr) );
    cutilSafeCall( cudaFree( d_vol_pitchedptr.ptr) );

    cutilSafeCall( cudaFree( p_device_g ) );
    cutilSafeCall( cudaFree( p_device_x ) );
    cutilSafeCall( cudaFree( d_temp_f ) );

    cutilSafeCall( cudaFreeArray(gt_tex_cudaArray) );

    cutilSafeCall( cudaThreadExit() );

    glDeleteTextures(1, &vol_tex_);

    delete [] h_projected_centers;
    delete [] h_tag_volume;
    delete [] h_vol_data;

    delete [] p_host_g;
    delete [] p_host_x;

    return true;
  }

  bool ASMGradCompute::frame_init(int level, std::list<float>& guess_x)
  {
    // ground truth images have already been loaded...

    fprintf(stderr, " <<====>> Initing frame, level %d\n", level);

    current_level_ = level;

    int length = 1<<level;
    int size = length * length *length;

    // set initial guess of x using ground truth image
    set_density_tags(level, h_tag_volume, guess_x, projected_centers_, true);

    // copy data to CUDA
    int i = 0;
    for (std::list<int>::const_iterator it = projected_centers_.begin();
      it != projected_centers_.end();
      ++i, ++it)
    {
      h_projected_centers[i] = *it;
    }


    cutilSafeCall( cudaMemcpy(
      d_projected_centers,
      h_projected_centers,
      sizeof(int)*projected_centers_.size(),
      cudaMemcpyHostToDevice));

    cutilSafeCall( cudaMemcpy(
      d_tag_volume, 
      h_tag_volume, 
      sizeof(int)*size, 
      cudaMemcpyHostToDevice) );


    return true;
  }


  bool ASMGradCompute::succframe_init(int level, std::list<float>& guess_x, ap::real_1d_array& prev_x)
  {
    // init using previous frame's result
    current_level_ = level;
    int length = 1<<level;
    int size = length * length * length;

    std::list<float> dummy_list;

    set_density_tags(level, h_tag_volume, dummy_list, projected_centers_, false);

    // allocate space for volume data on device
    size_t vol_size = 1 << current_level_;
    vol_extent = make_cudaExtent(vol_size*sizeof(float), vol_size, vol_size);
    cutilSafeCall( cudaMalloc3D(&d_vol_pitchedptr, vol_extent) );

    // copy data to CUDA
    int i = 0;
    for (std::list<int>::const_iterator  it = projected_centers_.begin();
      it != projected_centers_.end();
      ++it, ++i)
    {
      h_projected_centers[i] = *it;
    }

    cutilSafeCall( cudaMemcpy(
      d_projected_centers,
      h_projected_centers,
      sizeof(int)*projected_centers_.size(),
      cudaMemcpyHostToDevice) );
    cutilSafeCall( cudaMemcpy(
      d_tag_volume,
      h_tag_volume,
      sizeof(int)*size,
      cudaMemcpyHostToDevice) );

    return true;
  }


  bool ASMGradCompute::level_init(int level, std::list<float>& guess_x, ap::real_1d_array& prev_x)
  {
    current_level_ = level;

    fprintf(stderr, "  <<++++>> Initing level, %d\n", level);

    int length = 1<<level;
    int size = length * length *length;
    int n = prev_x.gethighbound();

    // first, we construct volume and upsample
    // for previous results
    Instance()->p_host_x[0] = 0.0f;
    for (int i = 1; i <= n; ++i)
    {
      Instance()->p_host_x[i] = prev_x(i);
    }

    // allocate space for volume data on device
    cutilSafeCall( cudaFree(d_vol_pitchedptr.ptr) );
    size_t vol_size = 1 << current_level_;
    vol_extent = make_cudaExtent(vol_size*sizeof(float), vol_size, vol_size);
    cutilSafeCall( cudaMalloc3D(&d_vol_pitchedptr, vol_extent) );

    // copy to device x
    cutilSafeCall( cudaMemcpy(
      Instance()->p_device_x,
      Instance()->p_host_x,
      (1+n)*(sizeof(float)),
      cudaMemcpyHostToDevice ) );

    cutilSafeCall( cudaGetLastError() );

    construct_volume_from_previous_cuda(
      Instance()->p_device_x,
      &(Instance()->d_vol_pitchedptr),
      Instance()->vol_extent,
      Instance()->d_tag_volume );

    cutilSafeCall( cudaGetLastError() );

    // followed calculation of tag volume and projected centers
    std::list<float> dummy_list;
    set_density_tags(level, h_tag_volume, dummy_list, projected_centers_, false);

    // cull the empty cells 
    int n_nonzero_items = projected_centers_.size() / (2*num_views);

    // copy data to CUDA
    int i = 0;
    for (std::list<int>::const_iterator it = projected_centers_.begin();
      it != projected_centers_.end();
      ++i, ++it)
    {
      h_projected_centers[i] = *it;
    }

    cutilSafeCall( cudaMemcpy(d_projected_centers, h_projected_centers, sizeof(int)*projected_centers_.size(), cudaMemcpyHostToDevice));
    cutilSafeCall( cudaMemcpy(d_tag_volume, h_tag_volume, sizeof(int)*size, cudaMemcpyHostToDevice) );

    cull_empty_cells_cuda(
      &(Instance()->d_vol_pitchedptr),
      Instance()->vol_extent,
      Instance()->d_tag_volume );

    // copy back to initiate guess_x
    guess_x.clear();

    get_guess_x_cuda(
      Instance()->p_device_x,
      &(Instance()->d_vol_pitchedptr),
      Instance()->vol_extent,
      Instance()->d_tag_volume );

    cutilSafeCall( cudaMemcpy(
      Instance()->p_host_x,
      Instance()->p_device_x,
      sizeof(float)*(1+n_nonzero_items),
      cudaMemcpyDeviceToHost) );

    guess_x.push_back(0.0f);
    for (int i = 1; i <= n_nonzero_items; ++i)
    {
      guess_x.push_back( Instance()->p_host_x[i] );
    }

    return true;
  }



  void ASMGradCompute::set_density_tags(
    int level,
    int *tag_volume,
    std::list<float> &density,
    std::list<int> &centers,
    bool is_init_density)
  {

    //////////////////////////////////
    /////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////
    cuda_imageutil::BMPImageUtil * debugBMP = new cuda_imageutil::BMPImageUtil[8];
    for (int img=0; img<8;++img)
    {
      debugBMP[img].SetSizes(p_asmodeling_->width_, p_asmodeling_->height_);
      debugBMP[img].ClearImage();
    }
    /////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////
    //////////////////////////////////
#if 0
    ///////////////////////
    //for (int i_view = 0; i_view<num_views; ++i_view)
    //{
    //  for (int kk = 0; kk < 2; ++ kk)
    //  {
    //    for (int jj = 0; jj < 2; ++ jj)
    //    {
    //      for (int ii = 0; ii < 2; ++ii)
    //      {
    //        float x = /*p_asmodeling_->trans_x_ +*/ ii*p_asmodeling_->box_size_;
    //        float y = /*p_asmodeling_->trans_y_ +*/ jj*p_asmodeling_->box_size_;
    //        float z = /*p_asmodeling_->trans_z_ +*/ kk*p_asmodeling_->box_size_;

    //        Vector4 cwc(x,y,z);
    //        Vector4  cec = p_asmodeling_->camera_extr_paras_[i_view] * cwc;

    //        cec.x /= cec.z;
    //        cec.y /= cec.z;
    //        cec.z /= cec.z;

    //        int pu = static_cast<int>( 0.5+
    //          p_asmodeling_->camera_intr_paras_[i_view](0,0)* cec.x + p_asmodeling_->camera_intr_paras_[i_view](0,2) );
    //        int pv = static_cast<int>( 0.5+
    //          p_asmodeling_->camera_intr_paras_[i_view](1,1)* cec.y + p_asmodeling_->camera_intr_paras_[i_view](1,2) );



    //        debugBMP[i_view].GetPixelAt(pu,pv)[0] = kk*255;
    //        debugBMP[i_view].GetPixelAt(pu,pv)[2] = jj*255;
    //        debugBMP[i_view].GetPixelAt(pu,pv)[1] = ii*255;
    //        
    //      }
    //    }
    //  }
    //  char tm[200];
    //  sprintf_s(tm,"../Data/Projected%02d.BMP", i_view);
    //  debugBMP[i_view].SaveImage(tm);
    //}
    //return;
    ///////////////////////
#endif

    int length = (1<<level);

    float wc_x[9];
    float wc_y[9];
    float wc_z[9];

    wc_x[8] = wc_y[8] = wc_z[8] = 0.0;

    PT2DVEC pts;
    pts.reserve(32);


    if (is_init_density)
    {
      density.clear();
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
                wc_x[wc_index] = (static_cast<float>(i+ii)/static_cast<float>(length) - 0.5f)
                  * p_asmodeling_->box_size_ + p_asmodeling_->trans_x_;
                wc_y[wc_index] = (static_cast<float>(j+jj)/static_cast<float>(length) - 0.5f)
                  * p_asmodeling_->box_size_ + p_asmodeling_->trans_y_;
                wc_z[wc_index] = (static_cast<float>(k+kk)/static_cast<float>(length) - 0.5f)
                  * p_asmodeling_->box_size_ + p_asmodeling_->trans_z_;

                wc_x[8] += wc_x[wc_index];
                wc_y[8] += wc_y[wc_index];
                wc_z[8] += wc_z[wc_index];

                ++ wc_index; // index increment
              }
            }
          }

          // calc the wc of the cell center
          wc_x[8] /= 8.0f;
          wc_y[8] /= 8.0f;
          wc_z[8] /= 8.0f;

          // number of pixels that are not zero valued
          int n_effective_pixels = 0;
          // sum of the values of all non-zero pixels
          unsigned int luminance_sum = 0;

          // for each camera
          for (int i_camera = 0; i_camera < p_asmodeling_->num_cameras_; ++i_camera)
          {
            pts.clear();

            int zbase = i_camera*p_asmodeling_->height_;

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
              if (vv<0 || vv>=p_asmodeling_->height_)
                continue;
              for (int uu = static_cast<int>(x_min); uu < static_cast<int>(x_max+0.5f); ++uu)
              {
                if (uu<0 || uu >= p_asmodeling_->width_)
                  continue;
                // Currently only R channel
                unsigned char pix = *(p_asmodeling_->ground_truth_image_.GetPixelAt(uu,vv+zbase));
                if (pix > 0 && tmpConvexHull.IfInConvexHull(uu*1.0f, vv*1.0f))  // "*1.0f" to make the compiler happy
                {
                  ++ n_effective_pixels;
                  luminance_sum += pix;
                }
              }
            }
          } // for i_camera

          // this cell do has some projected matters
          if (n_effective_pixels != 0)
          {
            // set tags
            ++ tag_index;
            int vol_index = p_asmodeling_->index3(i, j, k, length);
            tag_volume[vol_index] = tag_index;

            if (is_init_density)
              density.push_back( static_cast<float>(luminance_sum) / (255.0f*n_effective_pixels) );

            // calc the projected pixels of the current cell 
            for (int i_camera = 0; i_camera < p_asmodeling_->num_cameras_; ++i_camera)
            {
              int zbase = i_camera*p_asmodeling_->height_;
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
              int px = static_cast<int>( 0.5+
                p_asmodeling_->camera_intr_paras_[i_camera](0,0) * cec.x + p_asmodeling_->camera_intr_paras_[i_camera](0,2) );
              int py = static_cast<int>( 0.5+
                p_asmodeling_->camera_intr_paras_[i_camera](1,1) * cec.y + p_asmodeling_->camera_intr_paras_[i_camera](1,2) );

              /////////////////////////////////////////////////////////
              ////////////////////////////////////////////////////////
              debugBMP[i_camera].GetPixelAt(px,py)[0] = 255;
              debugBMP[i_camera].GetPixelAt(px,py)[1] = p_asmodeling_->ground_truth_image_.GetPixelAt(px, py + zbase)[0] + 20;
              //////////////////////////////////////////////////
              /////////////////////////////////////////////////

              centers.push_back(px);
              centers.push_back(py);
            } // for each camera

          } // if (n_effective_pixels != 0)

        } // for i
      } // for j
    } // for k

    /////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////
    for (int img=0; img<8;++img)
    {
      char path_buf[50];
      sprintf_s(path_buf, 50, "../Data/test%02d.bmp", img);
      debugBMP[img].SaveImage(path_buf);
    }
    /////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////
  }

  bool ASMGradCompute::set_asmodeling(ASModeling *p)
  {
    if (NULL == p)
      return false;
    p_asmodeling_ = p;
    num_views = p->num_cameras_;
    return true;
  }


} // namespace as_modeling

