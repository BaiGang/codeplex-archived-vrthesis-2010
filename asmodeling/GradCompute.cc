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
#include "../Utils/image/PFMImage.h"


#include "devfun_def.h"


/////////////////////////////////////////////
//#define __DEBUG_IMAGE_


namespace as_modeling
{

  ASMGradCompute* ASMGradCompute::instance_ = NULL;

  void ASMGradCompute::grad_compute(const ap::real_1d_array& x, double &f, ap::real_1d_array &g)
  {
    fprintf(stderr, "=== == === == Grad Computing ....\n");

    Timer tmer_1, tmer_2, tmer_3;
    tmer_1.start();

    f = 0.0;

    // length of x[] and g[]
    int n = x.gethighbound();

    fprintf(stderr, " %d variables to optimize...\n", n);

    // alloc size
    size_t size = n * sizeof(float);

    // set host_x using x[]
    Instance()->p_host_x[0] = 0.0f;
    for (int i = 1; i <=n; ++i)
    {
      Instance()->p_host_x[i] = static_cast<float>(x(i));
    }

    //////////////////////
    cudaEventRecord(Instance()->event_start_);

    // copy to device x
    cutilSafeCall( cudaMemcpy(
      Instance()->p_device_x,
      Instance()->p_host_x,
      size+sizeof(float),  // {xi...} plus {0.0}
      cudaMemcpyHostToDevice) );

    // set g[] to zero
    cutilSafeCall( cudaMemset(Instance()->p_device_g, 0, size+sizeof(float)) );

    // we need to upsample the low-resolution volume
    // to full resolution for rendering
    if (Instance()->current_level_ != Instance()->p_asmodeling_->max_vol_level_)
    {
      // construct volume using x[] and voxel tags
      construct_volume_cuda(
        Instance()->p_device_x,
        &(Instance()->d_vol_pitchedptr),
        Instance()->vol_extent
        );

      // upsampling
      upsample_volume_cuda(
        Instance()->current_level_,
        Instance()->p_asmodeling_->max_vol_level_,
        &(Instance()->d_vol_pitchedptr),
        &(Instance()->d_full_vol_pptr)
        );
      cutilSafeCall( cudaGetLastError() );
    }
    else
    {
      // construct volume using x[] and voxel tags
      construct_volume_cuda(
        Instance()->p_device_x,
        &(Instance()->d_full_vol_pptr),
        Instance()->full_vol_extent
        );
      cutilSafeCall( cudaGetLastError() );
    }



    // map gl graphics resource
    cutilSafeCall( cudaGraphicsMapResources(1, &(Instance()->resource_vol_)) );

    cutilSafeCall( cudaGraphicsSubResourceGetMappedArray(
      &(Instance()->vol_tex_cudaArray),
      Instance()->resource_vol_,
      0,
      0) );

    // copy 
    cudaMemcpy3DParms param = {0};
    param.dstArray = Instance()->vol_tex_cudaArray;
    param.srcPtr   = Instance()->d_full_vol_pptr;
    param.extent   = Instance()->vol_cudaArray_extent;
    param.kind     = cudaMemcpyDeviceToDevice;
    cutilSafeCall( cudaMemcpy3D(&param) );

    // unmap gl graphics resource after writing-to operation
    cutilSafeCall( cudaGraphicsUnmapResources(1, &Instance()->resource_vol_) );

    // //////////////////////////////////
    cudaEventSynchronize(Instance()->event_stop_);
    float copytime;
    cudaEventElapsedTime(&copytime, Instance()->event_start_, Instance()->event_stop_);
    fprintf(stderr, "CUDA EVENT PROFILING ==== set up vol tex used %f secs.\n", copytime * 0.001f);

    tmer_3.start();


    // reset the array for f[],
    // this is due to reduction only works for sizes that are power of 2
    int powtwo_length = nearest_pow2( n+1 );

    int length = 1 << Instance()->current_level_;

    cutilSafeCall( cudaMemset( Instance()->p_device_x, 0, powtwo_length * sizeof(float)));


    // perturbed tile size
    int mm = Instance()->p_asmodeling_->volume_interval_array_[Instance()->current_level_];
    int p_range = Instance()->p_asmodeling_->render_interval_array_[Instance()->current_level_];

    // calc f and g[]
    for (int i_view = 0; i_view < Instance()->num_views; ++i_view)
    {
      ///////////////////
      cudaEventRecord(Instance()->event_start_);

      // render to image 
      Instance()->renderer_->render_unperturbed(i_view, Instance()->vol_tex_, length);

      cutilSafeCall( cudaGraphicsMapResources(1, &(Instance()->resource_rr_)) );

      // get mapped array
      cutilSafeCall( cudaGraphicsSubResourceGetMappedArray(
        &(Instance()->rr_tex_cudaArray),
        Instance()->resource_rr_,
        0, 0) );

      // bind array to cuda tex
      bind_rrtex_cuda(Instance()->rr_tex_cudaArray);

      //////////////////////
      //cudaEventSynchronize(Instance()->event_stop_);
      //cudaEventElapsedTime(&copytime, Instance()->event_start_, Instance()->event_stop_);
      //fprintf(stderr, "CUDA EVENT PROFILING ==== render and bind tex used %f secs.\n", copytime * 0.001f);

      /////////////////////
      //cudaEventRecord(Instance()->event_start_);

      // Lauch kernel
      float ff = calculate_f_compact_cuda(
        i_view,
        Instance()->p_asmodeling_->height_,
        p_range,
        n + 1,
        Instance()->p_device_x,
        Instance()->d_temp_f,
        Instance()->scanplan_
        );
      ////////////////////
      cudaEventSynchronize(Instance()->event_stop_);
      cudaEventElapsedTime(&copytime, Instance()->event_start_, Instance()->event_stop_);
      fprintf(stderr, "CUDA EVENT PROFILING ==== calc f used %f secs.\n", copytime * 0.001f);

      fprintf(stderr, "++ ++ ++ F value of view %d is %f\n", i_view, ff);
      f += ff;
      //f += 1.0f;

      ///////////////////
      cudaEventRecord(Instance()->event_start_);

      // calc g[]
      for (int pt_slice = 0; pt_slice < length; ++pt_slice)
      {
        for (int pv = 0; pv < mm; ++pv)
        {
          for (int pu = 0; pu < mm; ++pu)
          {


            Instance()->renderer_->render_perturbed(
              i_view,                  // view
              Instance()->vol_tex_,    // volume texture 
              length,                  // resolution of the delegate box
              mm,                      // perturb group size (interval between perturbed pixel)
              pt_slice, pu, pv);       // specify perturbed group

            // accumulate g[]
            cutilSafeCall( cudaGraphicsMapResources(1, &(Instance()->resource_pr_)) );

            // get mapped array
            cutilSafeCall( cudaGraphicsSubResourceGetMappedArray(
              &(Instance()->pr_tex_cudaArray),
              Instance()->resource_pr_,
              0, 0) );

            // bind 2 cuda tex
            bind_prtex_cuda(Instance()->pr_tex_cudaArray);

            Timer a;
            a.start();
            calculate_g_cuda(
              Instance()->current_level_,
              Instance()->p_asmodeling_->height_,
              i_view,
              p_range,
              mm,
              pu,
              pv,
              Instance()->p_asmodeling_->camera_orientations_[i_view],
              pt_slice,
              Instance()->p_device_g
              );
            //fprintf(stderr, "\t\tCalcG\t%d\t%d\t%d\t%lf\n", pt_slice, pu, pv, a.stop());

            // unmap resource
            cutilSafeCall( cudaGraphicsUnmapResources(1, &(Instance()->resource_pr_)) );


          } // for pu
        } // for pv
      } // for each slice

      ////////////////////
      cudaEventSynchronize(Instance()->event_stop_);
      cudaEventElapsedTime(&copytime, Instance()->event_start_, Instance()->event_stop_);
      fprintf(stderr, "CUDA EVENT PROFILING ==== calc g (each pass) used %f secs.\n", copytime * 0.001f);

      // unmap resource
      cutilSafeCall( cudaGraphicsUnmapResources(1, &(Instance()->resource_rr_)) );
    } // for i_view


    fprintf(stderr, "==+++++=== F value : %f\n", f);

    fprintf(stderr, "-- == == -- ++ Calc F and G in this interation used %lf secs.\n", tmer_3.stop());

    // copy g[] from device to host 
    cutilSafeCall( cudaMemcpy(
      Instance()->p_host_g, 
      Instance()->p_device_g,
      (n+1)*sizeof(float),
      cudaMemcpyDeviceToHost) );

    // set g[]
    for (int i = 1; i <= n; ++i)
    {
      g(i) = Instance()->p_host_g[i];
    }

    fprintf(stderr, "=== == === == Grad Computing Used %lf secs.\n", tmer_1.stop());

  } // static grad_compute


  bool ASMGradCompute::get_data(int i_frame, int level, scoped_array<float>& data, ap::real_1d_array &x)
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
    construct_volume_linm_cuda(length, Instance()->p_device_x, Instance()->d_temp_f);

    // copy back to HOST
    cutilSafeCall( cudaMemcpy(
      data.get(),
      Instance()->d_temp_f,
      sizeof(float) * vol_size,
      cudaMemcpyDeviceToHost) );

    // show render result
    if (Instance()->current_level_ != Instance()->p_asmodeling_->max_vol_level_)
    {
      // construct volume using x[] and voxel tags
      construct_volume_cuda(
        Instance()->p_device_x,
        &(Instance()->d_vol_pitchedptr),
        Instance()->vol_extent
        );

      // upsampling
      upsample_volume_cuda(
        Instance()->current_level_,
        Instance()->p_asmodeling_->max_vol_level_,
        &(Instance()->d_vol_pitchedptr),
        &(Instance()->d_full_vol_pptr)
        );
      cutilSafeCall( cudaGetLastError() );

    }
    else
    {
      // construct volume using x[] and voxel tags
      construct_volume_cuda(
        Instance()->p_device_x,
        &(Instance()->d_full_vol_pptr),
        Instance()->full_vol_extent
        );
      cutilSafeCall( cudaGetLastError() );
    }
    // map gl graphics resource
    cutilSafeCall( cudaGraphicsMapResources(1, &(Instance()->resource_vol_)) );

    cutilSafeCall( cudaGraphicsSubResourceGetMappedArray(
      &(Instance()->vol_tex_cudaArray),
      Instance()->resource_vol_,
      0,
      0) );

    // copy 
    cudaMemcpy3DParms param = {0};
    param.dstArray = Instance()->vol_tex_cudaArray;
    param.srcPtr   = Instance()->d_full_vol_pptr;
    param.extent   = Instance()->vol_cudaArray_extent;
    param.kind     = cudaMemcpyDeviceToDevice;
    cutilSafeCall( cudaMemcpy3D(&param) );

    // unmap gl graphics resource after writing-to operation
    cutilSafeCall( cudaGraphicsUnmapResources(1, &Instance()->resource_vol_) );

    for (int i_view = 0; i_view < num_views; ++i_view)
    {
      renderer_->render_unperturbed(i_view, vol_tex_, 1 << current_level_);
      char path_buf[100];
      sprintf(path_buf, "../Data/Results/Frame%08d_View%02d_Level%d.PFM", i_frame, i_view, level);
      float * data = renderer_->get_render_res();
      float * img = new float [p_asmodeling_->width_*p_asmodeling_->height_];
      for (int y = 0; y < p_asmodeling_->height_; ++y)
      {
        for (int x = 0; x < p_asmodeling_->width_; ++x)
        {
          img[y*p_asmodeling_->width_+x] = data[4*(y*p_asmodeling_->width_+x)];
        }
      }
      PFMImage *sndipfm = new PFMImage(p_asmodeling_->width_,
        p_asmodeling_->height_,
        0, img);
      sndipfm->WriteImage(path_buf);
    }
    float * imgdata = new float [length * length*length];
    PFMImage * pfmhaha = new PFMImage(length, length*length, 0, imgdata);
    char pathbuf [100];
    sprintf(pathbuf, "../Data/Results/Frame%08d_Result_Level%d.PFM", i_frame, level);
    pfmhaha->WriteImage(pathbuf);
    delete pfmhaha;
    // set over

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
    current_level_ = Instance()->p_asmodeling_->initial_vol_level_;
    int max_length = Instance()->p_asmodeling_->max_vol_size_;
    int max_size = max_length * max_length * max_length;

    // sub size means, the non-empty voxels are typically a proportion of the whole volume
    int sub_size = 0.6 * max_size;

    int tex_buffer_size = max_size * sizeof(float);

    // OpenGL and renderer initiation in RenderGL...
    renderer_ = new RenderGL(p_asmodeling_);
    if (!renderer_->init())
      return false;

    ///////////////////////////////////////////////////////
    //           Init CUDA
    ///////////////////////////////////////////////////////
    cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );

    // warm up
    WarmUp();


    // create gl texture for volume data storage
    glGenTextures(1, &vol_tex_);
    glBindTexture(GL_TEXTURE_3D, vol_tex_);

    // set basic texture parameters
    glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

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

    cutilSafeCall( cudaMalloc((void**)(&p_device_x), sub_size*sizeof(float)) );

    cutilSafeCall( cudaMalloc((void**)(&p_device_g), sub_size * sizeof(float)) );

    cutilSafeCall( cudaMalloc((void**)(&d_temp_f), sub_size * sizeof(float)) );

    // alloc array for ground truth image
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    gt_cudaArray_extent = make_cudaExtent(p_asmodeling_->width_, p_asmodeling_->height_, num_views);
    cutilSafeCall( cudaMalloc3DArray(&gt_tex_cudaArray, &channelDesc, gt_cudaArray_extent) );

    //////////////////////////////////////////////////////////
    // alloc memory on HOST
    //////////////////////////////////////////////////////////
    h_projected_centers = new uint16 [2 * p_asmodeling_->num_cameras_ * sub_size];
    h_tag_volume = new int [ max_size ];

    p_host_g = new float [sub_size];
    p_host_x = new float [sub_size];

    // create event
    cudaEventCreate(&event_start_);
    cudaEventCreate(&event_stop_);

    return true;
  }

  bool ASMGradCompute::release(void)
  {
    fprintf(stderr, " <========>  Releasing ASMGradCompute..\n");

    cutilSafeCall( cudaThreadExit() );

    cutilSafeCall( cudaGraphicsUnregisterResource(resource_vol_));

    cudaEventDestroy(event_start_);
    cudaEventDestroy(event_stop_);

    cutilSafeCall( cudaFree( d_full_vol_pptr.ptr) );
    cutilSafeCall( cudaFree( d_vol_pitchedptr.ptr) );

    cutilSafeCall( cudaFree( p_device_g ) );
    cutilSafeCall( cudaFree( p_device_x ) );
    cutilSafeCall( cudaFree( d_temp_f ) );

    cutilSafeCall( cudaFreeArray(gt_tex_cudaArray) );

    cutilSafeCall( cudaFreeArray(pos_tag_cudaArray) );

    cutilSafeCall( cudaFreeArray(pcenters_cudaArray) );


    glDeleteTextures(1, &vol_tex_);

    delete renderer_;

    delete [] h_projected_centers;
    delete [] h_tag_volume;

    delete [] p_host_g;
    delete [] p_host_x;

    return true;
  }

  bool ASMGradCompute::frame_init(int level, std::vector<float>& guess_x)
  {
    // ground truth images have already been loaded...

    fprintf(stderr, " <<====>> Initing frame, level %d\n", level);

    current_level_ = level;

    int length = 1<<level;
    int size = length * length *length;

    // set initial guess of x using ground truth image
    set_density_tags(level, h_tag_volume, guess_x, projected_centers_, true);

    int num_items = projected_centers_.size() / (2 * num_views) + 1;

    //
    // allocate space for the cudaArray 
    //
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
    pos_tag_extent = make_cudaExtent(length, length, length);
    cudaMalloc3DArray( &pos_tag_cudaArray, &desc, pos_tag_extent);

    cudaChannelFormatDesc desc2 = cudaCreateChannelDesc<ushort2>();
    pcenters_extent = make_cudaExtent(512, num_items / 512 + ((num_items%512)?1:0), num_views);
    cudaMalloc3DArray( &pcenters_cudaArray, &desc2, pcenters_extent);

    ////////FILE * fp = fopen("../Data/test_pcenters.txt", "w");
    ////////for (int i = 0; i < num_items - 1; ++i)
    ////////{
    ////////  for (int j = 0; j < num_views; ++j)
    ////////  {
    ////////    fprintf(fp, "%d %d\t", projected_centers_[i*2*num_views + j*2],
    ////////      projected_centers_[i*2*num_views + j*2 + 1]);
    ////////  }
    ////////  fprintf(fp ,"\n");
    ////////}
    ////////fclose(fp);

    //
    // set projected center tex
    //
    for (int i_camera = 0; i_camera < num_views; ++i_camera)
    {
      h_projected_centers[0] = 0;
      h_projected_centers[1] = 0;
      for (int i_item = 1; i_item < num_items; ++i_item)
      {
        h_projected_centers[i_item*2] = projected_centers_[(i_item-1)*2*num_views + i_camera*2];
        h_projected_centers[i_item*2+1] = projected_centers_[(i_item-1)*2*num_views + i_camera*2+1];
      }


      ///////////////////////////////////////////////////////////////////////////////////////////////////
      //float * data = new float [p_asmodeling_->width_ * p_asmodeling_->height_];
      //memset(data, 0, sizeof(float)*p_asmodeling_->width_ * p_asmodeling_->height_);
      //for (int i = 0; i < num_items; ++i)
      //{
      //  int u = h_projected_centers[i*2];
      //  int v = h_projected_centers[i*2+1];
      //  data[v * p_asmodeling_->width_ + u] = 1.0f;
      //}
      //PFMImage * fm1 = new PFMImage(p_asmodeling_->width_, p_asmodeling_->height_, 0, data);
      //char path_buf[100];
      //sprintf(path_buf, "../Data/View%d.pfm", i_camera);
      //fm1->WriteImage(path_buf);
      //delete fm1;
      //delete [] data;
      ////////////////////////////////////////////////////////////////////////////////////////////////////

      cutilSafeCall( cudaGetLastError());

      cudaMemcpy3DParms param = {0};
      param.dstArray = pcenters_cudaArray;
      param.dstPos = make_cudaPos(0, 0, i_camera);
      param.srcPtr = make_cudaPitchedPtr( (void*)h_projected_centers, 512 * sizeof(ushort2), 512, num_items/512+((num_items%512)?1:0));
      param.extent = make_cudaExtent(pcenters_extent.width, pcenters_extent.height, 1);
      param.kind = cudaMemcpyHostToDevice;

      cutilSafeCall( cudaMemcpy3D(&param) );
    } // for i_camera



    // 
    // set position tag volume tex
    //
    cudaMemcpy3DParms param = {0};
    param.dstArray = pos_tag_cudaArray;
    param.extent = pos_tag_extent;
    param.srcPtr = make_cudaPitchedPtr( (void*)h_tag_volume, length * sizeof(int), length, length);
    param.kind = cudaMemcpyHostToDevice;

    cutilSafeCall( cudaMemcpy3D( &param) );


    //
    // Bind texture reference
    //
    bind_pcenters_cuda(pcenters_cudaArray);
    bind_postags_cuda(pos_tag_cudaArray);

    //// debug ///////////////////////////////////////
    //// out put projected centers
    //for (int i_view = 0; i_view < num_views; ++i_view)
    //{
    //  float * testpc = new float[p_asmodeling_->width_ * p_asmodeling_->height_];
    //  test_ProjectedCenters(i_view, p_asmodeling_->width_, p_asmodeling_->height_, testpc, pcenters_extent);
    //  PFMImage tmpis(p_asmodeling_->width_, p_asmodeling_->height_, 0, testpc);
    //  char pathbuff[100];
    //  sprintf(pathbuff, "../Data/GPU_View%d.pfm", i_view);
    //  tmpis.WriteImage(pathbuff);
    //  delete[] testpc;
    //}

    // set the renderer for current level
    renderer_->level_init(current_level_, vol_tex_);

    //// testing.. ////////////////////////////////////////////
    //for (int i = 0; i<num_views; ++i)
    //{
    //  cutilSafeCall( cudaGetLastError() );
    //  float * data = new float [p_asmodeling_->width_ * p_asmodeling_->height_ * sizeof(float)];

    //  fprintf(stderr, "Test GT for camera %d\n", i);
    //  test_GroundTruth(i, p_asmodeling_->width_, p_asmodeling_->height_, data);

    //  cutilSafeCall( cudaThreadSynchronize() );

    //  cutilSafeCall( cudaGetLastError() );

    //  PFMImage * ddd = new PFMImage(p_asmodeling_->width_, p_asmodeling_->height_, 0, data);
    //  char buf[200];
    //  sprintf(buf, "../Data/Results/GT%03d.pfm", i);
    //  ddd->WriteImage(buf);
    //  delete ddd;
    //  delete [] data;
    //}

    //printf("Haha : \n");
    //int aaa;
    //scanf("%d", &aaa);

    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_FLOAT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_EXCLUSIVE;
    
    scanplan_ = 0;
    CUDPPResult result = cudppPlan(&scanplan_, config, num_items, 1, 0);  

    if (CUDPP_SUCCESS != result)
    {
        fprintf(stderr, "Error creating CUDPPPlan\n");
        return false;
    }

    return true;
  }


  // using previous results
  // first downsample previous results
  // then upsample to set current guess
  // then feed back guess_x
  bool ASMGradCompute::succframe_init(int level, std::vector<float>& guess_x, ap::real_1d_array& prev_x)
  {
    // init using previous frame's result
    current_level_ = level;
    int length = 1<<level;
    int size = length * length * length;

    //////////////================
    struct cudaPitchedPtr tmp_vol;
    cudaExtent tmp_extent = make_cudaExtent(length/2*sizeof(float), length/2, length/2);
    cutilSafeCall( cudaMalloc3D(&tmp_vol, tmp_extent) );

    // downsample current result to tmp_vol
    memset(p_host_x, 0, prev_x.gethighbound()*sizeof(float));


    // upsample tmp_vol to vol_ptr

    // cull out empty voxels

    cudaFree(tmp_vol.ptr);
    //////////////================

    std::vector<float> dummy_list;
    set_density_tags(level, h_tag_volume, dummy_list, projected_centers_, false);

    int num_items = projected_centers_.size() / (2 * num_views) + 1;

    // allocate space for volume data on device
    size_t vol_size = 1 << current_level_;
    vol_extent = make_cudaExtent(vol_size*sizeof(float), vol_size, vol_size);
    cutilSafeCall( cudaMalloc3D(&d_vol_pitchedptr, vol_extent) );

    //
    // set projected center tex
    //
    for (int i_camera = 0; i_camera < num_views; ++i_camera)
    {
      h_projected_centers[0] = 0;
      h_projected_centers[1] = 0;
      for (int i_item = 1; i_item < num_items; ++i_item)
      {
        h_projected_centers[i_item*2] = projected_centers_[(i_item-1)*2*num_views + i_camera*2];
        h_projected_centers[i_item*2+1] = projected_centers_[(i_item-1)*2*num_views + i_camera*2+1];
      }

      cudaMemcpy3DParms param = {0};
      param.dstArray = pcenters_cudaArray;
      param.dstPos = make_cudaPos(0, 0, i_camera);
      param.srcPtr = make_cudaPitchedPtr( (void*)h_projected_centers, 512 * sizeof(ushort2), 512, num_items/512+((num_items%512)?1:0));
      param.extent = make_cudaExtent(pcenters_extent.width, pcenters_extent.height, 1);
      param.kind = cudaMemcpyHostToDevice;

      cutilSafeCall( cudaMemcpy3D(&param) );
    }

    // 
    // set position tag volume tex
    //
    cudaMemcpy3DParms param = {0};
    param.dstArray = pos_tag_cudaArray;
    param.extent = pos_tag_extent;
    param.srcPtr = make_cudaPitchedPtr( (void*)h_tag_volume, length * sizeof(int), length, length);
    param.kind = cudaMemcpyHostToDevice;

    cutilSafeCall( cudaMemcpy3D( &param) );

    //
    // Bind texture reference
    //
    bind_pcenters_cuda(pcenters_cudaArray);
    bind_postags_cuda(pos_tag_cudaArray);

    // cudpp for prefix summation
    CUDPPResult result = cudppDestroyPlan(scanplan_);
    if (CUDPP_SUCCESS != result)
    {
      printf("Error destroying CUDPPPlan\n");
      exit(-1);
    }

    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_FLOAT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_EXCLUSIVE;

    scanplan_ = 0;
    result = cudppPlan(&scanplan_, config, num_items, 1, 0);  

    if (CUDPP_SUCCESS != result)
    {
      fprintf(stderr, "Error creating CUDPPPlan\n");
      return false;
    }

    return true;
  }


  bool ASMGradCompute::level_init(int level, std::vector<float>& guess_x, ap::real_1d_array& prev_x)
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
      Instance()->vol_extent );

    cutilSafeCall( cudaGetLastError() );

    // followed calculation of tag volume and projected centers
    std::vector<float> dummy_list;
    set_density_tags(level, h_tag_volume, dummy_list, projected_centers_, false);

    int num_items = projected_centers_.size() / (2 * num_views) + 1;

    //
    // re-allocate space for density tag and projected centers
    //
    cutilSafeCall( cudaFreeArray( pcenters_cudaArray ) );
    cutilSafeCall( cudaFreeArray( pos_tag_cudaArray ) );

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
    pos_tag_extent = make_cudaExtent(length, length, length);
    cudaMalloc3DArray( &pos_tag_cudaArray, &desc, pos_tag_extent);

    cudaChannelFormatDesc desc2 = cudaCreateChannelDesc<ushort2>();
    pcenters_extent = make_cudaExtent(512, num_items / 512 + ((num_items%512)?1:0), num_views);
    cudaMalloc3DArray( &pcenters_cudaArray, &desc2, pcenters_extent);

    //
    // set projected center tex
    //
    for (int i_camera = 0; i_camera < num_views; ++i_camera)
    {
      h_projected_centers[0] = 0;
      h_projected_centers[1] = 0;
      for (int i_item = 1; i_item < num_items; ++i_item)
      {
        h_projected_centers[i_item*2] = projected_centers_[(i_item-1)*2*num_views + i_camera*2];
        h_projected_centers[i_item*2+1] = projected_centers_[(i_item-1)*2*num_views + i_camera*2+1];
      }

      cudaMemcpy3DParms param = {0};
      param.dstArray = pcenters_cudaArray;
      param.dstPos = make_cudaPos(0, 0, i_camera);
      param.srcPtr = make_cudaPitchedPtr( (void*)h_projected_centers, 512 * sizeof(ushort2), 512, num_items/512+((num_items%512)?1:0));
      param.extent = make_cudaExtent(pcenters_extent.width, pcenters_extent.height, 1);
      param.kind = cudaMemcpyHostToDevice;

      cutilSafeCall( cudaMemcpy3D(&param) );
    }


    // 
    // set position tag volume tex
    //
    cudaMemcpy3DParms param = {0};
    param.dstArray = pos_tag_cudaArray;
    param.extent = pos_tag_extent;
    param.srcPtr = make_cudaPitchedPtr( (void*)h_tag_volume, length * sizeof(int), length, length);
    param.kind = cudaMemcpyHostToDevice;

    cutilSafeCall( cudaMemcpy3D( &param) );

    //
    // Bind texture reference
    //
    bind_pcenters_cuda(pcenters_cudaArray);
    bind_postags_cuda(pos_tag_cudaArray);

    //cutilSafeCall( cudaMemcpy(
    //  d_tag_volume, 
    //  h_tag_volume, 
    //  sizeof(int)*size, 
    //  cudaMemcpyHostToDevice) );

    cull_empty_cells_cuda(
      &(Instance()->d_vol_pitchedptr),
      Instance()->vol_extent );

    cutilSafeCall( cudaGetLastError() );

    // copy back to initiate guess_x
    guess_x.clear();

    get_guess_x_cuda(
      Instance()->p_device_x,
      &(Instance()->d_vol_pitchedptr),
      Instance()->vol_extent );

    cutilSafeCall( cudaGetLastError() );

    cutilSafeCall( cudaMemcpy(
      Instance()->p_host_x,
      Instance()->p_device_x,
      sizeof(float)*num_items,
      cudaMemcpyDeviceToHost) );

    for (int i = 1; i < num_items; ++i)
    {
      guess_x.push_back( Instance()->p_host_x[i] );
    }

    // set renderer for current level
    renderer_->level_init(current_level_, vol_tex_);


    // cudpp for prefix summation
    CUDPPResult result = cudppDestroyPlan(scanplan_);
    if (CUDPP_SUCCESS != result)
    {
        printf("Error destroying CUDPPPlan\n");
        exit(-1);
    }

    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_FLOAT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_EXCLUSIVE;

    scanplan_ = 0;
    result = cudppPlan(&scanplan_, config, num_items, 1, 0);  

    if (CUDPP_SUCCESS != result)
    {
      fprintf(stderr, "Error creating CUDPPPlan\n");
      return false;
    }

    return true;
  }



  void ASMGradCompute::set_density_tags(
    int level,
    int *tag_volume,
    std::vector<float> &density,
    std::vector<uint16> &centers,
    bool is_init_density)
  {

    int length = (1<<level);

    float wc_x[9];
    float wc_y[9];
    float wc_z[9];

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
          wc_x[8] = wc_y[8] = wc_z[8] = 0.0;
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
          int ep_count_per_image[100];
          // sum of the values of all non-zero pixels
          unsigned int luminance_sum = 0;

          std::list<uint16> tmp_pcenters;
          // for each camera
          for (int i_camera = 0; i_camera < p_asmodeling_->num_cameras_; ++i_camera)
          {
            pts.clear();
            ep_count_per_image[i_camera] = 0;

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
              tmpPt.x = static_cast<int>(p_asmodeling_->camera_intr_paras_[i_camera](0,0) * ec.x
                + p_asmodeling_->camera_intr_paras_[i_camera](0,2) + 0.5f);
              tmpPt.y = static_cast<int>(p_asmodeling_->camera_intr_paras_[i_camera](1,1) * ec.y
                + p_asmodeling_->camera_intr_paras_[i_camera](1,2) + 0.5f);

              if (tmpPt.x > -1e-4 && tmpPt.y > -1e-4 &&
                tmpPt.x < p_asmodeling_->width_*1.0f && 
                tmpPt.y < p_asmodeling_->height_*1.0f)
                pts.push_back(tmpPt);
            }

            // Calc the effective pixels
            // construct the convex hull
            ConvexHull2D tmpConvexHull(pts);
            float x_min, x_max, y_min, y_max;
            tmpConvexHull.GetBoundingBox(x_min, x_max, y_min, y_max);

            // calc the projected area
            for (int vv = static_cast<int>(y_min); vv - y_max < 0.001f; ++vv)
            {
              if (vv < 0 || vv >= p_asmodeling_->height_)
                continue;
              for (int uu = static_cast<int>(x_min); uu - x_max < 0.001f; ++uu)
              {
                if (uu < 0 || uu >= p_asmodeling_->width_)
                  continue;

                if (!tmpConvexHull.IfInConvexHull(uu*1.0f, vv*1.0f))
                  continue;

                // Currently only R channel
                unsigned char pix = *(p_asmodeling_->ground_truth_image_.GetPixelAt(uu,vv+zbase));
                if (pix > 0)
                {
                  ++ ep_count_per_image[i_camera];
                  luminance_sum += pix;
                }
              }
            }

            // calc the projected center
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

            if (px < 0 || px >= p_asmodeling_->width_ || py < 0 || py >= p_asmodeling_->height_)
            {
              ep_count_per_image[i_camera] = 0;
            }
            tmp_pcenters.push_back(px);
            tmp_pcenters.push_back(p_asmodeling_->height_ - 1 - py);

          } // for i_camera

          n_effective_pixels = 0;
          bool is_effective = true;
          for (int icam = 0; icam < p_asmodeling_->num_cameras_; ++icam)
          {
            if (0 == ep_count_per_image[icam])
            {
              is_effective = false;
              break;
            }
            n_effective_pixels += ep_count_per_image[icam];
          }

          // this cell do has some projected matters
          if (is_effective)
          {
            // set tags
            ++ tag_index;
            int vol_index = p_asmodeling_->index3(i, j, k, length);
            tag_volume[vol_index] = tag_index;

            // set density
            if (is_init_density)
              density.push_back( static_cast<float>(luminance_sum) / (255.0f*n_effective_pixels) );

            // add pcenters
            for (std::list<uint16>::const_iterator it = tmp_pcenters.begin();
              it != tmp_pcenters.end(); ++it)
            {
              centers.push_back(*it);
            }

          } // is_effective

        } // for i
      } // for j
    } // for k

  }

  bool ASMGradCompute::set_asmodeling(ASModeling *p)
  {
    if (NULL == p)
      return false;
    p_asmodeling_ = p;
    num_views = p->num_cameras_;
    return true;
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////


} // namespace as_modeling

