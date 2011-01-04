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
    tmer_2.start();

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
        Instance()->vol_extent,
        Instance()->d_tag_volume
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
        Instance()->full_vol_extent,
        Instance()->d_tag_volume
        );
      cutilSafeCall( cudaGetLastError() );
    }

#if 0
    //int len = 1 << Instance()->current_level_;
    int len = Instance()->full_vol_extent.depth;
    fprintf(stderr, "==len : %d\n", len);
    float * xx = new float [len * len * len + 10];
    get_volume_cuda(7, Instance()->d_full_vol_pptr,
      /*Instance()->d_tag_volume,*/ Instance()->d_temp_f );
    cutilSafeCall( cudaGetLastError() );
    cutilSafeCall( cudaMemcpy(
      xx, Instance()->d_temp_f, len*len*len*sizeof(float), cudaMemcpyDeviceToHost) );
    cutilSafeCall( cudaGetLastError() );
    PFMImage * tp = new PFMImage(len, len*len, 0, xx);
    tp->WriteImage("../Data/vol_ful222.pfm");
    delete tp;
#endif

#if 0
    cutilSafeCall( cudaGetLastError() );
    int llen = 1 << Instance()->current_level_;
    fprintf(stderr, "--len : %d\n", llen);
    float * ixx = new float [llen * llen * llen];
    get_volume_cuda( Instance()->current_level_,
      Instance()->d_vol_pitchedptr,
      /*Instance()->d_tag_volume,*/
      Instance()->d_temp_f );
    cutilSafeCall( cudaMemcpy(
      ixx, Instance()->d_temp_f,
      llen*llen*llen*sizeof(float),
      cudaMemcpyDeviceToHost) );
    PFMImage * ttp = new PFMImage(llen, llen*llen, 0, ixx);
    ttp->WriteImage("../Data/show_vol_1.pfm");
    delete ttp;
#endif

#if 0
    int lenl = 1<<Instance()->current_level_;
    float * xv = new float[len * len * len];
    construct_volume_linm_cuda(
      lenl,
      Instance()->p_device_x,
      Instance()->d_temp_f,
      Instance()->d_tag_volume );
    cutilSafeCall( cudaMemcpy(
      xv, Instance()->d_temp_f, lenl*lenl*lenl*sizeof(float),cudaMemcpyDeviceToHost) );
    PFMImage * t1p = new PFMImage(lenl, lenl*lenl, 0, xv);
    t1p->WriteImage("../Data/vol_111.pfm");
    delete t1p;
#endif

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

    fprintf(stderr, "--- === --- COPY to GPU used %lf secs.\n", tmer_2.stop());
    tmer_3.start();

#if 0
    get_volume_cuda(
      ASModeling::max_vol_level_,
      Instance()->d_full_vol_pptr,
      /*Instance()->d_tag_volume,*/
      Instance()->p_device_x );
    int mxsize = 128*128*128;
    float * data = new float [mxsize];
    float * img = new float[mxsize * 3];

    cutilSafeCall( cudaMemcpy(
      data, Instance()->p_device_x, sizeof(float)* mxsize, cudaMemcpyDeviceToHost) );

    for (int y = 0; y < 128*128; ++y)
    {
      for (int x = 0; x < 128; ++x)
      {
        img[(y * 128 + x)*3 + 0] = data[y * 128 + x];
        img[(y * 128 + x)*3 + 1] = data[y * 128 + x];
        img[(y * 128 + x)*3 + 2] = data[y * 128 + x];
      }
    }

    PFMImage * tmpfpm = new PFMImage(128, 128*128,
      1, img);
    tmpfpm->WriteImage("../Data/TestVol.pfm");
    delete[] data;
    delete tmpfpm;

#endif

    // reset the array for f[],
    // this is due to reduction only works for sizes that are power of 2
    int powtwo_length = nearest_pow2( n );

    int length = 1 << Instance()->current_level_;

#ifdef __DISPLAY_SHUTDOWN_GUARD__
    char * dummy;
    cutilSafeCall( cudaMalloc((void**)(&dummy), sizeof(char) * 1024) );
#endif

    // calc f and g[]
    for (int i_view = 0; i_view < Instance()->num_views; ++i_view)
    {
		if (3 == i_view)
			continue;

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

      cutilSafeCall( cudaMemset( Instance()->p_device_x, 0, powtwo_length * sizeof(float)));

#if 0 // test projected centers
      int piwidth = Instance()->p_asmodeling_->width_;
      int piheight = Instance()->p_asmodeling_->height_;
      float * pc = new float [piwidth * piheight];

      tst_pcenters(Instance()->current_level_, piwidth, piheight,
        i_view, Instance()->num_views, Instance()->d_projected_centers,
        Instance()->d_tag_volume, pc);
      PFMImage * pcpfm = new PFMImage(piwidth, piheight, 0, pc);
      char img_path[100];
      sprintf(img_path, "../Data/Camera%02d/pcenters.pfm", i_view);
      pcpfm->WriteImage(img_path);
      delete pcpfm;
#endif

      //float * h_testdata = new float [Instance()->p_asmodeling_->width_*Instance()->p_asmodeling_->height_];
      //float * d_testdata;
      //cutilSafeCall( cudaMalloc((void**)&d_testdata, 
      //  sizeof(float)*Instance()->p_asmodeling_->width_*Instance()->p_asmodeling_->height_) );

      //cutilSafeCall( cudaMemset((void*)d_testdata, 0,
      //  sizeof(float)*Instance()->p_asmodeling_->width_*Instance()->p_asmodeling_->height_) );

      // launch kernel
      float ff = calculate_f_cuda(
        Instance()->current_level_,
        Instance()->p_asmodeling_->width_,
        Instance()->p_asmodeling_->height_,
        i_view,
        Instance()->num_views,
        n,
        powtwo_length,
        Instance()->p_asmodeling_->render_interval_array_[Instance()->current_level_],
        Instance()->d_projected_centers,
        Instance()->d_tag_volume,
        Instance()->p_device_x,
        Instance()->d_temp_f
        /*d_testdata*/);

      //cutilSafeCall( cudaMemcpy(h_testdata, d_testdata,
      //  sizeof(float)*Instance()->p_asmodeling_->width_*Instance()->p_asmodeling_->height_,
      //  cudaMemcpyDeviceToHost) );

      //char tmpbuf[100];
      //sprintf(tmpbuf, "../Data/Camera%02d/testF.pfm", i_view);
      //PFMImage * testFpfmimg = new PFMImage(Instance()->p_asmodeling_->width_,
      //  Instance()->p_asmodeling_->height_, 0, h_testdata);
      //testFpfmimg -> WriteImage(tmpbuf);
      //delete testFpfmimg;

      fprintf(stderr, "++ ++ ++ F value of view %d is %f\n", i_view, ff);
      f += ff;
      //f += 1.0f;

#if 0 // debug rr tex and gt tex
      static int count = 0;
      int rwidth = Instance()->p_asmodeling_->width_;
      int rheight = Instance()->p_asmodeling_->height_;

      float * rr =new float[ rwidth * rheight * sizeof(float)];
      float * gt =new float[ rwidth * rheight * sizeof(float)];
      //void test__(int width, int height, int iview, float * h_data1, float * h_data2);
      test__(rwidth, rheight, i_view, rr, gt);

      PFMImage * pfm1 = new PFMImage(rwidth, rheight, 0, rr);
      PFMImage * pfm2 = new PFMImage(rwidth, rheight, 0, gt);

      char path_buf[100];
      sprintf(path_buf, "../Data/Camera%02d/getitfromdeviceGT%06d.pfm", i_view, count);
      pfm2->WriteImage(path_buf);
      sprintf(path_buf, "../Data/Camera%02d/getitfromdeviceRR%06d.pfm", i_view, count);
      pfm1->WriteImage(path_buf);
      delete pfm1;
      delete pfm2;
      ++ count;
#endif

      // calc g[]
      for (int pt_slice = 0; pt_slice < length; ++pt_slice)
      {
        //fprintf(stderr, " --- --- perturbing slice %d...\n", pt_slice);

        int mm = Instance()->p_asmodeling_->volume_interval_array_[Instance()->current_level_];

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

            //// launch kernel
            calculate_g_cuda(
              Instance()->current_level_,
              Instance()->p_asmodeling_->width_,
              Instance()->p_asmodeling_->height_,
              i_view,
              Instance()->num_views,
              Instance()->p_asmodeling_->render_interval_array_[ Instance()->current_level_ ],
              mm,
              pu,
              pv,
              Instance()->p_asmodeling_->camera_orientations_[i_view],
              pt_slice,
              Instance()->d_projected_centers,
              Instance()->d_tag_volume,
              Instance()->p_device_g
              );

#if 0 // test perturbed restult tex
            int prwidth = Instance()->p_asmodeling_->width_;
            int prheight = Instance()->p_asmodeling_->height_;

            float * pr =new float[ prwidth * prheight * sizeof(float)];
            //void test__(int width, int height, int iview, float * h_data1, float * h_data2);
            tst_g(prwidth, prheight, i_view, pr);

            PFMImage * pfm3 = new PFMImage(rwidth, rheight, 0, pr);

            char path_buf[100];
            sprintf(path_buf, "../Data/Camera%02d/getitfromdevicePR.pfm", i_view);
            pfm3->WriteImage(path_buf);
            delete pfm3;
#endif

            // unmap resource
            cutilSafeCall( cudaGraphicsUnmapResources(1, &(Instance()->resource_pr_)) );

          } // for pu
        } // for pv
      } // for each slice

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

    fprintf(stderr, "\nG values:\n");
    for (int i = 1; i <= 7; ++i)
    {
      fprintf(stderr, "%lf ", g(i));
    }
    fprintf(stderr, "\n");

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

    ////// show render result
    ////if (Instance()->current_level_ != Instance()->p_asmodeling_->max_vol_level_)
    ////{
    ////  // construct volume using x[] and voxel tags
    ////  construct_volume_cuda(
    ////    Instance()->p_device_x,
    ////    &(Instance()->d_vol_pitchedptr),
    ////    Instance()->vol_extent,
    ////    Instance()->d_tag_volume
    ////    );

    ////  // upsampling
    ////  upsample_volume_cuda(
    ////    Instance()->current_level_,
    ////    Instance()->p_asmodeling_->max_vol_level_,
    ////    &(Instance()->d_vol_pitchedptr),
    ////    &(Instance()->d_full_vol_pptr)
    ////    );
    ////  cutilSafeCall( cudaGetLastError() );

    ////}
    ////else
    ////{
    ////  // construct volume using x[] and voxel tags
    ////  construct_volume_cuda(
    ////    Instance()->p_device_x,
    ////    &(Instance()->d_full_vol_pptr),
    ////    Instance()->full_vol_extent,
    ////    Instance()->d_tag_volume
    ////    );
    ////  cutilSafeCall( cudaGetLastError() );
    ////}
    ////// map gl graphics resource
    ////cutilSafeCall( cudaGraphicsMapResources(1, &(Instance()->resource_vol_)) );

    ////cutilSafeCall( cudaGraphicsSubResourceGetMappedArray(
    ////  &(Instance()->vol_tex_cudaArray),
    ////  Instance()->resource_vol_,
    ////  0,
    ////  0) );

    ////// copy 
    ////cudaMemcpy3DParms param = {0};
    ////param.dstArray = Instance()->vol_tex_cudaArray;
    ////param.srcPtr   = Instance()->d_full_vol_pptr;
    ////param.extent   = Instance()->vol_cudaArray_extent;
    ////param.kind     = cudaMemcpyDeviceToDevice;
    ////cutilSafeCall( cudaMemcpy3D(&param) );

    ////// unmap gl graphics resource after writing-to operation
    ////cutilSafeCall( cudaGraphicsUnmapResources(1, &Instance()->resource_vol_) );

    ////for (int i_view = 0; i_view < num_views; ++i_view)
    ////{
    ////  renderer_->render_unperturbed(i_view, vol_tex_, 1 << current_level_);
    ////  char path_buf[100];
    ////  sprintf(path_buf, "../Data/Results/Frame%08d_View%02d_Level%d.PFM", 0, i_view, level);
    ////  float * data = renderer_->rr_fbo_->ReadPixels();
    ////  float * img = new float [p_asmodeling_->width_*p_asmodeling_->height_];
    ////  for (int y = 0; y < p_asmodeling_->height_; ++y)
    ////  {
    ////    for (int x = 0; x < p_asmodeling_->width_; ++x)
    ////    {
    ////      img[y*p_asmodeling_->width_+x] = data[4*(y*p_asmodeling_->width_+x)];
    ////    }
    ////  }
    ////  PFMImage *sndipfm = new PFMImage(p_asmodeling_->width_,
    ////    p_asmodeling_->height_,
    ////    0, img);
    ////  sndipfm->WriteImage(path_buf);
    ////}

    ////float * imgdata = new float [length * length*length];
    ////PFMImage * pfmhaha = new PFMImage(length, length*length, 0, imgdata);
    ////char pathbuf [100];
    ////sprintf(pathbuf, "../Data/Results/Result_Level%d.PFM", level);
    ////pfmhaha->WriteImage(pathbuf);
    ////delete pfmhaha;
    ////// set over

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
	int sub_size = max_size * 0.4f;
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

    fprintf(stderr, "d_projected_centers size : %d\n", 2 * p_asmodeling_->num_cameras_ * sub_size);

    //cutilSafeCall( cudaMalloc<uint16>(&d_projected_centers, 2 * p_asmodeling_->num_cameras_ * max_size) );
    cutilSafeCall( cudaMalloc((void**)(&d_projected_centers), 2 * p_asmodeling_->num_cameras_ * sub_size * sizeof(uint16)) );

    //cutilSafeCall( cudaMalloc<int>(&d_tag_volume, max_size) );
    cutilSafeCall( cudaMalloc((void**)(&d_tag_volume), max_size * sizeof(int)) );

    //cutilSafeCall( cudaMalloc<float>(&p_device_x, max_size) );
    cutilSafeCall( cudaMalloc((void**)(&p_device_x), sub_size*sizeof(float)) );

    //cutilSafeCall( cudaMalloc<float>(&p_device_g, max_size) );
    cutilSafeCall( cudaMalloc((void**)(&p_device_g), sub_size * sizeof(float)) );

    //cutilSafeCall( cudaMalloc<float>(&d_temp_f,   max_size) );
    cutilSafeCall( cudaMalloc((void**)(&d_temp_f), max_size * sizeof(float)) );

    // alloc array for ground truth image
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    gt_cudaArray_extent = make_cudaExtent(p_asmodeling_->width_, p_asmodeling_->height_, num_views);
    cutilSafeCall( cudaMalloc3DArray(&gt_tex_cudaArray, &channelDesc, gt_cudaArray_extent) );

    //////////////////////////////////////////////////////////
    // alloc memory on HOST
    //////////////////////////////////////////////////////////
    h_vol_data = new float [max_size];
    h_projected_centers = new uint16 [2 * p_asmodeling_->num_cameras_ * sub_size];
    h_tag_volume = new int [ max_size ];

    p_host_g = new float [sub_size];
    p_host_x = new float [sub_size];

#if 0
    // test render here
    glBindTexture(GL_TEXTURE_3D, vol_tex_);
    float *tmp = new float[max_size];
    for (int i = 0; i < max_size; ++i)
    {
      tmp[i] = (i+0.5f) / (1.0f*max_size);
    }
    glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE32F_ARB, max_length, max_length, max_length, 0, GL_LUMINANCE, GL_FLOAT, tmp);
    CUT_CHECK_ERROR_GL2();

    for (int i = 0; i < p_asmodeling_->num_cameras_; ++i)
    {
      renderer_->render_unperturbed(i, vol_tex_, 1024);
    }

#endif

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

    delete renderer_;

    delete [] h_projected_centers;
    delete [] h_tag_volume;
    delete [] h_vol_data;

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

#if 0
    float * img = new float [length * length * length * 3];
    memset(img, 0, length * length * length * 3 * sizeof(float));
    for (int k = 0; k < length; ++k)
    {
      for (int j = 0; j < length; ++j)
      {
        for (int i = 0; i < length; ++i)
        {
          int vol_ind = p_asmodeling_->index3(i, j, k, length);
          int arr_ind = h_tag_volume[vol_ind];
          float val;
          if (0 == arr_ind)
            val = 0.0f;
          else
            val = guess_x[arr_ind-1];

          img[vol_ind*3 + 0] = val;
          img[vol_ind*3 + 1] = val;
          img[vol_ind*3 + 2] = val;
        }
      }
    }

    PFMImage * test = new PFMImage(length, length*length, 1, img);

    test->WriteImage("../Data/ShowInit.pfm");
    delete test;
#endif

    // copy data to CUDA
    int i = 0;
    for (std::list<uint16>::const_iterator it = projected_centers_.begin();
      it != projected_centers_.end();
      ++i, ++it)
    {
      h_projected_centers[i] = *it;
    }

    fprintf(stderr, "\n\n, sizeof projected_centers_ : %d\n", projected_centers_.size());

#if 0 // test projecting centers...  very ok....
    int tttwidth = Instance()->p_asmodeling_->width_;
    int tttheight = Instance()->p_asmodeling_->height_;

    for (int iview = 0; iview < Instance()->num_views; ++iview)
    {
      float * tttdata = new float [tttwidth * tttheight];
      memset(tttdata, 0, sizeof(float) * tttwidth * tttheight);

      for (int k = 0; k < length; ++k)
      {
        for (int j = 0; j < length; ++j)
        {
          for (int i = 0; i < length; ++i)
          {
            int vol_index = p_asmodeling_->index3(i, j, k, length);
            int arr_index = h_tag_volume[ vol_index ];

            uint16 u = h_projected_centers[2*Instance()->num_views*(arr_index-1)+2*iview];
            uint16 v = h_projected_centers[2*Instance()->num_views*(arr_index-1)+2*iview+1];

            if (u<tttwidth && v<tttheight)
              tttdata[v*tttwidth+u] = 1.0f;
          }
        }
      }

      PFMImage * tmpfpmm = new PFMImage(tttwidth,
        tttheight, 0, tttdata);
      char buf[100];
      sprintf(buf, "../Data/showPcenter%02d.pfm", iview);
      tmpfpmm -> WriteImage(buf);
      delete tmpfpmm;
    } // for iview
#endif

    cutilSafeCall( cudaMemcpy(
      d_projected_centers,
      h_projected_centers,
      sizeof(uint16)*projected_centers_.size(),
      cudaMemcpyHostToDevice));

    cutilSafeCall( cudaMemcpy(
      d_tag_volume, 
      h_tag_volume, 
      sizeof(int)*size, 
      cudaMemcpyHostToDevice) );


    return true;
  }


  bool ASMGradCompute::succframe_init(int level, std::vector<float>& guess_x, ap::real_1d_array& prev_x)
  {
    // init using previous frame's result
    current_level_ = level;
    int length = 1<<level;
    int size = length * length * length;

    std::vector<float> dummy_list;

    set_density_tags(level, h_tag_volume, dummy_list, projected_centers_, false);

    // allocate space for volume data on device
    size_t vol_size = 1 << current_level_;
    vol_extent = make_cudaExtent(vol_size*sizeof(float), vol_size, vol_size);
    cutilSafeCall( cudaMalloc3D(&d_vol_pitchedptr, vol_extent) );

    // copy data to CUDA
    int i = 0;
    for (std::list<uint16>::const_iterator  it = projected_centers_.begin();
      it != projected_centers_.end();
      ++it, ++i)
    {
      h_projected_centers[i] = *it;
    }

    cutilSafeCall( cudaMemcpy(
      d_projected_centers,
      h_projected_centers,
      sizeof(uint16)*projected_centers_.size(),
      cudaMemcpyHostToDevice) );
    cutilSafeCall( cudaMemcpy(
      d_tag_volume,
      h_tag_volume,
      sizeof(int)*size,
      cudaMemcpyHostToDevice) );

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
      Instance()->vol_extent,
      Instance()->d_tag_volume );

    cutilSafeCall( cudaGetLastError() );

    // followed calculation of tag volume and projected centers
    std::vector<float> dummy_list;
    set_density_tags(level, h_tag_volume, dummy_list, projected_centers_, false);

    // cull the empty cells 
    int n_nonzero_items = projected_centers_.size() / (2*num_views);

    // copy data to CUDA
    int i = 0;
    for (std::list<uint16>::const_iterator it = projected_centers_.begin();
      it != projected_centers_.end();
      ++i, ++it)
    {
      h_projected_centers[i] = *it;
    }

    fprintf(stderr, "++++++++++++++++++++++++++++++ nonzero voxels size : %d\n", n_nonzero_items);

    cutilSafeCall( cudaMemcpy(
      d_projected_centers,
      h_projected_centers,
      sizeof(uint16)*projected_centers_.size(),
      cudaMemcpyHostToDevice));

    cutilSafeCall( cudaMemcpy(
      d_tag_volume,
      h_tag_volume,
      sizeof(int)*size,
      cudaMemcpyHostToDevice) );

    cull_empty_cells_cuda(
      &(Instance()->d_vol_pitchedptr),
      Instance()->vol_extent,
      Instance()->d_tag_volume );

    cutilSafeCall( cudaGetLastError() );


    // copy back to initiate guess_x
    guess_x.clear();

    get_guess_x_cuda(
      Instance()->p_device_x,
      &(Instance()->d_vol_pitchedptr),
      Instance()->vol_extent,
      Instance()->d_tag_volume );

    cutilSafeCall( cudaGetLastError() );

    cutilSafeCall( cudaMemcpy(
      Instance()->p_host_x,
      Instance()->p_device_x,
      sizeof(float)*(1+n_nonzero_items),
      cudaMemcpyDeviceToHost) );

    for (int i = 1; i <= n_nonzero_items; ++i)
    {
      guess_x.push_back( Instance()->p_host_x[i] );
    }

    return true;
  }



  void ASMGradCompute::set_density_tags(
    int level,
    int *tag_volume,
    std::vector<float> &density,
    std::list<uint16> &centers,
    bool is_init_density)
  {

#ifdef __DEBUG_IMAGE_
    cuda_imageutil::BMPImageUtil * debugBMP = new cuda_imageutil::BMPImageUtil[8];
    for (int img=0; img<8;++img)
    {
      debugBMP[img].SetSizes(p_asmodeling_->width_, p_asmodeling_->height_);
      debugBMP[img].ClearImage();
    }
#endif

#if 0
    /////////////////////
    int tlength = 32;
    for (int i_view = 0; i_view<num_views; ++i_view)
    {
      for (int kk = -tlength; kk <= tlength; ++kk)
      {
        for (int jj = -tlength; jj <= tlength; ++jj)
        {
          for (int ii = -tlength; ii <= tlength; ++ii)
          {
            float x = p_asmodeling_->trans_x_ + ii*0.5/tlength*p_asmodeling_->box_size_;
            float y = p_asmodeling_->trans_y_ + jj*0.5/tlength*p_asmodeling_->box_size_;
            float z = p_asmodeling_->trans_z_ + kk*0.5/tlength*p_asmodeling_->box_size_;

            Vector4 cwc(x,y,z);
            Vector4  cec = p_asmodeling_->camera_extr_paras_[i_view] * cwc;

            cec.x /= cec.z;
            cec.y /= cec.z;
            cec.z /= cec.z;

            int pu = static_cast<int>( 0.5+
              p_asmodeling_->camera_intr_paras_[i_view](0,0)* cec.x + p_asmodeling_->camera_intr_paras_[i_view](0,2) );
            int pv = static_cast<int>( 0.5+
              p_asmodeling_->camera_intr_paras_[i_view](1,1)* cec.y + p_asmodeling_->camera_intr_paras_[i_view](1,2) );

            debugBMP[i_view].GetPixelAt(pu,pv)[0] = 255;
            debugBMP[i_view].GetPixelAt(pu,pv)[1] = 255;
            debugBMP[i_view].GetPixelAt(pu,pv)[2] = 255;

          }
        }
      }
      char tm[200];
      sprintf_s(tm,"../Data/Camera%02d/Projected.BMP", i_view);
      debugBMP[i_view].SaveImage(tm);
      debugBMP[i_view].ClearImage();
    }
    /////////////////////
#endif

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

#ifdef __DEBUG_IMAGE_
    // calc the projected pixels of the current cell 
    int cn_ind = 0;
    for (int k = -1; k < 2; k+=2)
    {
      for (int j = -1; j < 2; j+=2)
      {
        for (int i = -1; i < 2; i+=2)
        {
          for (int i_camera = 0; i_camera < p_asmodeling_->num_cameras_; ++i_camera)
          {
            Vector4 cwc;
            cwc.x = i*0.5*p_asmodeling_->box_size_ + p_asmodeling_->trans_x_;
            cwc.y = j*0.5*p_asmodeling_->box_size_ + p_asmodeling_->trans_y_;
            cwc.z = k*0.5*p_asmodeling_->box_size_ + p_asmodeling_->trans_z_;
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

            debugBMP[i_camera].GetPixelAt(px,py)[0] = 254 * (i+1)/2;
            debugBMP[i_camera].GetPixelAt(px,py)[1] = 254 * (j+1)/2;
            debugBMP[i_camera].GetPixelAt(px,py)[2] = 254 * (k+1)/2;

          } // for each camera

          ++ cn_ind;
        }
      }
    }

    for (int img=0; img<8;++img)
    {
      char path_buf[50];
      sprintf_s(path_buf, 50, "../Data/Camera%02d/test%02d.bmp", img, img);
      debugBMP[img].SaveImage(path_buf);
    }
#endif

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

