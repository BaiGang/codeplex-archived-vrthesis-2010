#ifndef _GRAD_COMPUTE_H_
#define _GRAD_COMPUTE_H_
#include <stdafx.h>
#include <cstdlib>
#include <cstdio>
#include <list>

#include "RenderGL.h"  // glew.h must be loaded befor gl.h

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "../L-BFGS-B/ap.h"
#include "../CudaImageUtil/CudaImgUtil.h"

namespace as_modeling{

  class ASModeling;

  class ASMGradCompute
  {
  public:

    // for lbfgsbminimize callback
    static void grad_compute(const ap::real_1d_array&, double&, ap::real_1d_array&);

    // Singleton pattern 
    static inline ASMGradCompute* Instance( )
    {
      if (NULL == instance_)
      {
        instance_ = new ASMGradCompute();
      }
      return instance_;
    }

    // init buffer texture
    // init CUDA
    bool init();

    // release resources
    bool release( );

    // get the volume data
    bool get_data(int level, scoped_array<float>& data, ap::real_1d_array &x);

    // set the loaded ground truth images
    bool set_ground_truth_images(cuda_imageutil::Image_4c8u& gt_images);

    // set the initial guess for x
    // construct the volume tags
    // set the projection center for each item
    bool frame_init(int level, std::list<float>& guess_x);

    // use previous level result
    // init the new x
    bool level_init(int level, std::list<float>& guess_x, ap::real_1d_array& prev_x);

    // use previous frame result
    // init the new x
    bool succframe_init(int level, std::list<float>& guess_x, ap::real_1d_array& prev_x);

  private:
    // set the volume tag and projection center for current level
    // set indicators for the current level
    //  level : curent level
    //  tag_volume : the tags of density volume, volume cell to array index mapping
    //  density : a list of non-zero density values, z-y-x lay out
    //  is_init_density : if true, the density will be set,
    //                    else, just set the tag_volume
    void set_density_tags(int level, int *tag_volume, std::list<float> &density, std::list<int> &centers, bool is_init_density);


  public:
    // must feed with an ASModeling 
    bool set_asmodeling(ASModeling *p);
    ~ASMGradCompute()
    {
      instance_->release();
      delete instance_;
      delete renderer_;
    }

  private:

    // no default constructor
    ASMGradCompute(){};

    static ASMGradCompute * instance_;

    // pointer to the caller ASModeling object
    ASModeling * p_asmodeling_;

    // Renderer
    RenderGL * renderer_;

    // CUDA Graphics resources
    cudaGraphicsResource * resource_vol_;
    cudaGraphicsResource * resource_rr_;
    cudaGraphicsResource * resource_pr_;

    cudaStream_t cuda_stream_;

    // pbo
    // for cuda access
    GLuint pbo_;

    // volume texture id
    GLuint vol_tex_;

    // 
    int current_level_;

    // CUDA host memory
    float * h_vol_data;       // 
    int * h_tag_volume;       // 
    int * h_projected_centers;

    std::list<int> projected_centers_;

    // for lbfgsb routine
    float * p_host_x;   // x, 
    float * p_host_g;

    // CUDA device memory

    cudaPitchedPtr d_vol_pitchedptr;
    cudaExtent     vol_extent;
    cudaPitchedPtr d_full_vol_pptr;
    cudaExtent     full_vol_extent;

    float * d_temp_f;

    int * d_projected_centers;
    int * d_tag_volume;

    cudaArray * vol_tex_cudaArray;
    cudaExtent  vol_cudaArray_extent;
    cudaArray * rr_tex_cudaArray;
    cudaArray * pr_tex_cudaArray;
    // 3d, z maps to different images
    cudaArray * gt_tex_cudaArray;
    cudaExtent  gt_cudaArray_extent;

    float * d_vol_bufferptr;
    size_t vol_buffer_num_bytes_;

    // for lbfgsb routine
    float * p_device_x;
    float * p_device_g;

    /////////////////////////////
    int num_views;
  };

  //ASMGradCompute::instance_ = NULL;

} // namespace as_modeling

#endif //_GRAD_COMPUTE_H_

