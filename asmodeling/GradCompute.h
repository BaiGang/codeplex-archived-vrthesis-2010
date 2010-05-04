#ifndef _GRAD_COMPUTE_H_
#define _GRAD_COMPUTE_H_

#include <cstdlib>
#include <cstdio>
#include <list>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
//#include "ASModeling.h"
#include "RenderGL.h"
#include "../L-BFGS-B/ap.h"

namespace as_modeling{

  class ASModeling;

  class ASMGradCompute
  {
  public:

    // for lbfgsbminimize callback
    static void grad_compute(ap::real_1d_array&, double&, ap::real_1d_array&);

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
    bool get_data(int &level, scoped_array<float>& data);

    // set the initial guess for x
    // construct the volume tags
    // set the projection center for each item
    bool frame_init(int level, std::list<float>& guess_x);

    // use previous level result
    // init the new x
    bool level_init(int level);

    // use previous frame result
    // init the new x
    bool succframe_init(int level);

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
    void set_asmodeling(ASModeling *p);
    ~ASMGradCompute(){delete instance_;}

  private:

    // no default constructor
    ASMGradCompute(){};

    static ASMGradCompute * instance_;

    // pointer to the caller ASModeling object
    ASModeling * p_asmodeling_;

    // CUDA resources
    cudaGraphicsResource * resource_vol_;
    cudaStream_t cuda_stream_;

    // pbo
    // for cuda access
    GLuint pbo_;

    // buffer texture id
    GLuint vol_tex_;

    // 
    int current_level_;

    // CUDA host memory
    float * h_vol_data;
    int * h_tag_volume;
    int * h_projected_centers;

    std::list<int> projected_centers_;

    // CUDA device memory
    float * d_vol_data;
    int * d_projected_centers;
    int * d_tag_volume;
    float * d_vol_bufferptr;
    size_t vol_buffer_num_bytes_;

    //cudaArray * tag_volume;
    //cudaChannelFormatDesc tag_channel_desc;
    //cudaExtent  tag_vol_extent;

    // texture ref
    //texture<int, 3, cudaReadModeElementType> *tag_tax_ref;
    //textureReference * tag_tax_ref;

    /////////////////////////////
    int num_views;
  };

  //ASMGradCompute::instance_ = NULL;

} // namespace as_modeling

#endif //_GRAD_COMPUTE_H_

