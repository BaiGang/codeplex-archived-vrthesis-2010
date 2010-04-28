#ifndef _GRAD_COMPUTE_H_
#define _GRAD_COMPUTE_H_

#include <cstdlib>
#include <cstdio>
#include <list>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <>
#include "../L-BFGS-B/ap.h"

namespace as_modeling{

  class ASModeling;

  class ASMGradCompute
  {
  public:

    // for lbfgsbminimize callback
    static void grad_compute(ap::real_1d_array, double&, ap::real_1d_array&);

    // init 3D volume texture
    // init CUDA
    // set the initial guess for x
    // construct the volume tags
    // set the projection center for each item
    void init(int level, std::list<float>& guess_x, std::list<int>& projection_center, int * tag_volume);

    // release resources
    void release( );

    // set the volume tag and projection center for current level
    void init_current_level(int level);
    
    // set indicators for the current level
    //  level : curent level
    //  tag_volume : the tags of density volume, volume cell to array index mapping
    //  density : a list of non-zero density values, z-y-x lay out
    //  is_init_density : if true, the density will be set,
    //                    else, just set the tag_volume
    void set_density_tags(int level, int * tag_volume, std::list<float>& density, bool is_init_density);

    // Set the density volume
    // using d_x and pre-generated tag_volume
    void set_volume(int level, float * d_x);

    // must feed with a ASModeling 
    explicit ASMGradCompute(ASModeling *p)
      :p_asmodeling_(p), vol_data(0), tag_volume(0){};

  private:
    // no default constructor
    ASMGradCompute(){};

    // pointer to the caller ASModeling object
    ASModeling * p_asmodeling_;

    // CUDA resources
    cudaGraphicsResource * resource_vol_;

    float * vol_data;

    int * tag_volume;

  };

} // namespace as_modeling

#endif //_GRAD_COMPUTE_H_