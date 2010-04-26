#ifndef _GRAD_COMPUTE_H_
#define _GRAD_COMPUTE_H_

#include <cstdlib>
#include <cstdio>
#include <list>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
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
    void init( );
    void release();

    void init_current_level(int level);
    
    // set indicators for the current level
    //  level : curent level
    //  tag_volume : the tags of density volume, volume cell to array index mapping
    //  density : a list of non-zero density values, z-y-x lay out
    //  is_init_density : if true, the density will be set,
    //                    else, just set the tag_volume
    void set_density_tags(int level, int * tag_volume, std::list<float>& density, bool is_init_density);

    // Set the density volume
    // using d_x and pre-
    void set_volume(int level, float * d_x);

    // must feed with a ASModeling 
    explicit ASMGradCompute(ASModeling *p)
      :p_asmodeling_(p){};

  private:
    // no default constructor
    ASMGradCompute(){};
    ASModeling * p_asmodeling_;

    // CUDA resources
    cudaGraphicsResource * resource_vol_;

    float * vol_data;

  };

} // namespace as_modeling

#endif //_GRAD_COMPUTE_H_