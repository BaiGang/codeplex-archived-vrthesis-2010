#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

// for ap::real_1d_array
#include "../L-BFGS-B/ap.h"

////////////////////////////////////////////////////////////////////////////
//                DEVICE code
////////////////////////////////////////////////////////////////////////////
#include "cuda_gradcompute_kernel.cu"
#include "cuda_raymarching_kernel.cu"

////////////////////////////////////////////////////////////////////////////
//                DEVICE variables
////////////////////////////////////////////////////////////////////////////
__device__ uint4 * index2pos;  // for each x[i], maps i to (px, py, pz)


////////////////////////////////////////////////////////////////////////////
//                 HOST variables
////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////
//                 HOST code
////////////////////////////////////////////////////////////////////////////

// The gradient computation routine for lbfgsbminimize()
__host__ void grad_compute(ap::real_1d_array &x, double &f, ap::real_1d_array &g)
{
  // set x

  // render x

  // calc f

  // render x+dx
  // first perturb x
  // then render

  // calc g[]
}

// set volume indicator, and 
extern "C"
void set_vol_indicator_cuda(unsigned char * ind_volume)
{
  // set volume and position mapping

}

// subdivide volume
extern "C"
void subdivide_volume_cuda(int prev_level, int next_level)
{
  // subdivide the current volume
  // the higher-level cells' density are set equal to their parent cells'

  // then use indicator to tick off vacuum cells

}