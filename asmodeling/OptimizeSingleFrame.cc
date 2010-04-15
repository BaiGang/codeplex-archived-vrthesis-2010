#include <cstdio>

#include "ASModeling.h"
#include "../L-BFGS-B/ap.h"
#include "../L-BFGS-B/lbfgsb.h"

////////////////////////////////////////////////////////////
//
//  OptimizeSingleFrame()
//    Reconstruct the density field for the first frame
//
//  Progressively optimization
//
////////////////////////////////////////////////////////////

namespace {
  extern "C"
  {
    // CUDA C functions here

    // grad computing function for lbfgsb routine
    void grad_compute(ap::real_1d_array &x, double &f, ap::real_1d_array &g);

    // set initial density, indicator and position mapping
    // uchar ind[length^3], 
    void set_vol_indicator_cuda(uchar * ind_volume);

    // subdivide volume
    void subdivide_volume_cuda(int prev_level, int next_level);

  }
} // unnamed namespace



namespace as_modeling
{
  bool ASModeling::OptimizeSingleFrame(int iframe)
  {
    // first, load the frame
    if (!load_captured_images(iframe))
    {
      fprintf(stderr, "<->  Error : Cannot load captured images, frame %d\n", iframe);
      return false;
    }

    int i_level = INITIAL_VOL_LEVEL;

    set_density_indicator(i_level, progressive_results_[0].get(), progressive_indicators_[0].get());

    // optimize the most coarse volume


    // progressively optimize finer volumes
    while (i_level <= MAX_VOL_LEVEL)
    {
    }



    return true;
  }

} // namespace as_modeling