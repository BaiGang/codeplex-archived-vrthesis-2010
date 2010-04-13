#include <cstdio>

#include "ASModeling.h"

////////////////////////////////////////////////////////////
//
//  OptimizeSingleFrame()
//    Reconstruct the density field for the first frame
//
//  Progressively optimization
//
////////////////////////////////////////////////////////////

extern "C"
{
}

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