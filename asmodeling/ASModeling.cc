#include <cstdio>

#include "ASModeling.h"

namespace as_modeling
{

  /////////////////////////////////////////////////
  //      initialization of the system
  /////////////////////////////////////////////////
  bool ASModeling::Initialize(const char * conf_filename, const char * camera_filename)
  {
    if (!load_camera_file(camera_filename))
    {
      fprintf(stderr, "Failed in loading camera file.\n");
      return false;
    }

    if (!load_configure_file(conf_filename))
    {
      fprintf(stderr, "Failed in loading configure file.\n");
      return false;
    }

    int num_levels = MAX_VOL_LEVEL - INITIAL_VOL_LEVEL + 1;
    
    // init image list
    ground_truth_images_.assign(num_cameras_);

    // init intermediate data
    // allocate space for progressive density/indicator volume
    scoped_array<float> * tmparrarr = new scoped_array<float> [num_levels];
    scoped_array<uchar> * tmpuchararr = new scoped_array<uchar> [num_levels];
    progressive_results_.reset(tmparrarr);
    progressive_indicators_.reset(tmpuchararr);
    for (int i_level = INITIAL_VOL_LEVEL; i_level <= MAX_VOL_LEVEL; ++i_level)
    {
      float * tmparr = new float [(1<<i_level)];
      uchar * tuchararr = new uchar [(1<<i_level)];
      progressive_results_[i_level].reset(tmparr);
      progressive_indicators_[i_level].reset(tuchararr);
    }

    return true;
  }


  /////////////////////////////////////////////////
  //      full modeling process
  /////////////////////////////////////////////////
  bool ASModeling::OptimizeProcess(int num_of_frames)
  {
    if (!OptimizeSingleFrame(0))
    {
      fprintf(stderr, "<<!-- Optimize First Frame Failed.\n");
      return false;
    }

    StoreVolumeData(0);

    for (int i = 1; i < num_of_frames; ++i)
    {
      if (!OptimizeSuccFrames(i))
      {
        fprintf(stderr, "<<!-- Optimize Frame %d Failed.\n", i);
        return false;
      }
      StoreVolumeData(i);
    }

    return true;
  }

  /////////////////////////////////////////////////
  //      helper functions
  /////////////////////////////////////////////////



} // namespace as_modeling