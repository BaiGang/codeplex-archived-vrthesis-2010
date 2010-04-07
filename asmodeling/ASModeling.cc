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
    

    return true;
  }
  /////////////////////////////////////////////////
  //      full modeling process
  /////////////////////////////////////////////////
  bool ASModeling::OptimizeProcess(int num_of_frames)
  {
    if (!OptimizeFirstFrame())
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