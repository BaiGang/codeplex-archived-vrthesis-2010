#include <stdafx.h>
#include <cstdio>

#include "ASModeling.h"


namespace as_modeling
{

  /////////////////////////////////////////////////
  //      initialization of the system
  /////////////////////////////////////////////////
  bool ASModeling::Initialize(const char * conf_filename, const char * camera_filename)
  {
    if (!load_configure_file(conf_filename))
    {
      fprintf(stderr, "Failed in loading configure file.\n");
      return false;
    }

    if (!load_camera_file(camera_filename))
    {
      fprintf(stderr, "Failed in loading camera file.\n");
      return false;
    }


    char * tmpcam = new char [num_cameras_];
    camera_orientations_.reset(tmpcam);
    for (int i = 0; i < num_cameras_; ++i)
    {
      // camera orientations
      Vector4 dir(
        camera_positions_[i].x - trans_x_,
        camera_positions_[i].y - trans_y_,
        camera_positions_[i].z - trans_z_,
        1.0
        );

      dir.normaVec();

      if (abs(dir.x)>abs(dir.y) && abs(dir.x)>abs(dir.z))
      {
        // along x
        if (dir.x < 0.0)
          camera_orientations_[i] = 'X';
        else
          camera_orientations_[i] = 'x';
      }
      else if (abs(dir.y)>abs(dir.x) && abs(dir.y)>abs(dir.z))
      {
        // along y
        if (dir.y < 0.0)
          camera_orientations_[i] = 'Y';
        else
          camera_orientations_[i] = 'y';

      }
      else if (abs(dir.z)>abs(dir.x) && abs(dir.z)>abs(dir.y))
      {
        // along z
        if (dir.z < 0.0)
          camera_orientations_[i] = 'Z';
        else
          camera_orientations_[i] = 'z';

      }
      else
      {
        // should not have been here
        fprintf(stderr, " ERROR : axis specifying error!\n\n");
        return false;
      }
    }

    // init image list
    // num_cameras images
    if (!ground_truth_image_.SetSizes(width_, height_*num_cameras_))
    {
      fprintf(stderr, " ERROR : alloc space for ground image error!\n\n");
      return false;
    }

    // init gradient computer
    ASMGradCompute::Instance()->set_asmodeling(this);
    if (!ASMGradCompute::Instance()->init())
    {
      return false;
    }

    // result storage and i/o
    result_data_.SetSizes(MAX_VOL_SIZE, MAX_VOL_SIZE*MAX_VOL_SIZE);

    return true;
  }


  /////////////////////////////////////////////////////////////////////////////
  //
  //      full modeling process
  //
  /////////////////////////////////////////////////////////////////////////////
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

} // namespace as_modeling