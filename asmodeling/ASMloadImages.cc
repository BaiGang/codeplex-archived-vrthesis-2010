#include <cstdio>

#include "ASModeling.h"

////////////////////////////////////////////////////////
//
//  Load captured image for current frame
//
//   All the frame images are stored at
//     ../Data/Camera**/Frame****
//     Cameras are numbered from 0 to num_cameras_-1
//
////////////////////////////////////////////////////////

namespace as_modeling
{
  bool ASModeling::load_captured_images(int iframe)
  {
    char path_buf[200];

    for (int i_camera = 0; i_camera < num_cameras_; ++i_camera)
    {
      sprintf_s(path_buf, 200, "../Data/Camera%02d/Frame%05d.bmp", i_camera, iframe);
      ground_truth_images_(i_camera).assign(path_buf);

    } // for each camera

    return true;
  }
} // namespace as_modeling