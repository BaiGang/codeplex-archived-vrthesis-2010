#include <stdafx.h>
#include <cstdio>

#include "ASModeling.h"
#include "GradCompute.h"
#include <lbfgsb.h>

#include "../Utils/Timer/CPUTimer.h"

////////////////////////////////////////////////////////////
//
//  OptimizeSingleFrame()
//    Reconstruct the density field for the first frame
//
//  Progressively optimization
//
////////////////////////////////////////////////////////////

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

    Timer tmer;

    ASMGradCompute::Instance()->set_ground_truth_images(ground_truth_image_);

    ////////////////////////////////////////////
    //// for debugging
    //cuda_imageutil::BMPImageUtil tmpbmp;
    //tmpbmp.SetSizes(ground_truth_image_.GetWidth(), ground_truth_image_.GetHeight());
    //for (int k = 0; k < num_cameras_; ++k)
    //{
    //  for (int j = 0; j < ground_truth_image_.GetHeight(); ++j)
    //  {
    //    for (int i = 0; i < ground_truth_image_.GetWidth(); ++i)
    //    {
    //      for (int c = 0; c < 3; ++c)
    //      {
    //        tmpbmp.GetPixelAt( i, k*ground_truth_image_.GetHeight()+j )[0] 
    //        = ground_truth_image_.GetPixelAt( i, k*ground_truth_image_.GetHeight()+j )[0];
    //        tmpbmp.GetPixelAt( i, k*ground_truth_image_.GetHeight()+j )[1] 
    //        = ground_truth_image_.GetPixelAt( i, k*ground_truth_image_.GetHeight()+j )[1];
    //        tmpbmp.GetPixelAt( i, k*ground_truth_image_.GetHeight()+j )[2] 
    //        = ground_truth_image_.GetPixelAt( i, k*ground_truth_image_.GetHeight()+j )[2];
    //      }
    //    }
    //  }
    //}
    //tmpbmp.SaveImage("../Data/groundtruth.bmp");
    ////////////////////////////////////////////

    int i_level = initial_vol_level_;
    std::vector<float> host_x;

    tmer.start();

    ASMGradCompute::Instance()->frame_init(i_level, host_x);

    fprintf(stderr, "TIMING : frame_init of level %d, used %lf secs.\n", i_level, tmer.stop());

    ///////////////////////////////////////////////////////////////////////
    //
    //  set parameters for lbfgsb minimize routine
    //
    ///////////////////////////////////////////////////////////////////////
    lbfgsb_x_.setbounds(1, host_x.size());
    lbfgsb_nbd_.setbounds(1, host_x.size());
    lbfgsb_l_.setbounds(1, host_x.size());
    lbfgsb_u_.setbounds(1, host_x.size());

    int index_x = 1;
    for (std::vector<float>::const_iterator it = host_x.begin();
      it != host_x.end();
      ++ it)
    {
      lbfgsb_x_(index_x) = *it;

      lbfgsb_nbd_(index_x) = constrain_type_;
      lbfgsb_l_(index_x) = lower_boundary_;
      lbfgsb_u_(index_x) = upper_boundary_;

      ++ index_x;
    }

    tmer.start();

    // optimize the most coarse volume
    lbfgsbminimize(host_x.size(),
      lbfgs_m_, 
      lbfgsb_x_,
      eps_g_, 
      eps_f_, 
      eps_x_,
      max_iter_,
      lbfgsb_nbd_,
      lbfgsb_l_,
      lbfgsb_u_,
      lbfgsb_info_code_,
      ASMGradCompute::grad_compute
      );

    fprintf(stderr, "TIMING : call lbfgsbminimize of level %d, used %lf secs.\n", i_level, tmer.stop());

    fprintf(stderr, "Level : %d\nreturned info code : %d\n\n", i_level, lbfgsb_info_code_);
    printf("Level : %d\nreturned info code : %d\n\n", i_level, lbfgsb_info_code_);

    //ASMGradCompute::Instance()->get_data(i_level, frame_volume_result_, lbfgsb_x_);
    //fprintf(stderr, "\n\n Reconstructed results saved...\n\n");

    // get data
    // copy the volume data to HOST
    if (!ASMGradCompute::Instance()->get_data(i_level, frame_volume_result_, lbfgsb_x_))
    {
      fprintf( stderr, "===== Could not get volume data...\n" );
      return false;
    }

    ++ i_level;

    // progressively optimize finer volumes
    while (i_level <= max_vol_level_)
    {
      fprintf(stderr, "===||||==== Optimizing level %d...\n", i_level);

      tmer.start();
      ASMGradCompute::Instance()->level_init(
        i_level, host_x, lbfgsb_x_);
      fprintf(stderr, "TIMING : level start of level %d used %lf secs.", i_level, tmer.stop());

      lbfgsb_x_.setbounds(1, host_x.size());
      lbfgsb_nbd_.setbounds(1, host_x.size());
      lbfgsb_l_.setbounds(1, host_x.size());
      lbfgsb_u_.setbounds(1, host_x.size());

      int index_x = 1;
      for (std::vector<float>::const_iterator it = host_x.begin();
        it != host_x.end();
        ++ it)
      {
        lbfgsb_x_(index_x) = *it;

        lbfgsb_nbd_(index_x) = constrain_type_;
        lbfgsb_l_(index_x) = lower_boundary_;
        lbfgsb_u_(index_x) = upper_boundary_;

        ++ index_x;
      }

      tmer.start();
      lbfgsbminimize(host_x.size(),
        lbfgs_m_, 
        lbfgsb_x_,
        eps_g_, 
        eps_f_, 
        eps_x_,
        max_iter_,
        lbfgsb_nbd_,
        lbfgsb_l_,
        lbfgsb_u_,
        lbfgsb_info_code_,
        ASMGradCompute::grad_compute
        );
      fprintf(stderr, "TIMING : call lbfgsbminimize of level %d, used %lf secs.\n", i_level, tmer.stop());
      printf("Level : %d\nreturned info code : %d\n\n", i_level, lbfgsb_info_code_);

      // get data
      // copy the volume data to HOST
      if (!ASMGradCompute::Instance()->get_data(i_level, frame_volume_result_, lbfgsb_x_))
      {
        fprintf( stderr, "===== Could not get volume data...\n" );
        return false;
      }

      ++ i_level;

    }

    //// get data
    //// copy the volume data to HOST
    //if (!ASMGradCompute::Instance()->get_data(max_vol_level_, frame_volume_result_, lbfgsb_x_))
    //{
    //  fprintf( stderr, "===== Could not get volume data...\n" );
    //  return false;
    //}


    return true;
  }

} // namespace as_modeling