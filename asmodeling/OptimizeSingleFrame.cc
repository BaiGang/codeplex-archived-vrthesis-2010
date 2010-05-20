#include <cstdio>

#include "ASModeling.h"
#include "GradCompute.h"
#include <lbfgsb.h>

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

    // set groundtruth image to CUDA
    //set_groundtruth_image();

    ASMGradCompute::Instance()->set_ground_truth_images(ground_truth_image_);

    int i_level = INITIAL_VOL_LEVEL;
    std::list<float> host_x;

    ASMGradCompute::Instance()->frame_init(i_level, host_x);

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
    for (std::list<float>::const_iterator it = host_x.begin();
      it != host_x.end();
      ++ it)
    {
      lbfgsb_x_(index_x) = *it;

      lbfgsb_nbd_(index_x) = constrain_type_;
      lbfgsb_l_(index_x) = lower_boundary_;
      lbfgsb_u_(index_x) = upper_boundary_;

      ++ index_x;
    }

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

    // finish this optimize call
    //delete_density_indicator();

    // progressively optimize finer volumes
    while (i_level <= MAX_VOL_LEVEL)
    {
      // TODO:
      // subdivide volume and the previous array x
      

      // TODO:
      // optimize routine

      ++ i_level;
    }

    // TODO:
    // store the resulted volume...
    // maybe use some kind of compression...
    //grad_computer_->

    return true;
  }

} // namespace as_modeling