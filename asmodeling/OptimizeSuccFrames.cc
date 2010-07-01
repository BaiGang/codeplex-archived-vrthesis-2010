#include <stdafx.h>
#include "ASModeling.h"
#include "GradCompute.h"
#include <lbfgsb.h>

namespace as_modeling
{
  bool ASModeling::OptimizeSuccFrames(int i_frame)
  {
    // load the ground truth image for current frame
    if (!load_captured_images(i_frame))
    {
      fprintf(stderr, "<->  Error : Cannot load captured images, frame %d\n", i_frame);
      return false;
    }

    // get result from previous frame
    //  x_pre

    std::vector<float> guess_x;

    // calc x using previous x and volume tag
    ASMGradCompute::Instance()->succframe_init(max_vol_level_, guess_x, lbfgsb_x_);

    // x
    int n = guess_x.size();

    // call lbfgsb minimize routine
    lbfgsbminimize(
      n,
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
      ASMGradCompute::grad_compute );

    return true;
  }

} // namespace as_modeling