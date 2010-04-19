#include <cstdio>

#include "ASModeling.h"
#include "../L-BFGS-B/lbfgsb.h"

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

  // calc gradient on cuda
  float cuda_grad_compute(float * p_host_x, float * p_host_g, int n);

  // set initial density, indicator and position mapping
  // int ind[length^3], 
  void set_density_indicator(int level, int * indicator);

  // subdivide volume
  void subdivide_volume_cuda(int prev_level, int next_level);

} // extern "C"


namespace {

  // grad computing function for lbfgsb routine
  //  this is actually a bridge between ASModeling and cuda routines
  void grad_compute(ap::real_1d_array x, double &f, ap::real_1d_array &g)
  {
    int n = x.gethighbound() - x.getlowbound() + 1;
    float * p_host_x = new float [n];
    float * p_host_g = new float [n];

    f = cuda_grad_compute(p_host_x, p_host_g, n);
  }

  // init volume
  // This piece of code looks quite like that in set_density_indicator()

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
    std::list<float> host_x;

    set_density_indicator(i_level, progressive_indicators_[0].get(),
      host_x, true);

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
      grad_compute);

    // progressively optimize finer volumes
    while (i_level <= MAX_VOL_LEVEL)
    {
      ++ i_level;
    }



    return true;
  }

} // namespace as_modeling