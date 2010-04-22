#include <cstdio>

#include "ASModeling.h"

extern "C"
{
  // cuda_gradcompute.cu
  bool set_camera_parameters_cuda(int n_camera, float * intr_para, float * extr_para);
}

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

    ////////////////////////////////////////////////////////
    //  Sets camera parameters to Device, also inits cuda
    ////////////////////////////////////////////////////////
    set_cameras();

    // init intermediate data
    // allocate space for progressive density/indicator volume
    scoped_array<int> * tmp_p_int = new scoped_array<int> [num_levels];
    progressive_indicators_.reset(tmp_p_int);
    for (int i_level = INITIAL_VOL_LEVEL; i_level <= MAX_VOL_LEVEL; ++i_level)
    {
      int * tuchararr = new int [(1<<i_level)];
      progressive_indicators_[i_level].reset(tuchararr);
    }

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

  ////////////////////////////////////////////////////////////////////////////////
  //
  //      helper functions
  //
  ////////////////////////////////////////////////////////////////////////////////
  bool ASModeling::set_groundtruth_image( )
  {
    // NOTE: the array in CImag is CImage::data
    float* images[MAX_NUM_CAMERAS];


    return true;
  } // set_groundtruth_image

  // set camera parameters to cuda
  // note, this function is called only once, at the begining of the optimization process
  bool ASModeling::set_cameras()
  {
    // here we init cuda and set camera parameters
    float * intr_para = new float[16 * num_cameras_];
    float * extr_para = new float[16 * num_cameras_];
    scoped_array<float> sa_intr, sa_extr;
    sa_intr.reset(intr_para);
    sa_extr.reset(extr_para);

    for (int i_camera = 0; i_camera < num_cameras_; ++i_camera)
    {
      int base = i_camera * 16;
      for (int p = 0; p < 16; ++p)
      {
        intr_para[base+p] = camera_intr_paras_[i_camera](p/4, p%4);
        extr_para[base+p] = camera_extr_paras_[i_camera](p/4, p%4);
      }
    }

    if (!set_camera_parameters_cuda(num_cameras_, intr_para, extr_para))
    {
      return false;
    }

    return true;
  }


} // namespace as_modeling