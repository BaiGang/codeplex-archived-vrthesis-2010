#ifndef _AS_MODELING_H_
#define _AS_MODELING_H_

#include "stdafx.h"
#include <string>
#include <list>

#include <scoped_ptr.h>
#include <CImg.h>

#include "../L-BFGS-B/ap.h"

#include "../Utils/math/geomath.h"

namespace as_modeling
{

  //=======================================================
  //
  //  A brief description of the whole pipeline
  //
  //    Initialize() loads in configure files and 
  //
  //    Fully optimize the first frame
  //      -- as a single image
  //      -- progressively 
  //    
  //     while not_finished
  //      use previous volume to initialize
  //      store previous volume
  //      optimize current frame
  //      store previous volume
  //    finished
  //
  //=======================================================

  //
  //  Note we must make clear which of those member varaibles 
  //  are constant for all frames and which are distinct
  //  for each frame.
  //  Carry on...
  //

  class ASModeling
  {
  public:

    // Load configure file and camera parameters
    bool Initialize(const char * conf_filename, const char * camera_filename);

    // The whole optimization / reconstruction process
    // Treat the first frame specially,
    // then optimize successor frames one by one :)
    // -- Note this call includes the below two.
    bool OptimizeProcess(int num_of_frames);


    // Functions below are called by OptimizeProcess

    // Optimize a single frame
    // results will be stored at result_volume_
    bool OptimizeSingleFrame(int i_frame);

    // Optimize successors frames
    // results of each frame are temporaraly 
    // stored at result_volume_
    bool OptimizeSuccFrames(int i_frame);

    // Store the resulted volume data
    // for each frame, once per frame
    //    todo : using perfect spatial hashing
    bool StoreVolumeData(int i_frame);


    ASModeling() {};
    ~ASModeling(){};

  private:

    ///////////////////////////////////////////////////
    // consts
    ///////////////////////////////////////////////////
    static const int MAX_NUM_CAMERAS     = 64; // relatively very large, we typically use 8 cameras
    static const int INITIAL_VOL_SIZE    = 32;
    static const int MAX_VOL_SIZE        = 256;
    static const int INITIAL_VOL_LEVEL   = 5;
    static const int MAX_VOL_LEVEL       = 8;

    ////////////////////////////////////////////////////
    //               helper routines
    ////////////////////////////////////////////////////
    bool load_camera_file(const char * filename);
    bool load_configure_file(const char * filename);
    bool load_captured_images(int iframe);

    bool set_cameras();

    // set ground truth image
    bool set_groundtruth_image();

    // set indicator for density existence at each voxel
    bool set_density_indicator(
      int level,
      int * ind_volume,
      std::list<float> & density_vectorized,
      bool is_init_vol);


    // convert (x,y,z) to index
    inline int index3(int x, int y, int z, int length)
    {
      return x + length * (y + length * z);
    }

    // convert 
    inline int index2(int x, int y, int length)
    {
      return x + length * y;
    }

    // data

    // captured images  
    cimg_library::CImgList<float> ground_truth_images_;

    // temporary indicators
    scoped_array< scoped_array<int> > progressive_indicators_;

    // full detailed result
    scoped_array< float > frame_volume_result_;

    scoped_array< float > frame_compressed_result_;

    ////////////////////////////////////////////
    //
    //  participating media parameters
    //
    ////////////////////////////////////////////
    float extinction_;
    float scattering_;
    float alpha_;

    ////////////////////////////////////////////
    //
    //  render parameters
    //
    ////////////////////////////////////////////
    int current_view_;   // deprecated
    int width_;
    int height_;
    int camera_width_;
    int camera_height_;
    int render_interval_;
    int rot_angles_;

    ////////////////////////////////////////////
    //
    //  light parameters
    //
    ////////////////////////////////////////////
    int light_type_;
    float light_intensity_;
    float light_x_;
    float light_y_;
    float light_z_;

    ////////////////////////////////////////////
    //
    //  volume parameters
    //
    ////////////////////////////////////////////
    float box_size_;
    int box_width_;
    int box_height_;
    int box_depth_;
    float trans_x_;
    float trans_y_;
    float trans_z_;
    int volume_interval_;

    ////////////////////////////////////////////
    //
    //  octree parameters
    //
    ////////////////////////////////////////////
    int octree_level_;
    int node_to_divide_;

    ////////////////////////////////////////////
    //
    //  L-BFGS_B parameters
    //
    ////////////////////////////////////////////
    float disturb_;
    float eps_g_;
    float eps_f_;
    float eps_x_;
    int max_iter_;
    int lbfgs_m_;
    int constrain_type_;
    float lower_boundary_;
    float upper_boundary_;

    ////////////////////////////////////////////
    //
    //  Camera parameters
    //
    ////////////////////////////////////////////
    scoped_array<Matrix4> camera_intr_paras_;
    scoped_array<Matrix4> camera_extr_paras_;
    scoped_array<Matrix4> camera_gl_extr_paras_;
    scoped_array<Vector4> camera_positions_;
    scoped_array<Matrix4> gl_projection_mats_;

    // number of cameras for capturation
    int num_cameras_;

    ////////////////////////////////////////////
    //
    //  Parameters for lbfgsbminimize routine
    //
    ////////////////////////////////////////////
    int lbfgsb_info_code_;

    ap::real_1d_array    lbfgsb_x_;
    ap::integer_1d_array lbfgsb_nbd_;
    ap::real_1d_array    lbfgsb_l_;
    ap::real_1d_array    lbfgsb_u_;

  };
} // as_modeling

#endif //_AS_MODELING_H_