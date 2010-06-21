#ifndef _AS_MODELING_H_
#define _AS_MODELING_H_

#include <stdafx.h>
#include <string>
#include <list>
#include <scoped_ptr.h>
#include <ap.h>
#include <math/geomath.h>
#include <CudaImgUtilBMP.h>

#include "GradCompute.h"


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


    //
    // Functions below are called by OptimizeProcess
    //

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

    // for rendering the scene of smoke
    friend class RenderGL;
    // for gradient computation on the CUDA
    friend class ASMGradCompute;

    ASModeling() {};
    ~ASModeling(){};

  private:

    ///////////////////////////////////////////////////
    // consts
    ///////////////////////////////////////////////////
    static const int INITIAL_VOL_LEVEL   = 5;
    static const int MAX_VOL_LEVEL       = 7;
    static const int INITIAL_VOL_SIZE    = 1 << INITIAL_VOL_LEVEL;
    static const int MAX_VOL_SIZE        = 1 << MAX_VOL_LEVEL;

    ////////////////////////////////////////////////////
    //               helper routines
    ////////////////////////////////////////////////////
    bool load_camera_file(const char * filename);
    bool load_configure_file(const char * filename);
    bool load_captured_images(int iframe);

    // convert (x,y,z) to index
    inline int index3(int x, int y, int z, int length)
    {return x + length * (y + length * z);}

    // convert 
    inline int index2(int x, int y, int length)
    {return x + length * y;}

    ////////////////////////////////////////////
    // data (very minor on CPU side)
    ////////////////////////////////////////////

    // captured images
    // We use one big chunk to hold all captured images
    cuda_imageutil::Image_4c8u ground_truth_image_;

    // density field result
    scoped_array<float> frame_volume_result_;
    scoped_array<float> frame_compressed_result_;

    cuda_imageutil::BMPImageUtil result_data_;

    //////////////////////////////////////////////
    // ALL CONFIGURE PARAMETERS
    // LOAD FROM CONFIGURE.XML
    //////////////////////////////////////////////

    ////////////////////////////////////////////
    //  participating media parameters
    ////////////////////////////////////////////
    float extinction_;
    float scattering_;
    float alpha_;

    ////////////////////////////////////////////
    //  render parameters
    ////////////////////////////////////////////
    int current_view_;   // deprecated
    int width_;
    int height_;
    int camera_width_;
    int camera_height_;
    int render_interval_;
    int rot_angles_;

    ////////////////////////////////////////////
    //  light parameters
    ////////////////////////////////////////////
    int light_type_;
    float light_intensity_;
    float light_x_;
    float light_y_;
    float light_z_;

    ////////////////////////////////////////////
    //  volume parameters
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
    //  octree parameters  -- deprecated
    ////////////////////////////////////////////
    int octree_level_;
    int node_to_divide_;

    ////////////////////////////////////////////
    //  L-BFGS_B parameters
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
    //  Camera parameters
    ////////////////////////////////////////////
    scoped_array<Matrix4> camera_intr_paras_;
    scoped_array<Matrix4> camera_extr_paras_;
    scoped_array<Matrix4> camera_gl_extr_paras_;
    scoped_array<Matrix4> camera_inv_gl_extr_paras_;
    scoped_array<Vector4> camera_positions_;
    scoped_array<Matrix4> gl_projection_mats_;

    // number of cameras for capturation
    int num_cameras_;

    // indicates each camera : along axis, 
    //   CAPITAL charactor for positive orientation
    scoped_array<char> camera_orientations_;

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