#ifndef __ASM_CONFIGURE_H_
#define __ASM_CONFIGURE_H_

#include <scoped_ptr.h>
#include "../Utils/math/matrix4.h"
#include "../Utils/math/vector4.h"

namespace asmodeling_block
{
  //
  class Configure
  {
  public:
    bool Init(const char * conf_filename, const char * camera_filename);
    bool LoadFile(const char * filename);
    bool LoadCamera(const char * filename);
    bool SaveFile(const char * filename);

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
    int render_interval_array_[32];  // 1 to 32, we use 5, 6, 7, 8
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
    float trans_x_;
    float trans_y_;
    float trans_z_;
    int volume_interval_array_[32];

    ///////////////////////////////////////////////////
    // sizes
    ///////////////////////////////////////////////////
    int initial_vol_level_;
    int max_vol_level_;
    int initial_vol_size_;
    int max_vol_size_;

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
  };
}// namespace asmodeling_block

#endif //__ASM_CONFIGURE_H_