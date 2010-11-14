#include "asmWorkspace.h"


#include "../Utils/image/BMPImage.h"

using namespace asmodeling_block;

bool Workspace::Init(const char * conf_filename, const char * camera_filename)
{
  // load configure and cameras
  if (!conf_.Init(conf_filename, camera_filename))
  {
    return false;
  }

  // init opengl
  if (!GLSys::Init())
  {
    fprintf(stderr, "<!> Error when initiating OpenGL..\n");
    return false;
  }

  // set cameras
  conf_.camera_orientations_.reset(new char[conf_.num_cameras_]);
  for (int i = 0; i < conf_.num_cameras_; ++i)
  {
    // camera orientations
    Vector4 dir(
      conf_.camera_positions_[i].x - conf_.trans_x_,
      conf_.camera_positions_[i].y - conf_.trans_y_,
      conf_.camera_positions_[i].z - conf_.trans_z_,
      1.0
      );

    dir.normaVec();

    if (abs(dir.x)>abs(dir.y) && abs(dir.x)>abs(dir.z))
    {
      // along x
      if (dir.x < 0.0)
        conf_.camera_orientations_[i] = 'X';
      else
        conf_.camera_orientations_[i] = 'x';
    }
    else if (abs(dir.y)>abs(dir.x) && abs(dir.y)>abs(dir.z))
    {
      // along y
      if (dir.y < 0.0)
        conf_.camera_orientations_[i] = 'Y';
      else
        conf_.camera_orientations_[i] = 'y';

    }
    else if (abs(dir.z)>abs(dir.x) && abs(dir.z)>abs(dir.y))
    {
      // along z
      if (dir.z < 0.0)
        conf_.camera_orientations_[i] = 'Z';
      else
        conf_.camera_orientations_[i] = 'z';

    }
    else
    {
      // should not have been here
      fprintf(stderr, " ERROR : axis specifying error!\n\n");
      return false;
    }
  }

  for (int i = 0; i < conf_.num_cameras_; ++i)
  {
    fprintf(stderr, "Camera %02d : %c\n", i, conf_.camera_orientations_[i]);
  }

  // init cuda
  cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );

  // init data
  captured_image_.reset(new float[conf_.width_ * conf_.height_]);

  return true;
}

bool Workspace::ProcessFrame(uint32_t frame)
{
  fprintf(stderr, "<-> Loading images for frame %u.\n", frame);
  if (!load_frame_images(frame))
  {
    fprintf(stderr, "<!> Error when loading frame %u.\n", frame);
  }

  // init values
  init_value_firstlevel(conf_.initial_vol_level_);

  // reconstruct the density volume
  reconstruct(conf_.initial_vol_level_);

  for (int ilevel = conf_.initial_vol_level_+1; ilevel <= conf_.max_vol_level_; ++ilevel)
  {
    init_value_upperlevel(ilevel);

    reconstruct(ilevel);
  }

  // store the reconstructed results
  store_frame_result(frame);

  return true;
}

bool Workspace::load_frame_images(uint32_t frame)
{
  BMPImage tmp_img;


  char buf[256];
  for (int icamera = 0; icamera < conf_.num_cameras_; ++icamera)
  {
    int image_offset = icamera * conf_.width_ * conf_.height_;

    sprintf(buf, "../Data/Camera%02d/Frame%05d.bmp", icamera, frame);
    if (!tmp_img.ReadImage(buf))
    {
      fprintf(stderr, "<!> Error while loading Frame %d, Camera %d.\n", frame, icamera);
      return false;
    }

    for (int v = 0; v < tmp_img.GetHeight(); ++v)
    {
      for (int u = 0; u < tmp_img.GetWidth(); ++u)
      {

      }
    }
  }

  return true;
}

bool Workspace::init_value_firstlevel(uint32_t level)
{
  //

  return true;
}

bool Workspace::init_value_upperlevel(uint32_t level)
{
  //

  return true;
}

bool Workspace::reconstruct(uint32_t level)
{
  // condition:
  //  the x_array_ already contains the intial guess of density value

  int n_variables = x_array_.gethighbound() + 1 - x_array_.getlowbound();

  lbfgsb_nbd_.setbounds(1, n_variables);
  lbfgsb_l_.setbounds(1, n_variables);
  lbfgsb_u_.setbounds(1, n_variables);

  for (int i = 1; i <= n_variables; ++i)
  {
    lbfgsb_nbd_(i) = conf_.constrain_type_;
    lbfgsb_l_(i) = conf_.lower_boundary_;
    lbfgsb_u_(i) = conf_.upper_boundary_;
  }

  lbfgsbminimize (n_variables,
    conf_.lbfgs_m_,
    x_array_,
    conf_.eps_g_,
    conf_.eps_f_,
    conf_.eps_x_,
    conf_.max_iter_,
    lbfgsb_nbd_,
    lbfgsb_l_,
    lbfgsb_u_,
    lbfgsb_info_code_,
    GradCompute );

  return true;
}

bool Workspace::store_frame_result(asmodeling_block::uint32_t frame)
{
  //

  return true;
}