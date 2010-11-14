#include "asmWorkspace.h"

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

  // init cuda


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
  init_value();

  // reconstruct the density volume
  reconstruct();

  // store the reconstructed results
  store_frame_result(frame);

  return true;
}

bool Workspace::load_frame_images(uint32_t frame)
{


  return true;
}

bool Workspace::init_value(void)
{
  //

  return true;
}

bool Workspace::reconstruct(void)
{
  //

  return true;
}

bool Workspace::store_frame_result(asmodeling_block::uint32_t frame)
{
  //

  return true;
}