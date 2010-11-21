#include "asmWorkspace.h"


#include "../Utils/image/BMPImage.h"
#include "../Utils/image/PFMImage.h"

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
  //
  init_cuda_resources();

  // init data
  captured_image_.reset(new float[conf_.num_cameras_ * conf_.width_ * conf_.height_]);
  captured_BMPimages_.reset(new BMPImage[conf_.num_cameras_]);

  // set the global pointer to the current instance
  // so the friend function GradCompute can access members here
  
  
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
  char buf[256];
  for (int icamera = 0; icamera < conf_.num_cameras_; ++icamera)
  {
    int image_offset = icamera * conf_.width_ * conf_.height_;

    sprintf(buf, "../Data/Camera%02d/Frame%05d.bmp", icamera, frame);
    if (!captured_BMPimages_[icamera].ReadImage(buf))
    {
      fprintf(stderr, "<!> Error while loading Frame %d, Camera %d.\n", frame, icamera);
      return false;
    }

    for (int v = 0; v < captured_BMPimages_[icamera].GetHeight(); ++v)
    {
      for (int u = 0; u < captured_BMPimages_[icamera].GetWidth(); ++u)
      {
        captured_image_[image_offset + v * conf_.width_ + u] = static_cast<float>(captured_BMPimages_[icamera].GetPixel(u,v).g);
      }
    }
  }

  return true;
}

bool Workspace::init_value_firstlevel(uint32_t level)
{
  // clear block_array
  // for each block
  //
  //    for each cell in the block
  //       calc the world-coordinate position of the cell
  //       for each view
  //           calc the convex-hull of the project area
  //           if has non-zero pixels
  //                

  block_length = (1 << level) / NBLOCK;
  
  float block_size = conf_.box_size_ / static_cast<float>(NBLOCK);
  float cell_size = conf_.box_size_ / static_cast<float>(1 << level);
  
  float vol_nll_x = conf_.trans_x_ - 0.5f * conf_.box_size_;
  float vol_nll_y = conf_.trans_y_ - 0.5f * conf_.box_size_;
  float vol_nll_z = conf_.trans_z_ - 0.5f * conf_.box_size_;
  
  blocks_cpu_.clear();
  blocks_gpu_.clear();
  
  along_x_.clear();
  along_y_.clear();
  along_z_.clear();
  
  for (uint32_t iblock = 0; iblock < NBLOCK; ++iblock)
  {
	float block_nll_x = vol_nll_x + iblock * block_size;
	for (uint32_t jblock = 0; jblock < NBLOCK; ++jblock)
	{
		float block_nll_y = vol_nll_y + jblock * block_size;
		for (uint32_t kblock = 0; kblock < NBLOCK; ++kblock)
		{
			float block_nll_z = vol_nll_z + kblock * block_size;
			
			bool empty = true;
			
			for (uint32_t icell = 0; icell < block_length; ++icell)
			{
				float cell_nll_x = block_nll_x + icell * cell_size;
				for (uint32_t jcell = 0; jcell < block_length; ++jcell)
				{
					float cell_nll_y = block_nll_y + jcell * cell_size;
					for (uint32_t kcell = 0; kcell < block_length; ++kcell)
					{
						float cell_nll_z = block_nll_z + kcell * cell_size;
						
						for (int iview = 0; iview < conf_.num_cameras_; ++iview)
						{
							// calc the projected area of the cell
							
							// convex hull
							
							// if non-empty, empty = false
							
						} // for iview
						
					} // for kcell
				} // for jcell
			} // for icell
			
			if (!empty)
			{
				// add current block to the array
				
				//blocks_cpu_.push-back();
				
				int pos = blocks_cpu_.size() - 1;
				
				// set along_x / y /z 
				
			}
			
		} // for kblock
	} // for jblock
  } // for iblock
  
  // set x_array_ here ....
  
  return true;
}

bool Workspace::init_value_upperlevel(uint32_t level)
{
  // for each block
  //    array-data[i,j,k] = old-array-data[i/2, j/2, k/2]
  
  // haha, excellent design here...

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

  fprintf(stderr, "<-> Optimizing level %u finished, return code: %d", level, lbfgsb_info_code_);

  return true;
}

bool Workspace::store_frame_result(asmodeling_block::uint32_t frame)
{
  //

  return true;
}

void Workspace::straight_render(int iview)
{
	//     for each block layer along this view
	//         for each slice on this layer
	//             for each block
	//                 render the quad
	//     calc the objective function value
}

void Workspace::perturb_render(int iview, int block_layer, int slice, int pu, int pv)
{
	//     for each block layer along this view
	//         for each slice on this layer
	//                 for each block
	//                     render the 
}