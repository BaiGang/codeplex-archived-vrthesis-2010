#include "ASModeling.h"

#include "../Utils/image/PFMImage.h"


namespace as_modeling
{
  bool ASModeling::StoreVolumeData(int i_frame)
  {
    char path_buf[100];

    fprintf(stderr, "Store volume data. Frame %d\n", i_frame);

    sprintf(path_buf, "../Data/Results/Frame%08d.pfm", i_frame);

    PFMImage * res_img = new PFMImage(MAX_VOL_SIZE, MAX_VOL_SIZE*MAX_VOL_SIZE, 0, frame_volume_result_.get());
    res_img->WriteImage(path_buf);

    //// also, we should show the render result of the reconstructed volume
    //for (int i_view = 0; i_view < num_cameras_; ++i_view)
    //{
    //  sprintf(path_buf, "../Data/Results/Frame%08d_View%02d.pfm", i_frame, i_view);
    //  
    //}

    return true;
  }

} // namespace as_modeling