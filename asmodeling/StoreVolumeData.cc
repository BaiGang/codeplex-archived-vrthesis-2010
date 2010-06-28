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

    return true;
  }

} // namespace as_modeling