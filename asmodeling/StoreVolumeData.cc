#include "ASModeling.h"


namespace as_modeling
{
  bool ASModeling::StoreVolumeData(int i_frame)
  {
    char path_buf[100];

    fprintf(stderr, "Store volume data. Frame %d\n", i_frame);

    sprintf(path_buf, "../Data/Results/Frame%08d.BMP", i_frame);

    result_data_.ClearImage();
    int length = MAX_VOL_SIZE;
    for (int z = 0; z < length; ++z)
    {
      for (int y = 0; y < length; ++y)
      {
        int row = z * length + y;
        for (int x = 0; x < length; ++x)
        {
          unsigned char pix = frame_compressed_result_[index3(x,y,z,length)];
          ///////////
          result_data_.GetPixelAt(row, x)[0] = pix * 255;
          result_data_.GetPixelAt(row, x)[1] = pix * 255;
          result_data_.GetPixelAt(row, x)[2] = pix * 255;
        }
      }
    }
    result_data_.SaveImage(path_buf);

    return true;
  }

} // namespace as_modeling