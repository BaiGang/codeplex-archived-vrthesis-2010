#include "ASModeling.h"

/////////////////////////////////////////////////////////////////////////////
//
//  set_density_indicator()
//    set an volume, so that if the density of a cell is not zero, the 
//  corresponding cell in indicator is 1.
//
//  param- level : current level of the density volume
//  param- ind_volume:  the incidator volume
//
//  preliminaries:  the corresponding images have been loaded
//
/////////////////////////////////////////////////////////////////////////////


namespace as_modeling
{
  bool ASModeling::set_density_indicator(int level, uchar * ind_volume)
  {
    int length = (1<<level);

    // for each cell (i,j,k) 
    for (int k = 0; k < length; ++k)
    {
      for (int j = 0; j < length; ++j)
      {
        for (int i = 0; i < length; ++i)
        {
          // calc the coordinates of the 8 corners

          // for each camera
          for (int i_camera = 0; i_camera < num_cameras_; ++i_camera)
          {

          } // for i_camera

        } // for i
      } // for j
    } // for k

    
    return true;
  }
} // namespace as_modeling