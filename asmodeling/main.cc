#include <stdafx.h>
#include <cstdio>
#include <cstdlib>

#include "ASModeling.h"


int main( )
{

  // to output logs
  freopen("../Data/log1.txt", "w", stderr);

  as_modeling::ASModeling modeler;

  modeler.Initialize("../Data/configure.xml", "../Data/camera.txt");

  //modeler.OptimizeProcess(1);

  for (int i_frame = 0; i_frame <= 6; ++ i_frame)
  {
    if (!modeler.OptimizeSingleFrame(i_frame))
    {
      fprintf(stderr, "<<!-- Optimize First Frame Failed.\n");
      return false;
    }
    modeler.StoreVolumeData(i_frame);
  } // for i_frame

  return 0;
}