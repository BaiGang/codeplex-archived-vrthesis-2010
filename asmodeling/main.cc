#include <cstdio>
#include <cstdlib>

#include <CImg.h>

#include "ASModeling.h"

#ifdef __cplusplus
int a = 10;
#else
int a = 20;
#endif

int main( )
{
  as_modeling::ASModeling modeler;

  //cimg_library::CImgList<unsigned char> test;
  //test.assign(5);

  //for(int i=0; i<5; ++i)
  //{
  //  char path_buf[200];
  //  sprintf_s(path_buf, 200, "../Data/Camera%02d/Frame%05d.bmp", i, 0);
  //  test(i).assign(path_buf);
  //}


  modeler.Initialize("../Data/configure.xml", "../Data/camera.txt");

  modeler.OptimizeProcess(1);




  return 0;
}