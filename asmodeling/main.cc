#include <cstdio>
#include <cstdlib>

#include "ASModeling.h"

#ifdef __cplusplus
int a = 10;
#else
int a = 20;
#endif

int main( )
{
  as_modeling::ASModeling modeler;

  //fprintf(stderr, "%d\n", a);

  modeler.Initialize("../Data/configure.xml", "../Data/camera.txt");

  modeler.OptimizeProcess(2);

  return 0;
}