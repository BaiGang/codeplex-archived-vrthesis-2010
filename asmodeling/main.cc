#include <stdafx.h>
#include <cstdio>
#include <cstdlib>

#include "ASModeling.h"


int main( )
{

  // to output logs
  freopen("../Data/log20101214.txt", "w", stderr);

  as_modeling::ASModeling modeler;

  modeler.Initialize("../Data/configure.xml", "../Data/camera.txt");

  modeler.OptimizeProcess(1);

  return 0;
}