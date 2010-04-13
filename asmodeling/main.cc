#include <cstdio>
#include <cstdlib>

#include "ASModeling.h"

int main( )
{
  as_modeling::ASModeling modeler;

  modeler.Initialize("../Data/configure.xml", "../Data/camera.txt");

  return 0;
}