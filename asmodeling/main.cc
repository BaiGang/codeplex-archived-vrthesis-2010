#include <stdafx.h>
#include <cstdio>
#include <cstdlib>

#include "ASModeling.h"


int main( )
{

  // to output logs
  //freopen("../Data/Results/log1.txt", "w", stderr);

  as_modeling::ASModeling modeler;

//modeler.Initialize("../Data/configure.xml", "../Data/camera.txt");
  modeler.Initialize("D:/BaiGang/NEWNEW/asmodeling-56327/Data/configure.xml",
	  "D:/BaiGang/NEWNEW/asmodeling-56327/Data/camera.txt");

  

  modeler.OptimizeProcess(100);

  return 0;
}