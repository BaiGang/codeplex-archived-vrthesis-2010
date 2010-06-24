#include <stdafx.h>
#include <cstdio>
#include <cstdlib>

#include "ASModeling.h"

//int kaka()
//{
//  // glut for creation of OpenGL context
//  int tmpargc = 2;
//  int * argc = &tmpargc;
//  char *tmp1 = "aaaa";
//  char *tmp2 = "bbbbbb";
//  char *argv[2];
//  argv[0] = tmp1;
//  argv[1] = tmp2;
//  glutInit(argc, argv);
//  glutCreateWindow("Dummy window..");
//
//  if (!InitGLExtensions())
//  {
//    return false;
//  }
//
//
//  return 0;
//}

int main( )
{

  //return kaka();

  as_modeling::ASModeling modeler;

  modeler.Initialize("../Data/configure.xml", "../Data/camera.txt");

  modeler.OptimizeProcess(1);

  return 0;
}