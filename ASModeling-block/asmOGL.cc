#include "asmOGL.h"

#include <cstdio>
#include <gl/glew.h>
#include <gl/glut.h>
#include "../Utils/openGL/GLSLShader.h"

using namespace asmodeling_block;

bool GLSys::_inited = false;

bool GLSys::Init(void)
{
  if (_inited)
    return _inited;

  fprintf(stderr, "Initiating OpenGL context.\n");

  // glut for creation of OpenGL context
  int tmpargc = 2;
  int * argc = &tmpargc;
  char *tmp1 = "aaaa";
  char *tmp2 = "bbbbbb";
  char *argv[2];
  argv[0] = tmp1;
  argv[1] = tmp2;
  glutInit(argc, argv);
  glutCreateWindow("Dummy window..");

  if (!InitGLExtensions())
  {
    fprintf(stderr, "Initiating GL error...\n");
    return false;
  }

  return _inited = true;
}

void GLSys::Release(void)
{
}