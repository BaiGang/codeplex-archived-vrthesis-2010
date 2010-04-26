//////////////////////////////////////////////////////////////////////
// GLFBO.h
//
// Fangyang SHEN, VRLab, Beihang University
// me@shenfy.net
//
// (C) Copyright VRLab, Beihang University 2009.
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
//  Brief Description
//
//	Simple Wrapper for OpenGL Frame Buffer Object
//
//////////////////////////////////////////////////////////////////////
#ifndef __GL_FBO_H_
#define __GL_FBO_H_

#include <gl/glew.h>

class CGLCamera;

class CGLFBO
{
public:
  CGLFBO(void);
  ~CGLFBO(void);

  void Init(int width, int height);
  void ReleaseFBO(void);
  void BeginDraw2FBO(void);
  void EndDraw2FBO(void);
  bool CheckFBOErr(void);
  bool IsReady(void);

  float *ReadPixels(void);

  int GetWidth(void) {return m_width;}
  int GetHeight(void) {return m_height;}

  // added by Bai, Gang
  // for convenient usage
  inline const GLuint& GetColorTex() const
  {return m_colorTexture;}

protected:
  GLuint m_fbo;
  GLuint m_colorTexture;
  GLuint m_depthTexture;

  float *m_output;
  int m_width, m_height;
};

#endif //__GL_FBO_H_