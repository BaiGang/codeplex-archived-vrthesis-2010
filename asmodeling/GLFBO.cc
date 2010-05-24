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
//#include "StdAfx.h"
#include "GLFBO.h"
//#include "GLCamera.h"
#include <iostream>

CGLFBO::CGLFBO(void) :
m_fbo(0), m_colorTexture(0), m_depthTexture(0),
m_width(0), m_height(0), m_output(0)
{
}

CGLFBO::~CGLFBO(void)
{
  ReleaseFBO();
}

void CGLFBO::Init(int width, int height)
{
  ReleaseFBO();
  m_width = width;
  m_height = height;

  //initialize color texture
  glGenTextures(1, &m_colorTexture);
  glBindTexture(GL_TEXTURE_2D, m_colorTexture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

  //initialize frame buffer
  glGenFramebuffersEXT(1, &m_fbo);
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fbo);

  glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, m_colorTexture, 0);

  // initialize depth renderbuffer
  glGenRenderbuffersEXT(1, &m_depthTexture);
  glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, m_depthTexture);
  glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT32, width, height);
  glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, m_depthTexture);
  CheckFBOErr();

  GLenum err;
  if (GL_NO_ERROR != (err = glGetError()))
    std::cerr << "OpenGL Error!" << std::endl;
}

void CGLFBO::ReleaseFBO()
{
  if (m_fbo != 0)
  {
    glDeleteFramebuffersEXT(1, &m_fbo);
    m_fbo = 0;
  }
  if (m_depthTexture != 0)
  {
    glDeleteTextures(1, &m_depthTexture);
    m_depthTexture = 0;
  }
  if (m_colorTexture != 0)
  {
    glDeleteTextures(1, &m_colorTexture);
    m_colorTexture = 0;
  }
  if (m_output != 0)
  {
    delete []m_output;
    m_output = 0;
  }
}

void CGLFBO::BeginDraw2FBO()
{
  if (!m_fbo || !m_depthTexture || !m_colorTexture)
    return;

  //pCamera->SetAspectRatio((float)m_width / m_height);

  //glBindTexture(GL_TEXTURE_2D, 0);
  glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, m_depthTexture);
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fbo);
  glPushAttrib(GL_VIEWPORT_BIT);
  //glViewport(0, 0, m_width, m_height);
  glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void CGLFBO::EndDraw2FBO()
{
  glPopAttrib();
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
  glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);

}

bool CGLFBO::IsReady()
{
  return (m_fbo != 0 && m_width != 0 && m_height != 0 && m_colorTexture != 0 && m_depthTexture != 0);
}

bool CGLFBO::CheckFBOErr()
{
  using namespace std;

  GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);

  bool isOK = false;

  switch(status) 
  {                                          
  case GL_FRAMEBUFFER_COMPLETE_EXT: // Everything's OK
    isOK = true;
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
    cerr << "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT\n";
    isOK = false;
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
    cerr << "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT\n";
    isOK = false;
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
    cerr << "GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT\n";
    isOK = false;
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
    cerr << "GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT\n";
    isOK = false;
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
    cerr << "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT\n";
    isOK = false;
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
    cerr << "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT\n";
    isOK = false;
    break;
  case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
    cerr << "GL_FRAMEBUFFER_UNSUPPORTED_EXT\n";
    isOK = false;
    break;
  default:
    cerr << "Unknown ERROR";
    isOK = false;
  }

  return isOK;
}

unsigned char *CGLFBO::ReadPixels()
{
  if (m_output == 0)
    m_output = new unsigned char[m_width * m_height * 4];
  memset(m_output, 0, m_width * m_height * 4 * sizeof(unsigned char));

  glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, m_depthTexture);
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fbo);
  glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
  glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, m_output);
  return m_output;
}