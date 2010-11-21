//Copyright (c) 2005 HU Yong (huyong@vrlab.buaa.edu.cn)
//  All rights reserved.

#include "GLSLShader.h"
//#include "../CommonTools.h"
#include <fstream>
//#pragma comment(lib, "glew32.lib")

//----------------------------------------------------------------------------
bool InitGLExtensions(void)
{
  if(g_bInitGLExtension == true)
  {
    return true;
  }

  g_bInitGLExtension = true;

  GLenum error = glewInit();
  if(GLEW_OK != error)
  {
    char * tmp = (char*)glewGetErrorString(error);
      fprintf(stderr, "Error : %s\n", tmp);
    return false;
  }
  bool glslInitFlag = false;
  glslInitFlag=CheckGLSL();
  if(glslInitFlag == false)
  {
    return false;
  }
  return true;
}

bool CheckGLSL()
{

  if(g_bUseGLSL == true)
  {
    return true;
  }
  if(!g_bInitGLExtension)
  {
    InitGLExtensions();
  }
  g_bUseGLSL = true;
  if (GL_TRUE != glewGetExtension("GL_ARB_fragment_shader"))
  {
    fprintf(stderr, "Error : GL_ARB_fragment_shader extension is not supported!\n");
    g_bUseGLSL = false;
  }

  if (GL_TRUE != glewGetExtension("GL_ARB_vertex_shader"))
  {
    fprintf(stderr, "Error : GL_ARB_vertex_shader extension is not supported!\n");
    g_bUseGLSL = false;
  }

  if (GL_TRUE != glewGetExtension("GL_ARB_shader_objects"))
  {
    fprintf(stderr, "Error : GL_ARB_shader_objects extension is not supported!\n");
    g_bUseGLSL = false;
  }
  return g_bUseGLSL;
}
bool CheckGL2()
{
  if(!g_bInitGLExtension)
  {
    InitGLExtensions();
  }

  return (GLEW_VERSION_2_0 == GL_TRUE);

}


//----------------------------------------------------------------------------
GLSLShader::GLSLShader()
{
  m_hGLSLFragmentShader = NULL;
  m_hGLSLProgramObject = NULL;
  m_hGLSLVertexShader = NULL;
}

GLSLShader::~GLSLShader()
{
  Release();
}


//----------------------------------------------------------------------------- 

bool GLSLShader::SetUniform1f(const char* varname, GLfloat v0)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  GLint loc = GetUniformLoc(varname);
  if (loc==-1) 
  {
    return false;
  }

  glUniform1f(loc, v0);
  return true;
}

//----------------------------------------------------------------------------- 

bool GLSLShader::SetUniform2f(const char* varname, GLfloat v0, GLfloat v1)
{

  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  GLint loc = GetUniformLoc(varname);
  if (loc==-1) 
  {
    return false;
  }

  glUniform2f(loc, v0, v1);

  return true;
}

//----------------------------------------------------------------------------- 

bool GLSLShader::SetUniform3f(const char* varname, GLfloat v0, GLfloat v1, GLfloat v2)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  GLint loc = GetUniformLoc(varname);
  if (loc==-1) 
  {
    return false;
  }

  glUniform3f(loc, v0, v1, v2);

  return true;
}

//----------------------------------------------------------------------------- 

bool GLSLShader::SetUniform4f(const char* varname, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  GLint loc = GetUniformLoc(varname);
  if (loc==-1) 
  {
    return false;
  }

  glUniform4f(loc, v0, v1, v2, v3);

  return true;
}

//----------------------------------------------------------------------------- 

bool GLSLShader::SetUniform1i(const char* varname, GLint v0)
{ 
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  GLint loc = GetUniformLoc(varname);
  if (loc==-1) 
  {
    return false;
  }

  glUniform1i(loc, v0);

  return true;
}
bool GLSLShader::SetUniform2i(const char* varname, GLint v0, GLint v1)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  GLint loc = GetUniformLoc(varname);
  if (loc==-1) 
  {
    return false;
  }

  glUniform2i(loc, v0, v1);


  return true;
}

//----------------------------------------------------------------------------- 

bool GLSLShader::SetUniform3i(const char* varname, GLint v0, GLint v1, GLint v2)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  GLint loc = GetUniformLoc(varname);
  if (loc==-1) 
  {
    return false;
  }

  glUniform3i(loc, v0, v1, v2);

  return true;
}
bool GLSLShader::SetUniform4i(const char* varname, GLint v0, GLint v1, GLint v2, GLint v3)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  GLint loc = GetUniformLoc(varname);
  if (loc==-1) 
  {
    return false;
  }

  glUniform4i(loc, v0, v1, v2, v3);

  return true;
}

//----------------------------------------------------------------------------- 

bool GLSLShader::SetUniform1fv(const char* varname, GLsizei count, const GLfloat *value)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  GLint loc = GetUniformLoc(varname);
  if (loc==-1) 
  {
    return false;
  }

  glUniform1fv(loc, count, value);

  return true;
}
bool GLSLShader::SetUniform2fv(const char* varname, GLsizei count, const GLfloat *value)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  GLint loc = GetUniformLoc(varname);
  if (loc==-1) 
  {
    return false;
  }

  glUniform2fv(loc, count, value);

  return true;
}

//----------------------------------------------------------------------------- 

bool GLSLShader::SetUniform3fv(const char* varname, GLsizei count, const GLfloat *value)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  GLint loc = GetUniformLoc(varname);
  if (loc==-1) 
  {
    return false;
  }

  glUniform3fv(loc, count, value);

  return true;
}

//----------------------------------------------------------------------------- 

bool GLSLShader::SetUniform4fv(const char* varname, GLsizei count, const GLfloat *value)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  GLint loc = GetUniformLoc(varname);
  if (loc==-1) 
  {
    return false;
  }

  glUniform4fv(loc, count, value);

  return true;
}

//----------------------------------------------------------------------------- 

bool GLSLShader::SetUniform1iv(const char* varname, GLsizei count, const GLint *value)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  GLint loc = GetUniformLoc(varname);
  if (loc==-1) 
  {
    return false;
  }

  glUniform1iv(loc, count, value);

  return true;
}

//----------------------------------------------------------------------------- 

bool GLSLShader::SetUniform2iv(const char* varname, GLsizei count, const GLint *value)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  GLint loc = GetUniformLoc(varname);
  if (loc==-1) 
  {
    return false;
  }

  glUniform2iv(loc, count, value);

  return true;
}

//----------------------------------------------------------------------------- 

bool GLSLShader::SetUniform3iv(const char* varname, GLsizei count, const GLint *value)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  GLint loc = GetUniformLoc(varname);
  if (loc==-1) 
  {
    return false;
  }

  glUniform3iv(loc, count, value);

  return true;
}

//----------------------------------------------------------------------------- 

bool GLSLShader::SetUniform4iv(const char* varname, GLsizei count, const GLint *value)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  GLint loc = GetUniformLoc(varname);
  if (loc==-1) 
  {
    return false;
  }

  glUniform4iv(loc, count, value);

  return true;
}

//----------------------------------------------------------------------------- 

bool GLSLShader::SetUniformMatrix2fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat *value)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  GLint loc = GetUniformLoc(varname);
  if (loc==-1) 
  {
    return false;
  }

  glUniformMatrix2fv(loc, count, transpose, value);

  return true;
}

//----------------------------------------------------------------------------- 

bool GLSLShader::SetUniformMatrix3fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat *value)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  GLint loc = GetUniformLoc(varname);
  if (loc==-1) 
  {
    return false;
  }

  glUniformMatrix3fv(loc, count, transpose, value);

  return true;
}

//----------------------------------------------------------------------------- 

bool GLSLShader::SetUniformMatrix4fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat *value)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  GLint loc = GetUniformLoc(varname);
  if (loc==-1) 
  {
    return false;
  }

  glUniformMatrix4fv(loc, count, transpose, value);

  return true;
}

//----------------------------------------------------------------------------- 

GLint GLSLShader::GetUniformLoc(const char *varname)
{
  GLint loc;

  loc = glGetUniformLocation(m_hGLSLProgramObject, varname);
  if (loc == -1) 
  {
    fprintf(stderr, "GLSLError : can't find uniform variable\n");
  }
  return loc;
}

//----------------------------------------------------------------------------- 

bool GLSLShader::GetUniformfv(const char* varname, GLfloat* values)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  GLint loc = GetUniformLoc(varname);
  if (loc==-1) 
  {
    return false;
  }	

  glGetUniformfv(m_hGLSLProgramObject, loc, values);
  return true;

}

//----------------------------------------------------------------------------- 

bool GLSLShader::GetUniformiv(const char* varname, GLint* values)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  GLint loc = GetUniformLoc(varname);
  if (loc==-1) 
  {
    return false;
  }	

  glGetUniformiv(m_hGLSLProgramObject, loc, values);
  return true;

}

//----------------------------------------------------------------------------- 
bool GLSLShader::SetVertexAttrib1f(GLuint index, GLfloat v0)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  glVertexAttrib1f(index, v0);

  return true;
}

//----------------------------------------------------------------------------- 
bool GLSLShader::SetVertexAttrib2f(GLuint index, GLfloat v0, GLfloat v1)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  glVertexAttrib2f(index, v0, v1);

  return true;
}

//----------------------------------------------------------------------------- 
bool GLSLShader::SetVertexAttrib3f(GLuint index, GLfloat v0, GLfloat v1, GLfloat v2)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  glVertexAttrib3f(index, v0, v1, v2);

  return true;
}

//----------------------------------------------------------------------------- 
bool GLSLShader::SetVertexAttrib4f(GLuint index, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3)
{
  if (!m_hGLSLProgramObject) 
  {
    return false;
  }

  glVertexAttrib4f(index, v0, v1, v2, v3);

  return true;
}
//----------------------------------------------------------------------------- 
string GLSLShader::LoadShaderFile(string strFileName)
{
  // 打开文件
  ifstream fin(strFileName.c_str());

  if(!fin)
    return "";

  string strLine = "";
  string strText = "";


  while(getline(fin, strLine))
  {
    strText = strText + "\n" + strLine;
  }


  fin.close();


  return strText;
}
//-------------------------------------------------------------------------------
void GLSLShader::InitShaders(const string strVertexShader, const string strFragmentShader)
{
  // 保存顶点shader和片断shader的文件字符串
  string strVShader, strFShader;

  if(!strVertexShader.length() || !strVertexShader.length())
  {
    return;
  }

  // 如果我们已经加载过一些shader程序，那么我首先将他们释放掉
  if(m_hGLSLProgramObject || m_hGLSLFragmentShader || m_hGLSLVertexShader)
  {
    Release();
  }

  // 获取指向顶点程序以及片断程序的指针
  m_hGLSLVertexShader = glCreateShader(GL_VERTEX_SHADER);
  m_hGLSLFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  //debug begin
#ifdef _DEBUG
  //	CommonTools::GLERR();
#endif
  //debug end
  // 读取shader程序
  strVShader = LoadShaderFile(strVertexShader.c_str());
  strFShader = LoadShaderFile(strFragmentShader.c_str());
  //debug begin
#ifdef _DEBUG
  //	CommonTools::GLERR();
#endif
  //debug end
  // Do a quick switch so we can do a double pointer below
  const char *szVShader = strVShader.c_str();
  const char *szFShader = strFShader.c_str();

  // Now this assigns the shader text file to each shader pointer
  glShaderSource(m_hGLSLVertexShader, 1, &szVShader, NULL);
  glShaderSource(m_hGLSLFragmentShader, 1, &szFShader, NULL);
  //debug begin
#ifdef _DEBUG
  //CommonTools::GLERR();
#endif
  //debug end
  // Now we actually compile the shader's code
  glCompileShader(m_hGLSLVertexShader);
  glCompileShader(m_hGLSLFragmentShader);
  //debug begin
#ifdef _DEBUG
  //CommonTools::GLERR();
#endif
  //debug end
  // Next we create a program object to represent our shaders
  m_hGLSLProgramObject = glCreateProgram();
  //debug begin
#ifdef _DEBUG
  //CommonTools::GLERR();
#endif
  //debug end
  // We attach each shader we just loaded to our program object
  glAttachObjectARB(m_hGLSLProgramObject, m_hGLSLVertexShader);
  glAttachObjectARB(m_hGLSLProgramObject, m_hGLSLFragmentShader);
  //debug begin
#ifdef _DEBUG
  //CommonTools::GLERR();
#endif
  //debug end
  // Our last init function is to link our program object with OpenGL
  glLinkProgramARB(m_hGLSLProgramObject);
  //debug begin
#ifdef _DEBUG
  //CommonTools::GLERR();
#endif
  //debug end
  int logLength = 0;
  glGetProgramiv(m_hGLSLProgramObject, GL_INFO_LOG_LENGTH, &logLength);

  if (logLength > 1)
  {
    char *szLog = (char*)malloc(logLength);
    int writtenLength = 0;

    glGetProgramInfoLog(m_hGLSLProgramObject, logLength, &writtenLength, szLog);

    fprintf(stderr, "GLSL Error : %s\n", szLog);

    free(szLog);
  }

  // Now, let's turn off the shader initially.
  glUseProgram(0);
}
void GLSLShader::ReloadShaders(const string strVertexShader, const string strFragmentShader)
{
  // 保存顶点shader和片断shader的文件字符串
  string strVShader, strFShader;

  if(!strVertexShader.length() || !strVertexShader.length())
  {
    return;
  }

  // 如果我们已经加载过一些shader程序，那么我首先将他们释放掉
  if(!m_hGLSLProgramObject && !m_hGLSLFragmentShader && !m_hGLSLVertexShader)
  {
    this->InitShaders(strVertexShader, strFragmentShader);
    return;
  }



  // 读取shader程序
  strVShader = LoadShaderFile(strVertexShader.c_str());
  strFShader = LoadShaderFile(strFragmentShader.c_str());

  // Do a quick switch so we can do a double pointer below
  const char *szVShader = strVShader.c_str();
  const char *szFShader = strFShader.c_str();

  // Now this assigns the shader text file to each shader pointer
  glShaderSource(m_hGLSLVertexShader, 1, &szVShader, NULL);
  glShaderSource(m_hGLSLFragmentShader, 1, &szFShader, NULL);
  //debug begin
#ifdef _DEBUG
  //CommonTools::GLERR();
#endif
  //debug end
  // Now we actually compile the shader's code
  glCompileShader(m_hGLSLVertexShader);
  glCompileShader(m_hGLSLFragmentShader);
  //debug begin
#ifdef _DEBUG
  //CommonTools::GLERR();
#endif
  //debug end

  // We attach each shader we just loaded to our program object
  glAttachObjectARB(m_hGLSLProgramObject, m_hGLSLVertexShader);
  glAttachObjectARB(m_hGLSLProgramObject, m_hGLSLFragmentShader);
  //debug begin
#ifdef _DEBUG
  //CommonTools::GLERR();
#endif
  //debug end
  // Our last init function is to link our program object with OpenGL
  glLinkProgramARB(m_hGLSLProgramObject);
  //debug begin
#ifdef _DEBUG
  //CommonTools::GLERR();
#endif
  //debug end
  int logLength = 0;
  glGetProgramiv(m_hGLSLProgramObject, GL_INFO_LOG_LENGTH, &logLength);

  if (logLength > 1)
  {
    char *szLog = (char*)malloc(logLength);
    int writtenLength = 0;

    glGetProgramInfoLog(m_hGLSLProgramObject, logLength, &writtenLength, szLog);

    fprintf(stderr, "GLSL Error : %s\n", szLog);

    free(szLog);
  }

  // Now, let's turn off the shader initially.
  //	glUseProgram(0);
}

//-------------------------------------------------------------------------------------

void GLSLShader::Begin()
{
  if(!g_bUseGLSL)
  {
    return;
  }
  if(m_hGLSLProgramObject == NULL)
  {
    return ;
  }
  glUseProgram(m_hGLSLProgramObject);
}

void GLSLShader::End()
{
  if(!g_bUseGLSL)
  {
    return;
  }
  if(m_hGLSLProgramObject == NULL)
  {
    return ;
  }
  glUseProgram(0);
}

//-------------------------------------------------------------------------------------
void GLSLShader::Release()
{
  // If our vertex shader pointer is valid, free it
  if(m_hGLSLVertexShader)
  {
    glDetachObjectARB(m_hGLSLProgramObject, m_hGLSLVertexShader);
    glDeleteObjectARB(m_hGLSLVertexShader);
    m_hGLSLVertexShader = NULL;
  }

  // If our fragment shader pointer is valid, free it
  if(m_hGLSLFragmentShader)
  {
    glDetachObjectARB(m_hGLSLProgramObject, m_hGLSLFragmentShader);
    glDeleteObjectARB(m_hGLSLFragmentShader);
    m_hGLSLFragmentShader = NULL;
  }

  // If our program object pointer is valid, free it
  if(m_hGLSLProgramObject)
  {
    glDeleteObjectARB(m_hGLSLProgramObject);
    m_hGLSLProgramObject = NULL;
  }
}