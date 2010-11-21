//Copyright (c) 2005 HU Yong (huyong@vrlab.buaa.edu.cn)
//  All rights reserved.

// Edited by BAI Gang
//  - console log print

#ifndef _GLSLSHADER_H
#define _GLSLSHADER_H
//#include <windows.h>
#include <gl/glew.h>
#include <string>
#include <iostream>
using namespace std;
/*
定义一些全局函数，用来测试显卡对GL扩展的支持
*/
static bool g_bUseGLSL = false;
static bool g_bInitGLExtension = false;
bool CheckGLSL(void);
bool CheckGL2(void);
//初始化glew
bool InitGLExtensions(void);

class GLSLShader
{
public:
  GLSLShader(void);
  ~GLSLShader(void);
  /*初始化shader
  strVertexShader 顶点shader文件名
  strFragment 片断shader文件名
  */
  void InitShaders(const string strVertexShader, const string strFragmentShader);
  void ReloadShaders(const string strVertexShader, const string strFragmentShader);
  //释放已经存在的shader程序
  void Release(void);

  //开始和停止使用shader
  void Begin(void);
  void End(void);

  GLint GetUniformLoc(const char* varname);
  // 设置 Uniform 变量
  bool SetUniform1f(const char* varname, GLfloat v0);  
  bool SetUniform2f(const char* varname, GLfloat v0, GLfloat v1); 
  bool SetUniform3f(const char* varname, GLfloat v0, GLfloat v1, GLfloat v2); 
  bool SetUniform4f(const char* varname, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);    

  bool SetUniform1i(const char* varname, GLint v0);
  bool SetUniform2i(const char* varname, GLint v0, GLint v1);
  bool SetUniform3i(const char* varname, GLint v0, GLint v1, GLint v2);
  bool SetUniform4i(const char* varname, GLint v0, GLint v1, GLint v2, GLint v3);

  bool SetUniform1fv(const char* varname, GLsizei count, const GLfloat *value);
  bool SetUniform2fv(const char* varname, GLsizei count, const GLfloat *value);
  bool SetUniform3fv(const char* varname, GLsizei count, const GLfloat *value);
  bool SetUniform4fv(const char* varname, GLsizei count, const GLfloat *value);
  bool SetUniform1iv(const char* varname, GLsizei count, const GLint *value);
  bool SetUniform2iv(const char* varname, GLsizei count, const GLint *value);
  bool SetUniform3iv(const char* varname, GLsizei count, const GLint *value);
  bool SetUniform4iv(const char* varname, GLsizei count, const GLint *value);

  bool SetUniformMatrix2fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat *value);
  bool SetUniformMatrix3fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat *value);
  bool SetUniformMatrix4fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat *value);

  // 返回uniform变量
  bool GetUniformfv(const char* varname, GLfloat* values);
  bool GetUniformiv(const char* varname, GLint* values); 

  // 顶点 Attribute 变量
  bool SetVertexAttrib1f(GLuint index, GLfloat v0);
  bool SetVertexAttrib2f(GLuint index, GLfloat v0, GLfloat v1);
  bool SetVertexAttrib3f(GLuint index, GLfloat v0, GLfloat v1, GLfloat v2);
  bool SetVertexAttrib4f(GLuint index, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);

protected:
private:
  //解析文件，并返回一个字符串
  string LoadShaderFile(string strFileName);
  GLhandleARB m_hGLSLProgramObject;
  GLhandleARB m_hGLSLVertexShader;
  GLhandleARB m_hGLSLFragmentShader;
};

#endif