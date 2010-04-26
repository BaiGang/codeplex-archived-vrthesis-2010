#ifndef _RENDER_GL_H_
#define _RENDER_GL_H_

#include <scoped_ptr.h>
#include "GLSLShader.h"
#include "GLFBO.h"

namespace as_modeling
{
  class ASModeling;

  // for rendering a scene of smoke
  // using the camera parameters...
  class RenderGL
  {
  public:

    bool init();

    void render(int i_view);

    inline const GLuint& get_render_result_tex()
    {return fbo_->GetColorTex();}

    explicit RenderGL(ASModeling *p)
      :asml_(p) {};


  private:
    // no default ctor
    RenderGL(){};

    // ASModeling 
    ASModeling * asml_;

    // render paras
    scoped_ptr<GLSLShader> shader_;
    scoped_ptr<CGLFBO> fbo_;

    // image parameters
    int width_;
    int height_;

  };
} // namespace as_modeling

#endif //_RENDER_GL_H_