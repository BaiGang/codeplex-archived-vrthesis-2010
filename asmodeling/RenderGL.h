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

    void render(int i); // dummy
    void render_unperturbed(int i_view);
    void render_perturbed(int i_view);

    inline const GLuint& get_render_result_tex()
    {return rr_fbo_->GetColorTex();}
    inline const GLuint& get_perturb_result_tex()
    {return pr_fbo_->GetColorTex();}

    explicit RenderGL(ASModeling *p)
      :asml_(p) {};


  private:
    // no default ctor
    RenderGL(){};

    // ASModeling 
    ASModeling * asml_;

    // render paras
    scoped_ptr<GLSLShader> shader_x_render_; // along x axis, unperturbed
    scoped_ptr<GLSLShader> shader_y_render_; // along y axis, ..
    scoped_ptr<GLSLShader> shader_z_render_; // along z axis, ..
    scoped_ptr<GLSLShader> shader_x_pertuerbed_; // along x axis, perturbed
    scoped_ptr<GLSLShader> shader_y_pertuerbed_; // along y axis, ..
    scoped_ptr<GLSLShader> shader_z_pertuerbed_; // along z axis, ..
    scoped_ptr<CGLFBO> rr_fbo_;     // render result fbo, for calc f
    scoped_ptr<CGLFBO> pr_fbo_;     // perturbed result fbo, for calc g

    // image parameters
    int width_;
    int height_;

  };
} // namespace as_modeling

#endif //_RENDER_GL_H_