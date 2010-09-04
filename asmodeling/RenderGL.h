#ifndef _RENDER_GL_H_
#define _RENDER_GL_H_
#include <stdafx.h>
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
    bool release();

    //! Generate a list for all primitives
    bool level_init(int level, GLuint vol_tex);

    void render_unperturbed(int i_view, GLuint vol_tex, int length);
    void render_perturbed(int i_view, GLuint vol_tex, int length, int interval, int slice, int pu, int pv);

    inline const GLuint& get_render_result_tex()
    {return rr_fbo_->GetColorTex();}
    inline const GLuint& get_perturb_result_tex()
    {return pr_fbo_->GetColorTex();}

    explicit RenderGL(ASModeling *p);

    ///// For Debugging...
    inline float * get_render_res()
    {
      return rr_fbo_->ReadPixels();
    }

  private:
    // no default ctor
    RenderGL(){};

    // ASModeling 
    ASModeling * asml_;

    // render paras
    scoped_ptr<GLSLShader> shader_x_render_; // along x axis, unperturbed
    scoped_ptr<GLSLShader> shader_y_render_; // along y axis, ..
    scoped_ptr<GLSLShader> shader_z_render_; // along z axis, ..

    scoped_ptr<GLSLShader> shader_x_perturbed_; // along x axis, perturbed
    scoped_ptr<GLSLShader> shader_y_perturbed_; // along y axis, ..
    scoped_ptr<GLSLShader> shader_z_perturbed_; // along z axis, ..

    void set_shader_unperturbed( GLSLShader * shader, GLuint vol_tex, int length);
    void set_shader_perturbed( GLSLShader * shader, GLuint vol_tex, int length);

    // display list index
    GLuint display_list_index_;

  private:
    scoped_ptr<CGLFBO> rr_fbo_;     // render result fbo, for calc f
    scoped_ptr<CGLFBO> pr_fbo_;     // perturbed result fbo, for calc g

    // image parameters
    int width_;
    int height_;

    ////////////////////
    int counter;

  };
} // namespace as_modeling

#endif //_RENDER_GL_H_