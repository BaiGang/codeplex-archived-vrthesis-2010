#include "RenderGL.h"
#include "ASModeling.h"
#include <GL/glut.h>

namespace as_modeling
{
  bool RenderGL::init()
  {
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
      // gl init error
      return false;
    }

    width_ = asml_->width_;
    height_ = asml_->height_;

    GLSLShader * tmpshader = new GLSLShader();
    shader_x_.reset(tmpshader);
    tmpshader = new GLSLShader();
    shader_y_.reset(tmpshader);
    tmpshader = new GLSLShader();
    shader_z_.reset(tmpshader);

    shader_x_->InitShaders(
      "../Data/GLSLShaders/RayMarchingBlend.vert",
      "../Data/GLSLShaders/RayMarchingBlendX.frag"
      );
    shader_x_->InitShaders(
      "../Data/GLSLShaders/RayMarchingBlend.vert",
      "../Data/GLSLShaders/RayMarchingBlendY.frag"
      );
    shader_x_->InitShaders(
      "../Data/GLSLShaders/RayMarchingBlend.vert",
      "../Data/GLSLShaders/RayMarchingBlendZ.frag"
      );

    CGLFBO * tmpfbo = new CGLFBO();
    rr_fbo_.reset(tmpfbo);
    rr_fbo_->Init(width_, height_);
    rr_fbo_->CheckFBOErr();

    tmpfbo = new CGLFBO();
    pr_fbo_.reset(tmpfbo);
    pr_fbo_->Init(width_, height_);
    pr_fbo_->CheckFBOErr();

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_DST_ALPHA, GL_ONE, GL_DST_ALPHA, GL_ZERO);

    return true;
  }

  void RenderGL::render(int i_view)
  {
    float proj_mat[16];
    float mv_mat[16];

    // get PROJECTION and MODELVIEW matrices from 
    // the ASMoceling object...
    asml_->gl_projection_mats_[i_view].GetData(proj_mat);
    asml_->camera_gl_extr_paras_[i_view].GetData(mv_mat);



    // set viewport, modelview and projection
    glViewport(0, 0, asml_->width_, asml_->height_);
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(proj_mat);
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(mv_mat);

    // adjust the volume
    glTranslatef(asml_->trans_x_, asml_->trans_y_, asml_->trans_z_);
    //glTranslatef(asml_->trans_x_, 0.0f, 0.0f);
    //glTranslatef(0.0f, asml_->trans_y_, 0.0f);
    //glTranslatef(0.0f, 0.0f, asml_->trans_z_);
    glRotatef(asml_->rot_angles_, 0.0f, 0.0f, 1.0f);

    // get the inversed camera matrix
    float inv_camera[16];
    Matrix4 tmp_camera = asml_->camera_gl_extr_paras_[i_view];
    tmp_camera.Inverse();
    tmp_camera.GetData(inv_camera);


    //// bind fbo
    //fbo_->BeginDraw2FBO();
    //{

    //  // render smoke
    //  shader_->Begin();
    //  {
    //    if (asml_->light_type_ == 1)
    //    {
    //      // point light
    //      shader_->SetUniform3f("lightPosWorld", asml_->light_x_, asml_->light_y_, asml_->light_z_);
    //    }
    //    else
    //    {
    //      // directional light
    //      Vector4 dir;
    //      dir.x = asml_->light_x_ - asml_->trans_x_;
    //      dir.y = asml_->light_y_ - asml_->trans_y_;
    //      dir.z = asml_->light_z_ - asml_->trans_z_;
    //      dir.w = 1.0;
    //      dir.normaVec();
    //      shader_->SetUniform3f("lightDirWorld", dir[0], dir[1], dir[2]);
    //    }
    //    shader_->SetUniform3f("lightIntensity", asml_->light_intensity_, asml_->light_intensity_, asml_->light_intensity_);
    //    
    //    // camera position
    //    shader_->SetUniform3f("cameraPos",
    //      asml_->camera_positions_[i_view].x,
    //      asml_->camera_positions_[i_view].y,
    //      asml_->camera_positions_[i_view].z);

    //    shader_->SetUniform1f("absorptionCoefficient", asml_->extinction_);
    //    shader_->SetUniform1f("scatteringCoefficient", asml_->scattering_);

    //    shader_->SetUniformMatrix4fv("cameraInv", 1, GL_FALSE, inv_camera);

    //    // set to shaders.....



    //  }
    //  shader_->End();

    //}
    //// unbind fbo
    //fbo_->EndDraw2FBO();

  }

} // namespace as_modeling