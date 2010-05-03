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
    shader_x_render_.reset(tmpshader);
    tmpshader = new GLSLShader();
    shader_y_render_.reset(tmpshader);
    tmpshader = new GLSLShader();
    shader_z_render_.reset(tmpshader);
    tmpshader = new GLSLShader();
    shader_x_pertuerbed_.reset(tmpshader);
    tmpshader = new GLSLShader();
    shader_y_pertuerbed_.reset(tmpshader);
    tmpshader = new GLSLShader();
    shader_z_pertuerbed_.reset(tmpshader);

    shader_x_render_->InitShaders(
      "../Data/GLSLShaders/RayMarchingBlend.vert",
      "../Data/GLSLShaders/RayMarchingBlendXU.frag"
      );
    shader_y_render_->InitShaders(
      "../Data/GLSLShaders/RayMarchingBlend.vert",
      "../Data/GLSLShaders/RayMarchingBlendYU.frag"
      );
    shader_z_render_->InitShaders(
      "../Data/GLSLShaders/RayMarchingBlend.vert",
      "../Data/GLSLShaders/RayMarchingBlendZU.frag"
      );

    shader_x_pertuerbed_->InitShaders(
      "../Data/GLSLShaders/RayMarchingBlend.vert",
      "../Data/GLSLShaders/RayMarchingBlendX.frag"
      );
    shader_y_pertuerbed_->InitShaders(
      "../Data/GLSLShaders/RayMarchingBlend.vert",
      "../Data/GLSLShaders/RayMarchingBlendY.frag"
      );
    shader_z_pertuerbed_->InitShaders(
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

    // NOTE : ����Ҫ���Ǻá�
    glBlendFuncSeparate(GL_DST_ALPHA, GL_ONE, GL_DST_ALPHA, GL_ZERO);

    return true;
  }

  bool RenderGL::release( )
  {
    // reallocate the resource
    return true;
  }

  void RenderGL::render_unperturbed(int i_view, GLuint vol_tex)
  {
    float proj_mat[16];
    float mv_mat[16];

    // get MODELVIEW and PROJECTION matrices
    asml_->gl_projection_mats_[i_view].GetData(proj_mat);
    asml_->camera_gl_extr_paras_[i_view].GetData(mv_mat);

    glViewport(0, 0, width_, height_);
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(proj_mat);
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(mv_mat);

    //// adjust the volume
    //glTranslatef(asml_->trans_x_, asml_->trans_y_, asml_->trans_z_);

    // get the inversed camera matrix
    float inv_camera[16];
    asml_->camera_inv_gl_extr_paras_[i_view].GetData(inv_camera);

    // choose shader
    Vector4 dir(asml_->camera_positions_[i_view].x - asml_->trans_x_,
      asml_->camera_positions_[i_view].y - asml_->trans_y_,
      asml_->camera_positions_[i_view].z - asml_->trans_z_);
    dir.normaVec();

    GLSLShader * shader = NULL;
    float step_size = 0.0f;
    char orientation = 'N';

    if (abs(dir.x)>abs(dir.y) && abs(dir.x)>abs(dir.z))
    {
      // along x
      shader = shader_x_render_.get();
      step_size = asml_->box_size_ / static_cast<float>(asml_->box_width_);
      orientation = 'X';
    }
    else if (abs(dir.y)>abs(dir.x) && abs(dir.y)>abs(dir.z))
    {
      // along y
      shader = shader_y_render_.get();
      step_size = asml_->box_size_ / static_cast<float>(asml_->box_height_);
      orientation = 'Y';
    }
    else if (abs(dir.z)>abs(dir.x) && abs(dir.z)>abs(dir.y))
    {
      // along z
      shader = shader_z_render_.get();
      step_size = asml_->box_size_ / static_cast<float>(asml_->box_depth_);
      orientation = 'Z';
    }
    else
    {
      // should not have been here
      fprintf(stderr, " ERROR : axis specifying error!\n\n");
    }

    // bind fbo
    rr_fbo_->BeginDraw2FBO();
    {
      // set shader uniforms
      shader->SetUniform1f("fwidth", static_cast<float>(asml_->box_width_));
      shader->SetUniform3f("lightIntensity", asml_->light_intensity_, 
        asml_->light_intensity_, asml_->light_intensity_);
      shader->SetUniform4f("lightPosWorld", asml_->light_x_, 
        asml_->light_y_, asml_->light_z_, 1.0f);
      shader->SetUniform1f("absorptionCoefficient", asml_->extinction_);
      shader->SetUniform1f("scatteringCoefficient", asml_->scattering_);
      shader->SetUniform4f("cameraPos", 
        asml_->camera_positions_[i_view].x,
        asml_->camera_positions_[i_view].y,
        asml_->camera_positions_[i_view].z,
        asml_->camera_positions_[i_view].w);
      shader->SetUniform1f("stepSize", step_size);

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_3D, vol_tex);
      shader->SetUniform1i("volumeTex", 0);
      shader->SetUniformMatrix4fv("cameraInv", 1, GL_FALSE, inv_camera);

      if ('X' == orientation)
      {
        if (dir.x < 0.0)
        {
          for (int i = 0; i < asml_->box_width_; ++i)
          {
            float tex_x_coord = (i+0.5) / static_cast<float>(asml_->box_width_);
            float geo_x_coord = asml_->box_size_ * tex_x_coord + asml_->trans_x_;

            glBegin(GL_QUADS);
            glTexCoord3f(tex_x_coord, 0.0f, 0.0f); glVertex3f(geo_x_coord, asml_->trans_y_ - 0.5 * asml_->box_size_, asml_->trans_z_ - 0.5 * asml_->box_size_);
            glTexCoord3f(tex_x_coord, 1.0f, 0.0f); glVertex3f(geo_x_coord, asml_->trans_y_ + 0.5 * asml_->box_size_, asml_->trans_z_ - 0.5 * asml_->box_size_);
            glTexCoord3f(tex_x_coord, 1.0f, 1.0f); glVertex3f(geo_x_coord, asml_->trans_y_ + 0.5 * asml_->box_size_, asml_->trans_z_ + 0.5 * asml_->box_size_);
            glTexCoord3f(tex_x_coord, 0.0f, 1.0f); glVertex3f(geo_x_coord, asml_->trans_y_ - 0.5 * asml_->box_size_, asml_->trans_z_ + 0.5 * asml_->box_size_);
            glEnd();
          } // for i
        }
        else
        {
          for (int i = asml_->box_width_ - 1; i >= 0; --i)
          {
            float tex_x_coord = (i+0.5) / static_cast<float>(asml_->box_width_);
            float geo_x_coord = asml_->box_size_ * tex_x_coord + asml_->trans_x_;

            glBegin(GL_QUADS);
            glTexCoord3f(tex_x_coord, 0.0f, 0.0f); glVertex3f(geo_x_coord, asml_->trans_y_ - 0.5 * asml_->box_size_, asml_->trans_z_ - 0.5 * asml_->box_size_);
            glTexCoord3f(tex_x_coord, 1.0f, 0.0f); glVertex3f(geo_x_coord, asml_->trans_y_ + 0.5 * asml_->box_size_, asml_->trans_z_ - 0.5 * asml_->box_size_);
            glTexCoord3f(tex_x_coord, 1.0f, 1.0f); glVertex3f(geo_x_coord, asml_->trans_y_ + 0.5 * asml_->box_size_, asml_->trans_z_ + 0.5 * asml_->box_size_);
            glTexCoord3f(tex_x_coord, 0.0f, 1.0f); glVertex3f(geo_x_coord, asml_->trans_y_ - 0.5 * asml_->box_size_, asml_->trans_z_ + 0.5 * asml_->box_size_);
            glEnd();
          } // for i
        }
      }
      else if ('Y' == orientation)
      {
        if (dir.y < 0.0)
        {
          for (int j = 0; j < asml_->box_height_; ++j)
          {
            float tex_y_coord = (j+0.5) / static_cast<float>(asml_->box_height_);
            float geo_y_coord = tex_y_coord * asml_->box_size_ + asml_->trans_y_;

            glBegin(GL_QUADS);
            glTexCoord3f(0.0f, tex_y_coord, 0.0f); glVertex3f(asml_->trans_x_ - 0.5*asml_->box_size_, geo_y_coord, asml_->trans_z_ - 0.5 * asml_->box_size_);
            glTexCoord3f(1.0f, tex_y_coord, 0.0f); glVertex3f(asml_->trans_x_ + 0.5*asml_->box_size_, geo_y_coord, asml_->trans_z_ - 0.5 * asml_->box_size_);
            glTexCoord3f(1.0f, tex_y_coord, 1.0f); glVertex3f(asml_->trans_x_ + 0.5*asml_->box_size_, geo_y_coord, asml_->trans_z_ + 0.5 * asml_->box_size_);
            glTexCoord3f(0.0f, tex_y_coord, 1.0f); glVertex3f(asml_->trans_x_ - 0.5*asml_->box_size_, geo_y_coord, asml_->trans_z_ + 0.5 * asml_->box_size_);
            glEnd();
          } // for j
        }
        else
        {
          for (int j = asml_->box_height_-1; j >= 0; --j)
          {
            float tex_y_coord = (j+0.5) / static_cast<float>(asml_->box_height_);
            float geo_y_coord = tex_y_coord * asml_->box_size_ + asml_->trans_y_;

            glBegin(GL_QUADS);
            glTexCoord3f(0.0f, tex_y_coord, 0.0f); glVertex3f(asml_->trans_x_ - 0.5*asml_->box_size_, geo_y_coord, asml_->trans_z_ - 0.5 * asml_->box_size_);
            glTexCoord3f(1.0f, tex_y_coord, 0.0f); glVertex3f(asml_->trans_x_ + 0.5*asml_->box_size_, geo_y_coord, asml_->trans_z_ - 0.5 * asml_->box_size_);
            glTexCoord3f(1.0f, tex_y_coord, 1.0f); glVertex3f(asml_->trans_x_ + 0.5*asml_->box_size_, geo_y_coord, asml_->trans_z_ + 0.5 * asml_->box_size_);
            glTexCoord3f(0.0f, tex_y_coord, 1.0f); glVertex3f(asml_->trans_x_ - 0.5*asml_->box_size_, geo_y_coord, asml_->trans_z_ + 0.5 * asml_->box_size_);
            glEnd();
          } // for j
        }

      }
      else if ('Z' == orientation)
      {
        if (dir.z < 0.0)
        {
          for (int k = 0; k < asml_->box_depth_; ++k)
          {
            float tex_z_coord = (k+0.5) / static_cast<float>(asml_->box_depth_);
            float geo_z_coord = tex_z_coord * asml_->box_size_ + asml_->trans_z_;

            glBegin(GL_QUADS);
            glTexCoord3f(0.0f, 0.0f, tex_z_coord); glVertex3f(asml_->trans_x_ - 0.5*asml_->box_size_, asml_->trans_y_ - 0.5*asml_->box_size_, geo_z_coord);
            glTexCoord3f(1.0f, 0.0f, tex_z_coord); glVertex3f(asml_->trans_x_ + 0.5*asml_->box_size_, asml_->trans_y_ - 0.5*asml_->box_size_, geo_z_coord);
            glTexCoord3f(1.0f, 1.0f, tex_z_coord); glVertex3f(asml_->trans_x_ + 0.5*asml_->box_size_, asml_->trans_y_ + 0.5*asml_->box_size_, geo_z_coord);
            glTexCoord3f(0.0f, 1.0f, tex_z_coord); glVertex3f(asml_->trans_x_ - 0.5*asml_->box_size_, asml_->trans_y_ + 0.5*asml_->box_size_, geo_z_coord);
            glEnd();
          }
        }
        else
        {
          for (int k = asml_->box_depth_ - 1; k >= 0; --k)
          {
            float tex_z_coord = (k+0.5) / static_cast<float>(asml_->box_depth_);
            float geo_z_coord = tex_z_coord * asml_->box_size_ + asml_->trans_z_;

            glBegin(GL_QUADS);
            glTexCoord3f(0.0f, 0.0f, tex_z_coord); glVertex3f(asml_->trans_x_ - 0.5*asml_->box_size_, asml_->trans_y_ - 0.5*asml_->box_size_, geo_z_coord);
            glTexCoord3f(1.0f, 0.0f, tex_z_coord); glVertex3f(asml_->trans_x_ + 0.5*asml_->box_size_, asml_->trans_y_ - 0.5*asml_->box_size_, geo_z_coord);
            glTexCoord3f(1.0f, 1.0f, tex_z_coord); glVertex3f(asml_->trans_x_ + 0.5*asml_->box_size_, asml_->trans_y_ + 0.5*asml_->box_size_, geo_z_coord);
            glTexCoord3f(0.0f, 1.0f, tex_z_coord); glVertex3f(asml_->trans_x_ - 0.5*asml_->box_size_, asml_->trans_y_ + 0.5*asml_->box_size_, geo_z_coord);
            glEnd();
          }
        }
      }
      else
      {
        // should not be here
      }


    }
    rr_fbo_->EndDraw2FBO();
  }

  void RenderGL::render_perturbed(int i_view, GLuint vol_tex, int slice, int pu, int pv)
  {
    float proj_mat[16];
    float mv_mat[16];

    // get MODELVIEW and PROJECTION matrices
    asml_->gl_projection_mats_[i_view].GetData(proj_mat);
    asml_->camera_gl_extr_paras_[i_view].GetData(mv_mat);

    glViewport(0, 0, width_, height_);
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(proj_mat);
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(mv_mat);

    //// adjust the volume
    //glTranslatef(asml_->trans_x_, asml_->trans_y_, asml_->trans_z_);

    // get the inversed camera matrix
    float inv_camera[16];
    asml_->camera_inv_gl_extr_paras_[i_view].GetData(inv_camera);

    // choose shader
    Vector4 dir(
      asml_->camera_positions_[i_view].x - asml_->trans_x_,
      asml_->camera_positions_[i_view].y - asml_->trans_y_,
      asml_->camera_positions_[i_view].z - asml_->trans_z_);
    dir.normaVec();

    GLSLShader * shader = NULL;
    float step_size = 0.0f;
    char orientation = 'N';
    if (abs(dir.x)>abs(dir.y) && abs(dir.x)>abs(dir.z))
    {
      // along x
      shader = shader_x_pertuerbed_.get();
      step_size = asml_->box_size_ / static_cast<float>(asml_->box_width_);
      orientation = 'X';
    }
    else if (abs(dir.y)>abs(dir.x) && abs(dir.y)>abs(dir.z))
    {
      // along y
      shader = shader_y_pertuerbed_.get();
      step_size = asml_->box_size_ / static_cast<float>(asml_->box_height_);
      orientation = 'Y';
    }
    else if (abs(dir.z)>abs(dir.x) && abs(dir.z)>abs(dir.y))
    {
      // along z
      shader = shader_z_pertuerbed_.get();
      step_size = asml_->box_size_ / static_cast<float>(asml_->box_depth_);
      orientation = 'Z';
    }
    else
    {
      // should not have been here
      fprintf(stderr, " ERROR : axis specifying error!\n\n");
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////


    // bind fbo
    pr_fbo_->BeginDraw2FBO();
    {
      // set shader uniforms
      //shader->SetUniform4i("disturbPara", asml_->volume_interval_, pu, pv, 0);
      shader->SetUniform1f("disturb", asml_->disturb_);
      shader->SetUniform1f("fwidth", static_cast<float>(asml_->box_width_));
      shader->SetUniform1f("disturb", asml_->disturb_);
      shader->SetUniform3f("lightIntensity", asml_->light_intensity_, 
        asml_->light_intensity_, asml_->light_intensity_);
      shader->SetUniform4f("lightPosWorld", asml_->light_x_, 
        asml_->light_y_, asml_->light_z_, 1.0f);
      shader->SetUniform1f("absorptionCoefficient", asml_->extinction_);
      shader->SetUniform1f("scatteringCoefficient", asml_->scattering_);
      shader->SetUniform4f("cameraPos", 
        asml_->camera_positions_[i_view].x,
        asml_->camera_positions_[i_view].y,
        asml_->camera_positions_[i_view].z,
        asml_->camera_positions_[i_view].w);
      shader->SetUniform1f("stepSize", step_size);

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_3D, vol_tex);
      shader->SetUniform1i("volumeTex", 0);

      shader->SetUniformMatrix4fv("cameraInv", 1, GL_FALSE, inv_camera);

      if ('X' == orientation)
      {
        float y_min = - asml_->box_size_ / 2.0;
        float y_max =   asml_->box_size_ / 2.0;
        float z_min = - asml_->box_size_ / 2.0;
        float z_max =   asml_->box_size_ / 2.0;

        if (dir.x < 0.0)
        {
          for (int i = 0; i < asml_->box_width_; ++i)
          {

          } // for i
        }
        else
        {
          for (int i = asml_->box_width_-1; i >= 0; --i)
          {
          } // for i
        }
      }
      else if ('Y' == orientation)
      {
        float x_min =  - asml_->box_size_ / 2.0;
        float x_max =    asml_->box_size_ / 2.0;
        float z_min =  - asml_->box_size_ / 2.0;
        float z_max =    asml_->box_size_ / 2.0;

      }
      else if ('Z' == orientation)
      {
        float x_min =  - asml_->box_size_ / 2.0;
        float x_max =    asml_->box_size_ / 2.0;
        float y_min =  - asml_->box_size_ / 2.0;
        float y_max =    asml_->box_size_ / 2.0;
      }
      else
      {
        // should not be here
      }

    }
    pr_fbo_->EndDraw2FBO();
  }

  RenderGL::RenderGL(ASModeling *p)
    :asml_(p) {};

} // namespace as_modeling