#include "RenderGL.h"
#include "ASModeling.h"
#include <GL/glut.h>

#include "../Utils/image/PFMImage.h"


//#define __TEST_RENDER__

namespace as_modeling
{
  bool RenderGL::init()
  {
    //////////////
    counter = 0;

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
    glBlendFuncSeparate(GL_DST_ALPHA, GL_ONE, GL_DST_ALPHA, GL_ZERO);

    return true;
  }

  bool RenderGL::release( )
  {
    // reallocate the resource
    return true;
  }

  void RenderGL::render_unperturbed(int i_view, GLuint vol_tex, int length)
  {
    //fprintf(stderr, "--- Render unperturbed : view %d \n", i_view);

    float proj_mat[16];
    float mv_mat[16];

    float half_size =  0.5 * asml_->box_size_;

    // get MODELVIEW and PROJECTION matrices
    asml_->gl_projection_mats_[i_view].GetData(proj_mat);
    asml_->camera_gl_extr_paras_[i_view].GetData(mv_mat);

    glViewport(0, 0, width_, height_);
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(proj_mat);
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(mv_mat);

    // model tranformation
    // adjust the volume
    //glTranslatef(asml_->trans_x_, asml_->trans_y_, asml_->trans_z_);

    // get the inversed camera matrix
    float inv_camera[16];
    asml_->camera_inv_gl_extr_paras_[i_view].GetData(inv_camera);

    // choose shader

    GLSLShader * shader = NULL;
    float step_size = asml_->box_size_ / static_cast<float>(length);

    if (asml_->camera_orientations_[i_view] == 'x' || asml_->camera_orientations_[i_view] == 'X')
    {
      // along x
      shader = shader_x_render_.get();
    }
    else if (asml_->camera_orientations_[i_view] == 'y' || asml_->camera_orientations_[i_view] == 'Y')
    {
      // along y
      shader = shader_y_render_.get();
    }
    else //if (asml_->camera_orientations_[i_view] == 'z' || asml_->camera_orientations_[i_view] == 'Z')
    {
      // along z
      shader = shader_z_render_.get();
    }

#ifdef __TEST_RENDER__
    rr_fbo_->BeginDraw2FBO();
    {
      //glBegin(GL_TRIANGLES);
      //glColor3f(1.0, 0.0, 0.0);
      //glVertex3f(0.0, 0.0, 0.0);
      //glColor3f(0.0, 1.0, 0.0);
      //glVertex3f(1.0, 1.0, 0.0);
      //glColor3f(0.0, 0.0, 1.0);
      //glVertex3f(1.0, 0.0, 0.0);
      //glEnd();

      int length = 32;
      glBegin(GL_POINTS);
      for (int ii=-1; ii<2; ii+=2)
      {
        for (int jj=-1; jj<2; jj+=2)
        {
          for (int kk=-1; kk<2; kk+=2)
          {
            glColor3f(1.0f*(ii+1)/2.0, 1.0f*(jj+1)/2.0f, 1.0f*(kk+1)/2.0f);
            glVertex3f(
              asml_->trans_x_ + ii*1.0 * half_size,
              asml_->trans_y_ + jj*1.0 * half_size,
              asml_->trans_z_ + kk*1.0 * half_size );
          }
        }
      }
      glEnd();

    }
    rr_fbo_->EndDraw2FBO();
#endif //__TEST_RENDER__

#ifndef __TEST_RENDER__
    // bind fbo
    rr_fbo_->BeginDraw2FBO();
    {
      glClearColor(0.0, 0.0, 0.0, 1.0);
      glDisable(GL_DEPTH_TEST);
      glEnable(GL_BLEND);
      glBlendFuncSeparate(GL_DST_ALPHA, GL_ONE, GL_DST_ALPHA, GL_ZERO);

      glClear(GL_COLOR_BUFFER_BIT);

      shader->Begin();

      // set shader uniforms
      //shader->SetUniform3f("boxTrans", asml_->trans_x_, asml_->trans_y_, asml_->trans_z_);
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


#if 0
      float tex_x_coord = (length/2+0.5) / static_cast<float>(length);
      float geo_x_coord = asml_->box_size_ * tex_x_coord + asml_->trans_x_ - half_size;

      glBegin(GL_QUADS);
      glTexCoord3f(tex_x_coord, 0.0f, 0.0f); glVertex3f(geo_x_coord, asml_->trans_y_ - half_size, asml_->trans_z_ - half_size);
      glTexCoord3f(tex_x_coord, 1.0f, 0.0f); glVertex3f(geo_x_coord, asml_->trans_y_ + half_size, asml_->trans_z_ - half_size);
      glTexCoord3f(tex_x_coord, 1.0f, 1.0f); glVertex3f(geo_x_coord, asml_->trans_y_ + half_size, asml_->trans_z_ + half_size);
      glTexCoord3f(tex_x_coord, 0.0f, 1.0f); glVertex3f(geo_x_coord, asml_->trans_y_ - half_size, asml_->trans_z_ + half_size);
      glEnd();
#endif

      if ('X' == asml_->camera_orientations_[i_view])
      {
        for (int i = 0; i < length; ++i)
        //int i = 31;
        {
          float tex_x_coord = (i+0.5) / static_cast<float>(length);
          float geo_x_coord = asml_->box_size_ * tex_x_coord + asml_->trans_x_ - half_size;

          glBegin(GL_QUADS);
          glTexCoord3f(tex_x_coord, 0.0f, 0.0f); glVertex3f(geo_x_coord, asml_->trans_y_ - half_size, asml_->trans_z_ - half_size);
          glTexCoord3f(tex_x_coord, 1.0f, 0.0f); glVertex3f(geo_x_coord, asml_->trans_y_ + half_size, asml_->trans_z_ - half_size);
          glTexCoord3f(tex_x_coord, 1.0f, 1.0f); glVertex3f(geo_x_coord, asml_->trans_y_ + half_size, asml_->trans_z_ + half_size);
          glTexCoord3f(tex_x_coord, 0.0f, 1.0f); glVertex3f(geo_x_coord, asml_->trans_y_ - half_size, asml_->trans_z_ + half_size);
          glEnd();
        } // for i
      }
      else if ('x' == asml_->camera_orientations_[i_view])
      {
        for (int i = length - 1; i >= 0; --i)
        //int i = 31;
        {
          float tex_x_coord = (i+0.5) / static_cast<float>(length);
          float geo_x_coord = asml_->box_size_ * tex_x_coord + asml_->trans_x_ - half_size;

          glBegin(GL_QUADS);
          glTexCoord3f(tex_x_coord, 0.0f, 0.0f); glVertex3f(geo_x_coord, asml_->trans_y_ - half_size, asml_->trans_z_ - half_size);
          glTexCoord3f(tex_x_coord, 1.0f, 0.0f); glVertex3f(geo_x_coord, asml_->trans_y_ + half_size, asml_->trans_z_ - half_size);
          glTexCoord3f(tex_x_coord, 1.0f, 1.0f); glVertex3f(geo_x_coord, asml_->trans_y_ + half_size, asml_->trans_z_ + half_size);
          glTexCoord3f(tex_x_coord, 0.0f, 1.0f); glVertex3f(geo_x_coord, asml_->trans_y_ - half_size, asml_->trans_z_ + half_size);
          glEnd();
        } // for i
      }
      else if ('Y' == asml_->camera_orientations_[i_view])
      {
        for (int j = 0; j < length; ++j)
        {
          float tex_y_coord = (j+0.5) / static_cast<float>(length);
          float geo_y_coord = tex_y_coord * asml_->box_size_ + asml_->trans_y_ - half_size;

          glBegin(GL_QUADS);
          glTexCoord3f(0.0f, tex_y_coord, 0.0f); glVertex3f(asml_->trans_x_ - half_size, geo_y_coord, asml_->trans_z_ - half_size);
          glTexCoord3f(1.0f, tex_y_coord, 0.0f); glVertex3f(asml_->trans_x_ + half_size, geo_y_coord, asml_->trans_z_ - half_size);
          glTexCoord3f(1.0f, tex_y_coord, 1.0f); glVertex3f(asml_->trans_x_ + half_size, geo_y_coord, asml_->trans_z_ + half_size);
          glTexCoord3f(0.0f, tex_y_coord, 1.0f); glVertex3f(asml_->trans_x_ - half_size, geo_y_coord, asml_->trans_z_ + half_size);
          glEnd();
        } // for j
      }
      else if ('y' == asml_->camera_orientations_[i_view])
      {
        for (int j = length-1; j >= 0; --j)
        {
          float tex_y_coord = (j+0.5) / static_cast<float>(length);
          float geo_y_coord = tex_y_coord * asml_->box_size_ + asml_->trans_y_ - half_size;

          glBegin(GL_QUADS);
          glTexCoord3f(0.0f, tex_y_coord, 0.0f); glVertex3f(asml_->trans_x_ - half_size, geo_y_coord, asml_->trans_z_ - half_size);
          glTexCoord3f(1.0f, tex_y_coord, 0.0f); glVertex3f(asml_->trans_x_ + half_size, geo_y_coord, asml_->trans_z_ - half_size);
          glTexCoord3f(1.0f, tex_y_coord, 1.0f); glVertex3f(asml_->trans_x_ + half_size, geo_y_coord, asml_->trans_z_ + half_size);
          glTexCoord3f(0.0f, tex_y_coord, 1.0f); glVertex3f(asml_->trans_x_ - half_size, geo_y_coord, asml_->trans_z_ + half_size);
          glEnd();
        } // for j
      }
      else if ('Z' == asml_->camera_orientations_[i_view])
      {
        for (int k = 0; k < length; ++k)
        {
          float tex_z_coord = (k+0.5) / static_cast<float>(length);
          float geo_z_coord = tex_z_coord * asml_->box_size_ + asml_->trans_z_ - half_size;

          glBegin(GL_QUADS);
          glTexCoord3f(0.0f, 0.0f, tex_z_coord); glVertex3f(asml_->trans_x_ - half_size, asml_->trans_y_ - half_size, geo_z_coord);
          glTexCoord3f(1.0f, 0.0f, tex_z_coord); glVertex3f(asml_->trans_x_ + half_size, asml_->trans_y_ - half_size, geo_z_coord);
          glTexCoord3f(1.0f, 1.0f, tex_z_coord); glVertex3f(asml_->trans_x_ + half_size, asml_->trans_y_ + half_size, geo_z_coord);
          glTexCoord3f(0.0f, 1.0f, tex_z_coord); glVertex3f(asml_->trans_x_ - half_size, asml_->trans_y_ + half_size, geo_z_coord);
          glEnd();
        }
      }
      else if ('z' == asml_->camera_orientations_[i_view])
      {
        for (int k = length - 1; k >= 0; --k)
        {
          float tex_z_coord = (k+0.5) / static_cast<float>(length);
          float geo_z_coord = tex_z_coord * asml_->box_size_ + asml_->trans_z_ - half_size;

          glBegin(GL_QUADS);
          glTexCoord3f(0.0f, 0.0f, tex_z_coord); glVertex3f(asml_->trans_x_ - half_size, asml_->trans_y_ - half_size, geo_z_coord);
          glTexCoord3f(1.0f, 0.0f, tex_z_coord); glVertex3f(asml_->trans_x_ + half_size, asml_->trans_y_ - half_size, geo_z_coord);
          glTexCoord3f(1.0f, 1.0f, tex_z_coord); glVertex3f(asml_->trans_x_ + half_size, asml_->trans_y_ + half_size, geo_z_coord);
          glTexCoord3f(0.0f, 1.0f, tex_z_coord); glVertex3f(asml_->trans_x_ - half_size, asml_->trans_y_ + half_size, geo_z_coord);
          glEnd();
        }
      }

      glFinish();

      shader->End();
    }
    rr_fbo_->EndDraw2FBO();
#endif //__TEST_RENDER__


#if 1
    float * data = rr_fbo_->ReadPixels();
    float * img = new float [3 * width_ * height_];
    for (int y = 0; y < height_; ++y)
    {
      for (int x = 0; x < width_; ++x)
      {
        for (int c = 0; c < 3; ++c)
        {
          img[y * width_ * 3 + x * 3 + c] = data[y * width_ * 4 + x * 4 + c];
        }
      }
    }
  /*  cuda_imageutil::BMPImageUtil tmpBmp;
    tmpBmp.SetSizes(width_, height_);
    for (int y = 0; y < height_; ++y)
    {
      for (int x = 0; x < width_; ++x)
      {
        tmpBmp.GetPixelAt(x,y)[0] = static_cast<unsigned char> (
          254.0f * data[((height_-1-y)*width_+x)*4] );
        tmpBmp.GetPixelAt(x,y)[1] = static_cast<unsigned char> (
          254.0f * data[((height_-1-y)*width_+x)*4 + 1]);
        tmpBmp.GetPixelAt(x,y)[2] = static_cast<unsigned char> (
          254.0f * data[((height_-1-y)*width_+x)*4 + 2]);
      }
    }*/
    char path_buf[100];
    sprintf(path_buf, "../Data/Camera%02d/show%06d.pfm", i_view, counter);
    PFMImage * tmpimg = new PFMImage(width_, height_, 1, img);
    tmpimg->WriteImage(path_buf);
    delete [] img;
    delete tmpimg;

    //sprintf(path_buf, "../Data/Camera%02d/show%06d.bmp", i_view, counter);
    //tmpBmp.SaveImage(path_buf);
    ++counter;
#endif

  }

  void RenderGL::render_perturbed(int i_view, GLuint vol_tex, int length, int slice, int pu, int pv)
  {
    float proj_mat[16];
    float mv_mat[16];

    float half_size = 0.5 * asml_->box_size_;

    //fprintf(stderr, "--- Render perturbed : view %d \n", i_view);

    // get MODELVIEW and PROJECTION matrices
    asml_->gl_projection_mats_[i_view].GetData(proj_mat);
    asml_->camera_gl_extr_paras_[i_view].GetData(mv_mat);

    glViewport(0, 0, width_, height_);
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(proj_mat);
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(mv_mat);

    //// adjust the volume
    glTranslatef(asml_->trans_x_, asml_->trans_y_, asml_->trans_z_);

    // get the inversed camera matrix
    float inv_camera[16];
    asml_->camera_inv_gl_extr_paras_[i_view].GetData(inv_camera);

    // choose shader

    GLSLShader * shader = NULL;
    float step_size = asml_->box_size_ / static_cast<float>(length);

    if (asml_->camera_orientations_[i_view] == 'x' || asml_->camera_orientations_[i_view] == 'X')
    {
      // along x
      shader = shader_x_pertuerbed_.get();
    }
    else if (asml_->camera_orientations_[i_view] == 'y' || asml_->camera_orientations_[i_view] == 'Y')
    {
      // along y
      shader = shader_y_pertuerbed_.get();
    }
    else //if (asml_->camera_orientations_[i_view] == 'z' || asml_->camera_orientations_[i_view] == 'Z')
    {
      // along z
      shader = shader_z_pertuerbed_.get();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __TEST_RENDER__
    pr_fbo_->BeginDraw2FBO();
    {
      glBegin(GL_POINTS);
      glColor3f(1.0, 1.0, 1.0);
      glVertex3f(0.0, 0.0, 0.0);
      glEnd();
    }
    pr_fbo_->EndDraw2FBO();
#endif //__TEST_RENDER__

#ifndef __TEST_RENDER__
    // bind fbo
    pr_fbo_->BeginDraw2FBO();
    {
      glClearColor(0.0, 0.0, 0.0, 1.0);
      glDisable(GL_DEPTH_TEST);
      glEnable(GL_BLEND);
      glBlendFuncSeparate(GL_DST_ALPHA, GL_ONE, GL_DST_ALPHA, GL_ZERO);

      glClear(GL_COLOR_BUFFER_BIT);

      shader->Begin();

      // set shader uniforms
      //shader->SetUniform3f("boxTrans", asml_->trans_x_, asml_->trans_y_, asml_->trans_z_);
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

      if ('X' == asml_->camera_orientations_[i_view])
      {
        for (int i = 0; i < length; ++i)
        //int i =1;
        {
          float tex_x_coord = (i+0.5) / static_cast<float>(length);
          float geo_x_coord = asml_->box_size_ * tex_x_coord + asml_->trans_x_ - half_size;

          shader->SetUniform4i("disturbPara", asml_->volume_interval_, pu, pv, (slice==i)?1:0);

          glBegin(GL_QUADS);
          glTexCoord3f(tex_x_coord, 0.0f, 0.0f); glVertex3f(geo_x_coord, asml_->trans_y_ - half_size, asml_->trans_z_ - half_size);
          glTexCoord3f(tex_x_coord, 1.0f, 0.0f); glVertex3f(geo_x_coord, asml_->trans_y_ + half_size, asml_->trans_z_ - half_size);
          glTexCoord3f(tex_x_coord, 1.0f, 1.0f); glVertex3f(geo_x_coord, asml_->trans_y_ + half_size, asml_->trans_z_ + half_size);
          glTexCoord3f(tex_x_coord, 0.0f, 1.0f); glVertex3f(geo_x_coord, asml_->trans_y_ - half_size, asml_->trans_z_ + half_size);
          glEnd();
        } // for i
      }
      else if ('x' == asml_->camera_orientations_[i_view])
      {
        for (int i = length - 1; i >= 0; --i)
        //int i = 1;
        {
          float tex_x_coord = (i+0.5) / static_cast<float>(length);
          float geo_x_coord = asml_->box_size_ * tex_x_coord + asml_->trans_x_ - half_size;

          shader->SetUniform4i("disturbPara", asml_->volume_interval_, pu, pv, (slice==i)?1:0);

          glBegin(GL_QUADS);
          glTexCoord3f(tex_x_coord, 0.0f, 0.0f); glVertex3f(geo_x_coord, asml_->trans_y_ - half_size, asml_->trans_z_ - half_size);
          glTexCoord3f(tex_x_coord, 1.0f, 0.0f); glVertex3f(geo_x_coord, asml_->trans_y_ + half_size, asml_->trans_z_ - half_size);
          glTexCoord3f(tex_x_coord, 1.0f, 1.0f); glVertex3f(geo_x_coord, asml_->trans_y_ + half_size, asml_->trans_z_ + half_size);
          glTexCoord3f(tex_x_coord, 0.0f, 1.0f); glVertex3f(geo_x_coord, asml_->trans_y_ - half_size, asml_->trans_z_ + half_size);
          glEnd();
        } // for i
      }
      else if ('Y' == asml_->camera_orientations_[i_view])
      {
        for (int j = 0; j < length; ++j)
        {
          float tex_y_coord = (j+0.5) / static_cast<float>(length);
          float geo_y_coord = tex_y_coord * asml_->box_size_ + asml_->trans_y_ - half_size;

          shader->SetUniform4i("disturbPara", asml_->volume_interval_, pu, pv, (slice==j)?1:0);

          glBegin(GL_QUADS);
          glTexCoord3f(0.0f, tex_y_coord, 0.0f); glVertex3f(asml_->trans_x_ - half_size, geo_y_coord, asml_->trans_z_ - half_size);
          glTexCoord3f(1.0f, tex_y_coord, 0.0f); glVertex3f(asml_->trans_x_ + half_size, geo_y_coord, asml_->trans_z_ - half_size);
          glTexCoord3f(1.0f, tex_y_coord, 1.0f); glVertex3f(asml_->trans_x_ + half_size, geo_y_coord, asml_->trans_z_ + half_size);
          glTexCoord3f(0.0f, tex_y_coord, 1.0f); glVertex3f(asml_->trans_x_ - half_size, geo_y_coord, asml_->trans_z_ + half_size);
          glEnd();
        } // for j
      }
      else if ('y' == asml_->camera_orientations_[i_view])
      {
        for (int j = length-1; j >= 0; --j)
        {
          float tex_y_coord = (j+0.5) / static_cast<float>(length);
          float geo_y_coord = tex_y_coord * asml_->box_size_ + asml_->trans_y_ - half_size;

          shader->SetUniform4i("disturbPara", asml_->volume_interval_, pu, pv, (slice==j)?1:0);

          glBegin(GL_QUADS);
          glTexCoord3f(0.0f, tex_y_coord, 0.0f); glVertex3f(asml_->trans_x_ - half_size, geo_y_coord, asml_->trans_z_ - half_size);
          glTexCoord3f(1.0f, tex_y_coord, 0.0f); glVertex3f(asml_->trans_x_ + half_size, geo_y_coord, asml_->trans_z_ - half_size);
          glTexCoord3f(1.0f, tex_y_coord, 1.0f); glVertex3f(asml_->trans_x_ + half_size, geo_y_coord, asml_->trans_z_ + half_size);
          glTexCoord3f(0.0f, tex_y_coord, 1.0f); glVertex3f(asml_->trans_x_ - half_size, geo_y_coord, asml_->trans_z_ + half_size);
          glEnd();
        } // for j
      }
      else if ('Z' == asml_->camera_orientations_[i_view])
      {
        for (int k = 0; k < length; ++k)
        {
          float tex_z_coord = (k+0.5) / static_cast<float>(length);
          float geo_z_coord = tex_z_coord * asml_->box_size_ + asml_->trans_z_ - half_size;

          shader->SetUniform4i("disturbPara", asml_->volume_interval_, pu, pv, (slice==k)?1:0);

          glBegin(GL_QUADS);
          glTexCoord3f(0.0f, 0.0f, tex_z_coord); glVertex3f(asml_->trans_x_ - half_size, asml_->trans_y_ - half_size, geo_z_coord);
          glTexCoord3f(1.0f, 0.0f, tex_z_coord); glVertex3f(asml_->trans_x_ + half_size, asml_->trans_y_ - half_size, geo_z_coord);
          glTexCoord3f(1.0f, 1.0f, tex_z_coord); glVertex3f(asml_->trans_x_ + half_size, asml_->trans_y_ + half_size, geo_z_coord);
          glTexCoord3f(0.0f, 1.0f, tex_z_coord); glVertex3f(asml_->trans_x_ - half_size, asml_->trans_y_ + half_size, geo_z_coord);
          glEnd();
        }
      }
      else if ('z' == asml_->camera_orientations_[i_view])
      {
        for (int k = length - 1; k >= 0; --k)
        {
          float tex_z_coord = (k+0.5) / static_cast<float>(length);
          float geo_z_coord = tex_z_coord * asml_->box_size_ + asml_->trans_z_ - half_size;

          shader->SetUniform4i("disturbPara", asml_->volume_interval_, pu, pv, (slice==k)?1:0);

          glBegin(GL_QUADS);
          glTexCoord3f(0.0f, 0.0f, tex_z_coord); glVertex3f(asml_->trans_x_ - half_size, asml_->trans_y_ - half_size, geo_z_coord);
          glTexCoord3f(1.0f, 0.0f, tex_z_coord); glVertex3f(asml_->trans_x_ + half_size, asml_->trans_y_ - half_size, geo_z_coord);
          glTexCoord3f(1.0f, 1.0f, tex_z_coord); glVertex3f(asml_->trans_x_ + half_size, asml_->trans_y_ + half_size, geo_z_coord);
          glTexCoord3f(0.0f, 1.0f, tex_z_coord); glVertex3f(asml_->trans_x_ - half_size, asml_->trans_y_ + half_size, geo_z_coord);
          glEnd();
        }
      }

      shader->End();
    }
    pr_fbo_->EndDraw2FBO();
#endif //__TEST_RENDER__

  }

  RenderGL::RenderGL(ASModeling *p)
    :asml_(p) {};

} // namespace as_modeling