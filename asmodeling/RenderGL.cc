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
    glDeleteBuffers(6, vbo_ids_);
    return true;
  }


  bool RenderGL::level_init(int level)
  {
    int length = 1 << level;
    float half_size =  0.5 * asml_->box_size_;

    int size_vertices = 3 * 4 * length;
    int size_texcoord = 3 * 4 * length;
    float *vertices_ptr = new float[size_vertices];
    float *texcoord_ptr = new float[size_texcoord];
    int ind_vert = 0;
    int ind_txcd = 0;

    if (level > asml_->initial_vol_level_)
    {
      glDeleteBuffers(6, vbo_ids_);
    }
    glGenBuffers(6, vbo_ids_);

    // X
    ind_vert = 0;
    ind_txcd = 0;
    for (int i = 0; i < length; ++i)
    {
      float tex_x_coord = (i+0.5) / static_cast<float>(length);
      float geo_x_coord = asml_->box_size_ * tex_x_coord + asml_->trans_x_ - half_size;

      // corner 1
      vertices_ptr[ind_vert++] = geo_x_coord;
      vertices_ptr[ind_vert++] = asml_->trans_y_ - half_size;
      vertices_ptr[ind_vert++] = asml_->trans_z_ - half_size;

      texcoord_ptr[ind_txcd++] = tex_x_coord;
      texcoord_ptr[ind_txcd++] = 0.0f;
      texcoord_ptr[ind_txcd++] = 0.0f;

      // corner 2
      vertices_ptr[ind_vert++] = geo_x_coord;
      vertices_ptr[ind_vert++] = asml_->trans_y_ + half_size;
      vertices_ptr[ind_vert++] = asml_->trans_z_ - half_size;

      texcoord_ptr[ind_txcd++] = tex_x_coord;
      texcoord_ptr[ind_txcd++] = 1.0f;
      texcoord_ptr[ind_txcd++] = 0.0f;

      // corner 3
      vertices_ptr[ind_vert++] = geo_x_coord;
      vertices_ptr[ind_vert++] = asml_->trans_y_ + half_size;
      vertices_ptr[ind_vert++] = asml_->trans_z_ + half_size;

      texcoord_ptr[ind_txcd++] = tex_x_coord;
      texcoord_ptr[ind_txcd++] = 1.0f;
      texcoord_ptr[ind_txcd++] = 1.0f;

      // corner 4
      vertices_ptr[ind_vert++] = geo_x_coord;
      vertices_ptr[ind_vert++] = asml_->trans_y_ - half_size;
      vertices_ptr[ind_vert++] = asml_->trans_z_ + half_size;

      texcoord_ptr[ind_txcd++] = tex_x_coord;
      texcoord_ptr[ind_txcd++] = 0.0f;
      texcoord_ptr[ind_txcd++] = 1.0f;
    } // for i

    glBindBuffer(GL_ARRAY_BUFFER, vbo_ids_[0]);
    glBufferData(GL_ARRAY_BUFFER, size_vertices + size_texcoord, 0, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size_vertices, vertices_ptr);
    glBufferSubData(GL_ARRAY_BUFFER, size_vertices, size_texcoord, texcoord_ptr);

    // x
    ind_vert = 0;
    ind_txcd = 0;
    for (int i = length - 1; i >= 0; --i)
    {
      float tex_x_coord = (i+0.5) / static_cast<float>(length);
      float geo_x_coord = asml_->box_size_ * tex_x_coord + asml_->trans_x_ - half_size;

      // corner 1
      vertices_ptr[ind_vert++] = geo_x_coord;
      vertices_ptr[ind_vert++] = asml_->trans_y_ - half_size;
      vertices_ptr[ind_vert++] = asml_->trans_z_ - half_size;

      texcoord_ptr[ind_txcd++] = tex_x_coord;
      texcoord_ptr[ind_txcd++] = 0.0f;
      texcoord_ptr[ind_txcd++] = 0.0f;

      // corner 2
      vertices_ptr[ind_vert++] = geo_x_coord;
      vertices_ptr[ind_vert++] = asml_->trans_y_ + half_size;
      vertices_ptr[ind_vert++] = asml_->trans_z_ - half_size;

      texcoord_ptr[ind_txcd++] = tex_x_coord;
      texcoord_ptr[ind_txcd++] = 1.0f;
      texcoord_ptr[ind_txcd++] = 0.0f;

      // corner 3
      vertices_ptr[ind_vert++] = geo_x_coord;
      vertices_ptr[ind_vert++] = asml_->trans_y_ + half_size;
      vertices_ptr[ind_vert++] = asml_->trans_z_ + half_size;

      texcoord_ptr[ind_txcd++] = tex_x_coord;
      texcoord_ptr[ind_txcd++] = 1.0f;
      texcoord_ptr[ind_txcd++] = 1.0f;

      // corner 4
      vertices_ptr[ind_vert++] = geo_x_coord;
      vertices_ptr[ind_vert++] = asml_->trans_y_ - half_size;
      vertices_ptr[ind_vert++] = asml_->trans_z_ + half_size;

      texcoord_ptr[ind_txcd++] = tex_x_coord;
      texcoord_ptr[ind_txcd++] = 0.0f;
      texcoord_ptr[ind_txcd++] = 1.0f;
    } // for i

    glBindBuffer(GL_ARRAY_BUFFER, vbo_ids_[1]);
    glBufferData(GL_ARRAY_BUFFER, size_vertices + size_texcoord, 0, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size_vertices, vertices_ptr);
    glBufferSubData(GL_ARRAY_BUFFER, size_vertices, size_texcoord, texcoord_ptr);

    // Y
    ind_vert = 0;
    ind_txcd = 0;
    for (int j = 0; j < length; ++j)
    {
      float tex_y_coord = (j+0.5) / static_cast<float>(length);
      float geo_y_coord = tex_y_coord * asml_->box_size_ + asml_->trans_y_ - half_size;

      // corner 1
      vertices_ptr[ind_vert++] = asml_->trans_x_ - half_size;
      vertices_ptr[ind_vert++] = geo_y_coord;
      vertices_ptr[ind_vert++] = asml_->trans_z_ - half_size;

      texcoord_ptr[ind_txcd++] = 0.0f;
      texcoord_ptr[ind_txcd++] = tex_y_coord;
      texcoord_ptr[ind_txcd++] = 0.0f;

      // corner 2
      vertices_ptr[ind_vert++] = asml_->trans_x_ + half_size;
      vertices_ptr[ind_vert++] = geo_y_coord;
      vertices_ptr[ind_vert++] = asml_->trans_z_ - half_size;

      texcoord_ptr[ind_txcd++] = 1.0f;
      texcoord_ptr[ind_txcd++] = tex_y_coord;
      texcoord_ptr[ind_txcd++] = 0.0f;

      // corner 3
      vertices_ptr[ind_vert++] = asml_->trans_x_ + half_size;
      vertices_ptr[ind_vert++] = geo_y_coord;
      vertices_ptr[ind_vert++] = asml_->trans_z_ + half_size;

      texcoord_ptr[ind_txcd++] = 1.0f;
      texcoord_ptr[ind_txcd++] = tex_y_coord;
      texcoord_ptr[ind_txcd++] = 1.0f;

      // corner 4
      vertices_ptr[ind_vert++] = asml_->trans_x_ - half_size;
      vertices_ptr[ind_vert++] = geo_y_coord;
      vertices_ptr[ind_vert++] = asml_->trans_z_ + half_size;

      texcoord_ptr[ind_txcd++] = 0.0f;
      texcoord_ptr[ind_txcd++] = tex_y_coord;
      texcoord_ptr[ind_txcd++] = 1.0f;

    } // for j
    glBindBuffer(GL_ARRAY_BUFFER, vbo_ids_[2]);
    glBufferData(GL_ARRAY_BUFFER, size_vertices + size_texcoord, 0, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size_vertices, vertices_ptr);
    glBufferSubData(GL_ARRAY_BUFFER, size_vertices, size_texcoord, texcoord_ptr);

    // y
    ind_vert = 0;
    ind_txcd = 0;
    for (int j = length-1; j >= 0; --j)
    {
      float tex_y_coord = (j+0.5) / static_cast<float>(length);
      float geo_y_coord = tex_y_coord * asml_->box_size_ + asml_->trans_y_ - half_size;

      // corner 1
      vertices_ptr[ind_vert++] = asml_->trans_x_ - half_size;
      vertices_ptr[ind_vert++] = geo_y_coord;
      vertices_ptr[ind_vert++] = asml_->trans_z_ - half_size;

      texcoord_ptr[ind_txcd++] = 0.0f;
      texcoord_ptr[ind_txcd++] = tex_y_coord;
      texcoord_ptr[ind_txcd++] = 0.0f;

      // corner 2
      vertices_ptr[ind_vert++] = asml_->trans_x_ + half_size;
      vertices_ptr[ind_vert++] = geo_y_coord;
      vertices_ptr[ind_vert++] = asml_->trans_z_ - half_size;

      texcoord_ptr[ind_txcd++] = 1.0f;
      texcoord_ptr[ind_txcd++] = tex_y_coord;
      texcoord_ptr[ind_txcd++] = 0.0f;

      // corner 3
      vertices_ptr[ind_vert++] = asml_->trans_x_ + half_size;
      vertices_ptr[ind_vert++] = geo_y_coord;
      vertices_ptr[ind_vert++] = asml_->trans_z_ + half_size;

      texcoord_ptr[ind_txcd++] = 1.0f;
      texcoord_ptr[ind_txcd++] = tex_y_coord;
      texcoord_ptr[ind_txcd++] = 1.0f;

      // corner 4
      vertices_ptr[ind_vert++] = asml_->trans_x_ - half_size;
      vertices_ptr[ind_vert++] = geo_y_coord;
      vertices_ptr[ind_vert++] = asml_->trans_z_ + half_size;

      texcoord_ptr[ind_txcd++] = 0.0f;
      texcoord_ptr[ind_txcd++] = tex_y_coord;
      texcoord_ptr[ind_txcd++] = 1.0f;

    } // for j
    glBindBuffer(GL_ARRAY_BUFFER, vbo_ids_[3]);
    glBufferData(GL_ARRAY_BUFFER, size_vertices + size_texcoord, 0, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size_vertices, vertices_ptr);
    glBufferSubData(GL_ARRAY_BUFFER, size_vertices, size_texcoord, texcoord_ptr);

    // Z
    ind_vert = 0;
    ind_txcd = 0;
    for (int k = 0; k < length; ++k)
    {
      float tex_z_coord = (k+0.5) / static_cast<float>(length);
      float geo_z_coord = tex_z_coord * asml_->box_size_ + asml_->trans_z_ - half_size;

      // corner 1
      vertices_ptr[ind_vert++] = asml_->trans_x_ - half_size;
      vertices_ptr[ind_vert++] = asml_->trans_y_ - half_size;
      vertices_ptr[ind_vert++] = geo_z_coord;

      texcoord_ptr[ind_txcd++] = 0.0f;
      texcoord_ptr[ind_txcd++] = 0.0f;
      texcoord_ptr[ind_txcd++] = tex_z_coord;

      // corner 2
      vertices_ptr[ind_vert++] = asml_->trans_x_ + half_size;
      vertices_ptr[ind_vert++] = asml_->trans_y_ - half_size;
      vertices_ptr[ind_vert++] = geo_z_coord;

      texcoord_ptr[ind_txcd++] = 1.0f;
      texcoord_ptr[ind_txcd++] = 0.0f;
      texcoord_ptr[ind_txcd++] = tex_z_coord;

      // corner 3
      vertices_ptr[ind_vert++] = asml_->trans_x_ + half_size;
      vertices_ptr[ind_vert++] = asml_->trans_y_ + half_size;
      vertices_ptr[ind_vert++] = geo_z_coord;

      texcoord_ptr[ind_txcd++] = 1.0f;
      texcoord_ptr[ind_txcd++] = 1.0f;
      texcoord_ptr[ind_txcd++] = tex_z_coord;

      // corner 4
      vertices_ptr[ind_vert++] = asml_->trans_x_ - half_size;
      vertices_ptr[ind_vert++] = asml_->trans_y_ + half_size;
      vertices_ptr[ind_vert++] = geo_z_coord;

      texcoord_ptr[ind_txcd++] = 0.0f;
      texcoord_ptr[ind_txcd++] = 1.0f;
      texcoord_ptr[ind_txcd++] = tex_z_coord;
    } // for k
    glBindBuffer(GL_ARRAY_BUFFER, vbo_ids_[4]);
    glBufferData(GL_ARRAY_BUFFER, size_vertices + size_texcoord, 0, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size_vertices, vertices_ptr);
    glBufferSubData(GL_ARRAY_BUFFER, size_vertices, size_texcoord, texcoord_ptr);

    // z
    ind_vert = 0;
    ind_txcd = 0;
    for (int k = length - 1; k >= 0; --k)
    {
      float tex_z_coord = (k+0.5) / static_cast<float>(length);
      float geo_z_coord = tex_z_coord * asml_->box_size_ + asml_->trans_z_ - half_size;

      // corner 1
      vertices_ptr[ind_vert++] = asml_->trans_x_ - half_size;
      vertices_ptr[ind_vert++] = asml_->trans_y_ - half_size;
      vertices_ptr[ind_vert++] = geo_z_coord;

      texcoord_ptr[ind_txcd++] = 0.0f;
      texcoord_ptr[ind_txcd++] = 0.0f;
      texcoord_ptr[ind_txcd++] = tex_z_coord;

      // corner 2
      vertices_ptr[ind_vert++] = asml_->trans_x_ + half_size;
      vertices_ptr[ind_vert++] = asml_->trans_y_ - half_size;
      vertices_ptr[ind_vert++] = geo_z_coord;

      texcoord_ptr[ind_txcd++] = 1.0f;
      texcoord_ptr[ind_txcd++] = 0.0f;
      texcoord_ptr[ind_txcd++] = tex_z_coord;

      // corner 3
      vertices_ptr[ind_vert++] = asml_->trans_x_ + half_size;
      vertices_ptr[ind_vert++] = asml_->trans_y_ + half_size;
      vertices_ptr[ind_vert++] = geo_z_coord;

      texcoord_ptr[ind_txcd++] = 1.0f;
      texcoord_ptr[ind_txcd++] = 1.0f;
      texcoord_ptr[ind_txcd++] = tex_z_coord;

      // corner 4
      vertices_ptr[ind_vert++] = asml_->trans_x_ - half_size;
      vertices_ptr[ind_vert++] = asml_->trans_y_ + half_size;
      vertices_ptr[ind_vert++] = geo_z_coord;

      texcoord_ptr[ind_txcd++] = 0.0f;
      texcoord_ptr[ind_txcd++] = 1.0f;
      texcoord_ptr[ind_txcd++] = tex_z_coord;
    } // for k
    glBindBuffer(GL_ARRAY_BUFFER, vbo_ids_[5]);
    glBufferData(GL_ARRAY_BUFFER, size_vertices + size_texcoord, 0, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size_vertices, vertices_ptr);
    glBufferSubData(GL_ARRAY_BUFFER, size_vertices, size_texcoord, texcoord_ptr);

    glBindBuffer(GL_ARRAY_BUFFER_ARB, 0);

    delete [] vertices_ptr;
    delete [] texcoord_ptr;

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

    // choose shader and vbo
    GLSLShader * shader = NULL;
    int vbo = 0;
    float step_size = asml_->box_size_ / static_cast<float>(length);

    switch (asml_->camera_orientations_[i_view])
    {
    case 'X':
      shader = shader_x_render_.get();
      vbo = vbo_ids_[0];
      break;
    case 'x':
      shader = shader_x_render_.get();
      vbo = vbo_ids_[1];
      break;
    case 'Y':
      shader = shader_y_render_.get();
      vbo = vbo_ids_[2];
      break;
    case 'y':
      shader = shader_y_render_.get();
      vbo = vbo_ids_[3];
      break;
    case 'Z':
      shader = shader_z_render_.get();
      vbo = vbo_ids_[4];
      break;
    case 'z':
      shader = shader_z_render_.get();
      vbo = vbo_ids_[5];
      break;
    }

    // bind fbo
    rr_fbo_->BeginDraw2FBO();
    {
      glClearColor(0.0, 0.0, 0.0, 1.0);
      glDisable(GL_DEPTH_TEST);
      glEnable(GL_BLEND);
      glBlendFuncSeparate(GL_DST_ALPHA, GL_ONE, GL_DST_ALPHA, GL_ZERO);

      glClear(GL_COLOR_BUFFER_BIT);

      shader->Begin();

      glBindBuffer(GL_ARRAY_BUFFER, vbo);
      glEnableClientState(GL_COLOR_ARRAY);
      glEnableClientState(GL_VERTEX_ARRAY);

      glTexCoordPointer(3, GL_FLOAT, 0, (void*)(3 * 4 * sizeof(float)*length));
      glVertexPointer(3, GL_FLOAT, 0, 0);

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

      // Draw
      glDrawArrays(GL_QUADS, 0, 4 * length);

      glDisableClientState(GL_COLOR_ARRAY);
      glDisableClientState(GL_VERTEX_ARRAY);

      glBindBuffer(GL_ARRAY_BUFFER, 0);


      shader->End();
    }
    rr_fbo_->EndDraw2FBO();


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

    char path_buf[100];
    sprintf(path_buf, "../Data/Camera%02d/show%06d.pfm", i_view, counter);
    PFMImage * tmpimg = new PFMImage(width_, height_, 1, img);
    tmpimg->WriteImage(path_buf);
    delete [] img;
    delete tmpimg;

    ++counter;
#endif

  }

  void RenderGL::render_perturbed(int i_view, GLuint vol_tex, int length, int interval, int slice, int pu, int pv)
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

    ////// adjust the volume
    //glTranslatef(asml_->trans_x_, asml_->trans_y_, asml_->trans_z_);

    // get the inversed camera matrix
    float inv_camera[16];
    asml_->camera_inv_gl_extr_paras_[i_view].GetData(inv_camera);

    // choose shader and vbo
    GLSLShader * shader = NULL;
    int vbo = 0;
    float step_size = asml_->box_size_ / static_cast<float>(length);

    switch (asml_->camera_orientations_[i_view])
    {
    case 'X':
      shader = shader_x_render_.get();
      vbo = vbo_ids_[0];
      break;
    case 'x':
      shader = shader_x_render_.get();
      vbo = vbo_ids_[1];
      break;
    case 'Y':
      shader = shader_y_render_.get();
      vbo = vbo_ids_[2];
      break;
    case 'y':
      shader = shader_y_render_.get();
      vbo = vbo_ids_[3];
      break;
    case 'Z':
      shader = shader_z_render_.get();
      vbo = vbo_ids_[4];
      break;
    case 'z':
      shader = shader_z_render_.get();
      vbo = vbo_ids_[5];
      break;
    }

    // bind fbo
    pr_fbo_->BeginDraw2FBO();
    {
      glClearColor(0.0, 0.0, 0.0, 1.0);
      glDisable(GL_DEPTH_TEST);
      glEnable(GL_BLEND);
      glBlendFuncSeparate(GL_DST_ALPHA, GL_ONE, GL_DST_ALPHA, GL_ZERO);

      glClear(GL_COLOR_BUFFER_BIT);

      shader->Begin();

      glBindBuffer(GL_ARRAY_BUFFER, vbo);
      glEnableClientState(GL_COLOR_ARRAY);
      glEnableClientState(GL_VERTEX_ARRAY);

      glTexCoordPointer(3, GL_FLOAT, 0, (void*)(3 * 4 * sizeof(float)*length));
      glVertexPointer(3, GL_FLOAT, 0, 0);

      // set shader uniforms
      shader->SetUniform1f("fwidth", 1.0f*length);
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
      shader->SetUniform4i("disturbPara", interval, pu, pv, slice);

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_3D, vol_tex);
      shader->SetUniform1i("volumeTex", 0);

      shader->SetUniformMatrix4fv("cameraInv", 1, GL_FALSE, inv_camera);

      // Draw
      glDrawArrays(GL_QUADS, 0, 4 * length);

      glDisableClientState(GL_COLOR_ARRAY);
      glDisableClientState(GL_VERTEX_ARRAY);

      glBindBuffer(GL_ARRAY_BUFFER, 0);

      shader->End();
    }
    pr_fbo_->EndDraw2FBO();

#if 0
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

    char path_buf[100];
    sprintf(path_buf, "../Data/Camera%02d/grad%06d.pfm", i_view, counter);
    PFMImage * tmpimg = new PFMImage(width_, height_, 1, img);
    tmpimg->WriteImage(path_buf);
    delete [] img;
    delete tmpimg;

    //sprintf(path_buf, "../Data/Camera%02d/show%06d.bmp", i_view, counter);
    //tmpBmp.SaveImage(path_buf);
    ++counter;
#endif

  }

  RenderGL::RenderGL(ASModeling *p)
    :asml_(p) {};

} // namespace as_modeling