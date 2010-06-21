// ASMloadcameras.cpp
// Load in camera parameters
// Bai, Gang.
// March 22ed, 2010.

#include <stdafx.h>
#include <cstdio>

#include "ASModeling.h"

namespace as_modeling
{
#ifndef ELSE_FAILURE
#define ELSE_FAILURE else { return false; }
#endif

  bool ASModeling::load_camera_file(const char * filename)
  {
    int itmp;
    float ftmp;

    FILE *fp = fopen(filename, "r");
    if (NULL==fp)
      return false;

    if (fscanf(fp, "%d", &itmp)==1)
    {
      num_cameras_ = itmp;
    }
    ELSE_FAILURE;

    Matrix4 * tmpptr1 = new Matrix4[num_cameras_];
    camera_intr_paras_.reset(tmpptr1);
    tmpptr1 = new Matrix4[num_cameras_];
    camera_extr_paras_.reset(tmpptr1);
    tmpptr1 = new Matrix4[num_cameras_];
    camera_gl_extr_paras_.reset(tmpptr1);
    tmpptr1 = new Matrix4[num_cameras_];
    camera_inv_gl_extr_paras_.reset(tmpptr1);
    tmpptr1 = new Matrix4[num_cameras_];
    gl_projection_mats_.reset(tmpptr1);
    Vector4 * tmpptr2 = new Vector4[num_cameras_];
    camera_positions_.reset(tmpptr2);

    // for each camera
    for (int i = 0; i < num_cameras_; ++i)
    {
      // load in intrinsic parameters
      for (int row = 0; row < 3; ++row)
      {
        for (int col = 0; col < 3; ++col)
        {
          if (fscanf(fp, "%f", & ftmp)==1)
          {
            camera_intr_paras_[i](row,col) = ftmp;
          }
          ELSE_FAILURE;
        } // col
      } // row

      for (int col = 0; col < 4; ++col)
      {
        if (fscanf(fp, "%f", & ftmp)==1)
        {
          camera_intr_paras_[i](3,col) = ftmp;
        }
        ELSE_FAILURE;
      }

      // load in extrinsic parameters
      for (int row = 0; row < 4; ++row)
      {
        for (int col = 0; col < 4; ++col)
        {
          if (fscanf(fp, "%f", & ftmp)==1)
          {
            camera_extr_paras_[i](row, col) = ftmp;
          }
          ELSE_FAILURE;
        } // col
      } // row

      // calc camera position
      Matrix4 camera_mat(camera_extr_paras_[i]);
      camera_mat.Inverse();
      camera_positions_[i].x = camera_mat(0,3);
      camera_positions_[i].y = camera_mat(1,3);
      camera_positions_[i].z = camera_mat(2,3);
      camera_positions_[i].w = 1.0;

      // convert the extr mat to match the gl form
      Matrix4 trans;
      trans.identityMat();
      trans(1,1) = -1;
      trans(2,2) = -1;
      camera_gl_extr_paras_[i] = trans * camera_extr_paras_[i];

      // set the inverse matrix of gl_extr_para
      camera_inv_gl_extr_paras_[i] = camera_gl_extr_paras_[i];
      camera_inv_gl_extr_paras_[i].Inverse();

      // calc glProjection mat from intr paras
      float proj[16];
      float zn = 0.1f;
      float zf = 1000.0f;
      memset(proj, 0, sizeof(proj));
      proj[0] = 2.0f*camera_intr_paras_[i](0,0)/camera_width_;
      proj[5] = 2.0f*camera_intr_paras_[i](1,1)/camera_height_;
      proj[8] = -2.0f*(camera_intr_paras_[i](0,2)-camera_width_*0.5f)/camera_width_;
      proj[9] = 2.0f*(camera_intr_paras_[i](1,2)-camera_height_*0.5f)/camera_height_;
      proj[10] = -(zf+zn)/(zf-zn);
      proj[11] = -1.0f;
      proj[14] = -2.0f*(zf*zn)/(zf-zn);
      gl_projection_mats_[i].SetMatrix(proj);


    } // for i

    fclose(fp);

#if 0
    // for testing
    FILE *debug = fopen("../Data/debug_camera.txt", "w");
    for (int i = 0; i < num_cameras_; ++i)
    {
      //// intr para

      //for(int j=0; j<4;++j)
      //{
      //  for (int k=0; k<4;++k)
      //  {
      //    fprintf(debug, "%f  ", camera_intr_paras_[i](j,k));
      //  }
      //  fprintf(debug, "\n");
      //}
      //fprintf(debug, "\n");


      //// extr para

      //for(int j=0; j<4;++j)
      //{
      //  for (int k=0; k<4;++k)
      //  {
      //    fprintf(debug, "%f  ", camera_extr_paras_[i](j,k));
      //  }
      //  fprintf(debug, "\n");
      //}
      //fprintf(debug, "\n");

      // gl extr para
      fprintf(debug, "GL Extr Para : \n");
      for(int j=0; j<4;++j)
      {
        for (int k=0; k<4;++k)
        {
          fprintf(debug, "%f\t", camera_gl_extr_paras_[i](j,k));
        }
        fprintf(debug, "\n");
      }
      fprintf(debug, "\n");

      // gl projection
      fprintf(debug, "GL projection : \n");
      for (int j = 0; j < 4; ++j)
      {
        for (int k = 0; k < 4; ++k)
        {
          fprintf(debug, "%f\t", gl_projection_mats_[i](j,k));
        }
        fprintf(debug, "\n");
      }
      fprintf(debug, "\n");

      // camera position
      fprintf(debug, " Camera Position : \n");
      for(int j=0; j<4; ++j)
      {
        fprintf(debug, "%f\t", camera_positions_[i][j]);
      }
      fprintf(debug, "\n");

      //// inverse gl extr para
      //Matrix4 mulres = camera_gl_extr_paras_[i] * camera_inv_gl_extr_paras_[i];
      //for(int j=0; j<4; ++j)
      //{
      //  for(int k=0; k<4;++k)
      //  {
      //    fprintf(debug, "%f  ", mulres(j,k));
      //  }
      //  fprintf(debug, "\n");
      //}

    } // for i
    fclose(debug);
#endif

    return true;
  }

} //namespace as_modeling