#include "asmConfigure.h"

#include "../TinyXML/tinyxml.h"

using namespace asmodeling_block;

namespace
{
  // for convenietly getting values from XML node
#define TXFloatValue( p, v ) (p)->QueryFloatAttribute("value", (v));
#define TXIntValue( p, v ) (p)->QueryIntAttribute("value", (v));

  // return false when failed
#ifndef ELSE_FAILURE
#define ELSE_FAILURE else { return false; }
#endif
} // unnamed namespace


bool Configure::Init(const char * conf_filename, const char * camera_filename)
{
  if (!LoadFile(conf_filename))
  {
    fprintf(stderr, "Error when loading configure file..\n");
    return false;
  }

  if (!LoadCamera(camera_filename))
  {
    fprintf(stderr, "Error when loading camera file..\n");
    return false;
  }

  return true;
}

bool Configure::LoadFile(const char * filename)
{
  // File open
  TiXmlDocument doc(filename);
  if (!doc.LoadFile())
    return false;

  TiXmlHandle hDoc(&doc);
  TiXmlElement * pElem;
  TiXmlHandle hRoot(0);

  //////////////////////////
  // load items
  //////////////////////////
  pElem = hDoc.FirstChildElement().Element();
  if (!pElem)
    return false;
  hRoot = TiXmlHandle(pElem);


  // Participating media parameters
  TiXmlHandle * hPmedia = & (hRoot.FirstChild("PMedia"));

  pElem = hPmedia->FirstChild("extinction").Element();
  if (!pElem)
    return false;
  TXFloatValue(pElem, &extinction_);

  pElem = hPmedia->FirstChild("scattering").Element();
  if (!pElem)
    return false;
  TXFloatValue(pElem, &scattering_);

  pElem = hPmedia->FirstChild("alpha").Element();
  if (!pElem)
    return false;
  TXFloatValue(pElem, &alpha_);

  // Render parameters
  TiXmlHandle * hRender = & (hRoot.FirstChild("Render"));

  pElem = hRender->FirstChild("CurrentView").Element();
  if (!pElem)
    return false;
  TXIntValue(pElem, &current_view_ );

  pElem = hRender->FirstChild("width").Element();
  if (!pElem)
    return false;
  TXIntValue(pElem, &width_ );

  pElem = hRender->FirstChild("height").Element();
  if (!pElem)
    return false;
  TXIntValue(pElem, &height_ );

  TiXmlHandle* hRInterval = &(hRender->FirstChild("RenderInterval"));
  if (!hRInterval)
    return false;

  pElem = hRInterval->FirstChild("Level5").Element();
  if (!pElem)
    return false;
  TXIntValue(pElem, &(render_interval_array_[5]));

  pElem = hRInterval->FirstChild("Level6").Element();
  if (!pElem)
    return false;
  TXIntValue(pElem, &(render_interval_array_[6]));

  pElem = hRInterval->FirstChild("Level7").Element();
  if (!pElem)
    return false;
  TXIntValue(pElem, &(render_interval_array_[7]));

  pElem = hRInterval->FirstChild("Level8").Element();
  if (!pElem)
    return false;
  TXIntValue(pElem, &(render_interval_array_[8]));


  pElem = hRender->FirstChild("RotAngle").Element();
  if (!pElem)
    return false;
  TXIntValue(pElem, &rot_angles_ );

  // Light parameters
  TiXmlHandle * hLight = & (hRoot.FirstChild("Light"));

  pElem = hLight->FirstChild("LightType").Element();
  if (!pElem)
    return false;
  TXIntValue(pElem, &light_type_);

  pElem = hLight->FirstChild("LightIntensityR").Element();
  if (!pElem)
    return false;
  TXFloatValue(pElem, &light_intensity_);


  pElem = hLight->FirstChild("LightX").Element();
  if (!pElem)
    return false;
  TXFloatValue(pElem, &light_x_);

  pElem = hLight->FirstChild("LightY").Element();
  if (!pElem)
    return false;
  TXFloatValue(pElem, &light_y_);

  pElem = hLight->FirstChild("LightZ").Element();
  if (!pElem)
    return false;
  TXFloatValue(pElem, &light_z_);

  // Volume parameters
  TiXmlHandle * hVolume = & (hRoot.FirstChild("Volume"));

  pElem = hVolume->FirstChild("BoxSize").Element();
  if (!pElem)
    return false;
  TXFloatValue(pElem, &box_size_);

  pElem = hVolume->FirstChild("VolumeInitialLevel").Element();
  if (!pElem)
    return false;
  TXIntValue(pElem, &initial_vol_level_);
  initial_vol_size_ = 1 << initial_vol_level_;

  pElem = hVolume->FirstChild("VolumeMaxLevel").Element();
  if (!pElem)
    return false;
  TXIntValue(pElem, &max_vol_level_);
  max_vol_size_ = 1 << max_vol_level_;

  pElem = hVolume->FirstChild("TransX").Element();
  if (!pElem)
    return false;
  TXFloatValue(pElem, &trans_x_);

  pElem = hVolume->FirstChild("TransY").Element();
  if (!pElem)
    return false;
  TXFloatValue(pElem, &trans_y_);

  pElem = hVolume->FirstChild("TransZ").Element();
  if (!pElem)
    return false;
  TXFloatValue(pElem, &trans_z_);

  TiXmlHandle* hVInterval = &(hVolume->FirstChild("VolInterval"));
  if (!hVInterval)
    return false;

  pElem = hVInterval->FirstChild("Level5").Element();
  if (!pElem)
    return false;
  TXIntValue(pElem, &volume_interval_array_[5]);

  pElem = hVInterval->FirstChild("Level6").Element();
  if (!pElem)
    return false;
  TXIntValue(pElem, &volume_interval_array_[6]);

  pElem = hVInterval->FirstChild("Level7").Element();
  if (!pElem)
    return false;
  TXIntValue(pElem, &volume_interval_array_[7]);

  pElem = hVInterval->FirstChild("Level8").Element();
  if (!pElem)
    return false;
  TXIntValue(pElem, &volume_interval_array_[8]);

  // L-BFGS-B parameters
  TiXmlHandle * hLbfgsb = & (hRoot.FirstChild("LBFGSB"));

  pElem = hLbfgsb->FirstChild("disturb").Element();
  if (!pElem)
    return false;
  TXFloatValue(pElem, &disturb_);

  pElem = hLbfgsb->FirstChild("EpsG").Element();
  if (!pElem)
    return false;
  TXFloatValue(pElem, &eps_g_);

  pElem = hLbfgsb->FirstChild("EpsF").Element();
  if (!pElem)
    return false;
  TXFloatValue(pElem, &eps_f_);

  pElem = hLbfgsb->FirstChild("EpsX").Element();
  if (!pElem)
    return false;
  TXFloatValue(pElem, &eps_x_);

  pElem = hLbfgsb->FirstChild("MaxIts").Element();
  if (!pElem)
    return false;
  TXIntValue(pElem, &max_iter_);

  pElem = hLbfgsb->FirstChild("m").Element();
  if (!pElem)
    return false;
  TXIntValue(pElem, &lbfgs_m_);

  pElem = hLbfgsb->FirstChild("ConstrainType").Element();
  if (!pElem)
    return false;
  TXIntValue(pElem, &constrain_type_);

  pElem = hLbfgsb->FirstChild("LowerBound").Element();
  if (!pElem)
    return false;
  TXFloatValue(pElem, &lower_boundary_);

  pElem = hLbfgsb->FirstChild("UpperBound").Element();
  if (!pElem)
    return false;
  TXFloatValue(pElem, &upper_boundary_);

  // CUDA parameters
  TiXmlHandle * hCuda = &(hRoot.FirstChild("CUDA"));

  return true;
  return true;
}

bool Configure::SaveFile(const char *filename)
{
  return false;
}

bool Configure::LoadCamera(const char * filename)
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
      proj[0] = 2.0f*camera_intr_paras_[i](0,0)/width_;
      proj[5] = 2.0f*camera_intr_paras_[i](1,1)/height_;
      proj[8] = -2.0f*(camera_intr_paras_[i](0,2)-width_*0.5f)/width_;
      proj[9] = 2.0f*(camera_intr_paras_[i](1,2)-height_*0.5f)/height_;
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