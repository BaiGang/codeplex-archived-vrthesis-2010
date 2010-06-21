// ASMloadcameras.cpp
// Load in camera parameters
// Bai, Gang.
// March 22ed, 2010.

#include <stdafx.h>
#include <cstdio>
#include <tinyxml.h>

#include "ASModeling.h"

namespace
{
#define TXFloatValue( p, v ) (p)->QueryFloatAttribute("value", (v));
#define TXIntValue( p, v ) (p)->QueryIntAttribute("value", (v));
} // unnamed namespace

namespace as_modeling
{
  bool ASModeling::load_configure_file(const char *filename)
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

    pElem = hRender->FirstChild("CameraWidth").Element();
    if (!pElem)
      return false;
    TXIntValue(pElem, &camera_width_ );

    pElem = hRender->FirstChild("CameraHeight").Element();
    if (!pElem)
      return false;
    TXIntValue(pElem, &camera_height_ );

    pElem = hRender->FirstChild("RenderInterval").Element();
    if (!pElem)
      return false;
    TXIntValue(pElem, &render_interval_ );

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

    pElem = hVolume->FirstChild("VolWidth").Element();
    if (!pElem)
      return false;
    TXIntValue(pElem, &box_width_);

    pElem = hVolume->FirstChild("VolHeight").Element();
    if (!pElem)
      return false;
    TXIntValue(pElem, &box_height_);

    pElem = hVolume->FirstChild("VolDepth").Element();
    if (!pElem)
      return false;
    TXIntValue(pElem, &box_depth_);

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

    pElem = hVolume->FirstChild("VolInterval").Element();
    if (!pElem)
      return false;
    TXIntValue(pElem, &volume_interval_);

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
  }

} // namespace as_modeling