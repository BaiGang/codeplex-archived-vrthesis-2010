#ifndef _G_CALCULATION_CU_
#define _G_CALCULATION_CU_

#include <cuda_runtime.h>
#include "utils.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Calc G[] on the cuda
//
/////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void calc_g(
                       int img_width,
                       int img_height,
                       int i_view,           // which view, relates to ofset in proj_centers...
                       int n_view,           // num of different views/cameras
                       //int n,                // num of items in array x, f, and g
                       int interval,         // the occupycation radius of projection
                       unsigned short * proj_centers,   // 
                       int * tag_vol,
                       float* g_array
                       )
{
  int vol_index = index3(threadIdx.x, blockIdx.y, blockIdx.x, blockDim.x);
  int arr_index = tag_vol[vol_index];

  if (arr_index != 0)
  {
    // pixel on the image
    int pu = proj_centers[n_view * 2 * (arr_index-1) + 2 * i_view];
    int pv = proj_centers[n_view * 2 * (arr_index-1) + 2 * i_view + 1];

    if (pu < img_width && pv < img_height)
    {
      float gg = 0.0f;
      for (int uu = pu - interval; uu <= pu + interval; ++uu)
      {
        for (int vv = pv - interval; vv <= pv + interval; ++vv)
        {
          float4 rr4 = tex2D(render_result, float(uu)+0.5, float(vv)+0.5);
          uchar4 gt4 = tex3D(ground_truth, float(uu)+0.5, float(img_height-1-vv)+0.5, float(i_view)+0.5);
          float4 pr4 = tex2D(perturbed_result, float(uu)+0.5, float(vv)+0.5);

          gg += -2.0f * (gt4.x/255.0-rr4.x) * (float)(pr4.x-rr4.x)/(disturb_value);
        }
      }
      g_array[arr_index] = g_array[arr_index] + gg;
    }

  } // arr_index != 0
}

#endif //_G_CALCULATION_CU_