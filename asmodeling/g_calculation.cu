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
                       int i_view,           // which view, relates to ofset in proj_centers...
                       int n_view,           // num of different views/cameras
                       int n,                // num of items in array x, f, and g
                       int interval,         // the occupycation radius of projection
                       int * proj_centers,   // 
                       int * tag_vol,
                       float* g_array
                       )
{
  int vol_index = index3(threadIdx.x, blockIdx.y, blockIdx.x, blockDim.x);
  int arr_index = tag_vol[vol_index];

  if (arr_index != 0)
  {
    int pu = proj_centers[2 * n_view * (arr_index-1)];
    int pv = proj_centers[2 * n_view * (arr_index-1) + 1];

    float gg = 0.0f;

    for (int uu = pu - interval; uu <= pu + interval; ++uu)
    {
      for (int vv = pv - interval; vv <= pv + interval; ++vv)
      {
        float4 rr4 = tex2D(render_result, float(uu), float(vv));
        uchar4 gt4 = tex3D(ground_truth, float(uu), float(vv), float(i_view));
        float4 pr4 = tex2D(perturbed_result, float(uu), float(vv));

        gg += -2.0f * (gt4.x/255.0-rr4.x) * (float)(pr4.x-rr4.x)/(disturb_value);
      }
    }

    g_array[arr_index] = g_array[arr_index] + gg;
  } // arr_index != 0
}

#endif //_G_CALCULATION_CU_