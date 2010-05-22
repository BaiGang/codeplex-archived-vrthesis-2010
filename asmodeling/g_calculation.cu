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
  unsigned int k = __mul24(gridDim.x, blockIdx.x);
  unsigned int j = __mul24(gridDim.y, blockIdx.y);
  unsigned int i = __mul24(blockDim.x, threadIdx.x);

  int vol_index = index3(i, j, k, blockDim.x);
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
        uchar4 rr4 = tex2D(render_result, float(uu), float(vv));
        uchar4 gt4 = tex3D(ground_truth, float(uu), float(vv), float(i_view));
        uchar4 pr4 = tex2D(perturbed_result, float(uu), float(vv));

        gg += -2.0f * (float)(gt4.x-rr4.x) * (float)(pr4.x-rr4.x)/(255.0*255.0*disturb_value);
      }
    }

    g_array[arr_index] = g_array[arr_index] + gg;
  } // arr_index != 0
}

#endif //_G_CALCULATION_CU_