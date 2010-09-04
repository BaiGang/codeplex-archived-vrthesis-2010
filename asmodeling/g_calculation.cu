#ifndef _G_CALCULATION_CU_
#define _G_CALCULATION_CU_

#include <cuda_runtime.h>
#include "utils.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Calc G[] on the cuda
//
/////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
//
//  Calc and sum g[] only on perturbed slice...
//
//
/////////////////////////////////////////////////////////

// the perturbed slice is perpendicular to X axis
//
//   x : pt_slice
//   y : threadIdx.x
//   z : blockIdx.x
//  
//
__global__ void calc_g_x(
                         int img_height,
                         int i_view,           // which view, relates to ofset in proj_centers...
                         int interval,         // the occupycation radius of projection
                         int pt_slice,
                         int pt_tilesize,
                         int pt_u,
                         int pt_v,
                         float* g_array
                         )
{
  int arr_index = tex3D(
    position_tag,
    __int2float_rn(pt_slice),
    __int2float_rn(threadIdx.x * pt_tilesize + pt_u),
    __int2float_rn(blockIdx.x * pt_tilesize + pt_v)
    );

    ushort2 uv = tex3D(
      pcenters,
      __int2float_rn(arr_index & 0x1ff),
      __int2float_rn(arr_index >> 9),
      __int2float_rn(i_view)
      );

    int pu = uv.x;
    int pv = uv.y;

    float gg = 0.0f;
    for (int uu = pu - interval; uu <= pu + interval; ++uu)
    {
      for (int vv = pv - interval; vv <= pv + interval; ++vv)
      {
        float4 rr4 = tex2D(render_result, __int2float_rn(uu), __int2float_rn(vv));
        float4 gt4 = tex3D(ground_truth, __int2float_rn(uu), __int2float_rn(img_height-1-vv), __int2float_rn(i_view));
        float4 pr4 = tex2D(perturbed_result, __int2float_rn(uu), __int2float_rn(vv));

        gg += __fdiv_rn(-2.0f * __fmul_rn((gt4.x - rr4.x), (pr4.x - rr4.x)), disturb_value);
      }
    }
    g_array[arr_index] += gg;

}

// the perturbed slice is perpendicular to Y axis
//
//   x : threadIdx.x
//   y : pt_slice
//   z : blockIdx.x
//  
//
__global__ void calc_g_y(
                         int img_height,
                         int i_view,           // which view, relates to ofset in proj_centers...
                         int interval,         // the occupycation radius of projection
                         int pt_slice,
                         int pt_tilesize,
                         int pt_u,
                         int pt_v,
                         float* g_array
                         )
{
  int arr_index = tex3D(
    position_tag,
    __int2float_rn(threadIdx.x * pt_tilesize + pt_u),
    __int2float_rn(pt_slice),
    __int2float_rn(blockIdx.x * pt_tilesize + pt_v)
    );

    ushort2 uv = tex3D(
      pcenters,
      __int2float_rn(arr_index & 0x1ff),
      __int2float_rn(arr_index >> 9),
      __int2float_rn(i_view)
      );

    int pu = uv.x;
    int pv = uv.y;

    float gg = 0.0f;
    for (int uu = pu - interval; uu <= pu + interval; ++uu)
    {
      for (int vv = pv - interval; vv <= pv + interval; ++vv)
      {
        float4 rr4 = tex2D(render_result, __int2float_rn(uu), __int2float_rn(vv));
        float4 gt4 = tex3D(ground_truth, __int2float_rn(uu), __int2float_rn(img_height-1-vv), __int2float_rn(i_view));
        float4 pr4 = tex2D(perturbed_result, __int2float_rn(uu), __int2float_rn(vv));

        gg += __fdiv_rn(-2.0f * __fmul_rn((gt4.x - rr4.x), (pr4.x - rr4.x)), disturb_value);

      }
    }
    g_array[arr_index] += gg;

}

// the perturbed slice is perpendicular to Z axis
//
//   x : threadIdx.x
//   y : blockIdx.x
//   z : pt_slice
//  
//
__global__ void calc_g_z(
                         int img_height,
                         int i_view,           // which view, relates to ofset in proj_centers...
                         int interval,         // the occupycation radius of projection
                         int pt_slice,
                         int pt_tilesize,
                         int pt_u,
                         int pt_v,
                         float* g_array
                         )
{
  int arr_index = tex3D(
    position_tag,
    __int2float_rn(threadIdx.x * pt_tilesize + pt_u),
    __int2float_rn(blockIdx.x * pt_tilesize + pt_v),
    __int2float_rn(pt_slice)
    );

    ushort2 uv = tex3D(
      pcenters,
      __int2float_rn(arr_index & 0x1ff),
      __int2float_rn(arr_index >> 9),
      __int2float_rn(i_view)
      );

    int pu = uv.x;
    int pv = uv.y;

    float gg = 0.0f;
    for (int uu = pu - interval; uu <= pu + interval; ++uu)
    {
      for (int vv = pv - interval; vv <= pv + interval; ++vv)
      {
        float4 rr4 = tex2D(render_result, __int2float_rn(uu), __int2float_rn(vv));
        float4 gt4 = tex3D(ground_truth, __int2float_rn(uu), __int2float_rn(img_height-1-vv), __int2float_rn(i_view));
        float4 pr4 = tex2D(perturbed_result, __int2float_rn(uu), __int2float_rn(vv));

        gg += __fdiv_rn(-2.0f * __fmul_rn((gt4.x - rr4.x), (pr4.x - rr4.x)), disturb_value);

      }
    }
    g_array[arr_index] += gg;

}


#endif //_G_CALCULATION_CU_