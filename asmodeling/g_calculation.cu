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
                       uint16 * proj_centers,   // 
                       int * tag_vol,
                       float* g_array,
                       clock_t * timer
                       )
{
  if (threadIdx.x == 0)
  {
    timer[blockIdx.x * gridDim.y + blockIdx.y] = clock();
  }


  int vol_index = index3(threadIdx.x, blockIdx.y, blockIdx.x, blockDim.x);
  int arr_index = tag_vol[vol_index];

  if (arr_index != 0)
  {
    // pixel on the image
    int prtu = n_view * 2 * (arr_index-1) + 2 * i_view;
    uint16 pu = proj_centers[prtu];
    uint16 pv = proj_centers[prtu + 1];

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

          //float fgt = uint8_to_float(gt4.x);
          //float frr = rr4.x;
          //float fpr = pr4.x;

          //gg += -2.0f * (fgt - frr) * (fpr - frr) / disturb_value;

          gg += -2.0f * (uint8_to_float(gt4.x) - rr4.x) *(pr4.x - rr4.x)
            / disturb_value;
        }
      }
      g_array[arr_index] = g_array[arr_index] + gg;
    }

  } // arr_index != 0



  __syncthreads();
  if (threadIdx.x == 0)
  {
    timer[blockIdx.x * gridDim.y + blockIdx.y + gridDim.x * gridDim.y] = clock();
  }

}


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
                         int img_width,
                         int img_height,
                         int i_view,           // which view, relates to ofset in proj_centers...
                         int n_view,           // num of different views/cameras
                         int interval,         // the occupycation radius of projection
                         int pt_slice,
                         int pt_tilesize,
                         int pt_u,
                         int pt_v,
                         uint16 * proj_centers,   // 
                         int * tag_vol,
                         float* g_array
                         //clock_t * timer
                         )
{
  //if (threadIdx.x == 0)
  //{
  //  timer[blockIdx.x * blockDim.x + threadIdx.x] = clock();
  //}

  int vol_index = index3(
    pt_slice,
    threadIdx.x * pt_tilesize + pt_u,
    blockIdx.x * pt_tilesize + pt_v,
    blockDim.x * pt_tilesize
    );

  int arr_index = tag_vol[ vol_index ];

  if (arr_index != 0)
  {
    // pixel on the image
    int prtu = n_view * 2 * (arr_index-1) + 2 * i_view;
    uint16 pu = proj_centers[prtu];
    uint16 pv = proj_centers[prtu + 1];

    float gg = 0.0f;
    for (int uu = pu - interval; uu <= pu + interval; ++uu)
    {
      for (int vv = pv - interval; vv <= pv + interval; ++vv)
      {
        float4 rr4 = tex2D(render_result, float(uu), float(vv));
        uchar4 gt4 = tex3D(ground_truth, float(uu), float(img_height-1-vv), float(i_view));
        float4 pr4 = tex2D(perturbed_result, float(uu), float(vv));

        gg += -2.0f * (uint8_to_float(gt4.x) - rr4.x) *(pr4.x - rr4.x)
          / disturb_value;
      }
    }
    g_array[arr_index] += gg;

  } // arr_index != 0

  //__syncthreads();
  //if (threadIdx.x == 0)
  //{
  //  timer[blockDim.x * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x] = clock();
  //}
}

// the perturbed slice is perpendicular to Y axis
//
//   x : threadIdx.x
//   y : pt_slice
//   z : blockIdx.x
//  
//
__global__ void calc_g_y(
                         int img_width,
                         int img_height,
                         int i_view,           // which view, relates to ofset in proj_centers...
                         int n_view,           // num of different views/cameras
                         int interval,         // the occupycation radius of projection
                         int pt_slice,
                         int pt_tilesize,
                         int pt_u,
                         int pt_v,
                         uint16 * proj_centers,   // 
                         int * tag_vol,
                         float* g_array
                         //clock_t * timer
                         )
{
  //if (threadIdx.x == 0)
  //{
  //  timer[blockIdx.x * blockDim.x + threadIdx.x] = clock();
  //}

  int vol_index = index3(
    threadIdx.x * pt_tilesize + pt_u,
    pt_slice,
    blockIdx.x * pt_tilesize + pt_v,
    blockDim.x * pt_tilesize
    );

  int arr_index = tag_vol[ vol_index ];

  if (arr_index != 0)
  {
    // pixel on the image
    int prtu = n_view * 2 * (arr_index-1) + 2 * i_view;
    uint16 pu = proj_centers[prtu];
    uint16 pv = proj_centers[prtu + 1];

    float gg = 0.0f;
    for (int uu = pu - interval; uu <= pu + interval; ++uu)
    {
      for (int vv = pv - interval; vv <= pv + interval; ++vv)
      {
        float4 rr4 = tex2D(render_result, float(uu), float(vv));
        uchar4 gt4 = tex3D(ground_truth, float(uu), float(img_height-1-vv), float(i_view));
        float4 pr4 = tex2D(perturbed_result, float(uu), float(vv));

        gg += -2.0f * (uint8_to_float(gt4.x) - rr4.x) *(pr4.x - rr4.x)
          / disturb_value;
      }
    }
    g_array[arr_index] += gg;

  } // arr_index != 0

  //__syncthreads();
  //if (threadIdx.x == 0)
  //{
  //  timer[blockDim.x * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x] = clock();
  //}
}

// the perturbed slice is perpendicular to Z axis
//
//   x : threadIdx.x
//   y : blockIdx.x
//   z : pt_slice
//  
//
__global__ void calc_g_z(
                         int img_width,
                         int img_height,
                         int i_view,           // which view, relates to ofset in proj_centers...
                         int n_view,           // num of different views/cameras
                         int interval,         // the occupycation radius of projection
                         int pt_slice,
                         int pt_tilesize,
                         int pt_u,
                         int pt_v,
                         uint16 * proj_centers,   // 
                         int * tag_vol,
                         float* g_array
                         //clock_t * timer
                         )
{
  //if (threadIdx.x == 0)
  //{
  //  timer[blockIdx.x * blockDim.x + threadIdx.x] = clock();
  //}

  int vol_index = index3(
    threadIdx.x * pt_tilesize + pt_u,
    blockIdx.x * pt_tilesize + pt_v,
    pt_slice,
    blockDim.x * pt_tilesize
    );

  int arr_index = tag_vol[ vol_index ];

  if (arr_index != 0)
  {
    int prtu = n_view * 2 * (arr_index-1) + 2 * i_view;
    uint16 pu = proj_centers[ prtu ];
    uint16 pv = proj_centers[ prtu + 1];

    float gg = 0.0f;
    for (int uu = pu - interval; uu <= pu + interval; ++uu)
    {
      for (int vv = pv - interval; vv <= pv + interval; ++vv)
      {
        float4 rr4 = tex2D(render_result, float(uu), float(vv));
        uchar4 gt4 = tex3D(ground_truth, float(uu), float(img_height-1-vv), float(i_view));
        float4 pr4 = tex2D(perturbed_result, float(uu), float(vv));

        gg += -2.0f * (uint8_to_float(gt4.x) - rr4.x) *(pr4.x - rr4.x)
          / disturb_value;
      }
    }
    g_array[arr_index] += gg;

  } // arr_index != 0

  //__syncthreads();
  //if (threadIdx.x == 0)
  //{
  //  timer[blockDim.x * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x] = clock();
  //}
}

#endif //_G_CALCULATION_CU_