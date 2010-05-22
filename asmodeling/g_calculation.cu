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

  } // arr_index != 0
}

#endif //_G_CALCULATION_CU_