#ifndef _F_CALCULATION_CU_
#define _F_CALCULATION_CU_

#include <cuda_runtime.h>
#include <utils.cuh>

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Calc F on the cuda
//
/////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void calc_f(
                       int i_view,           // which view, relates to ofset in proj_centers...
                       int n_view,           // num of different views/cameras
                       int n,                // num of items in array x, f, and g
                       int interval,         // the occupycation radius of projection
                       int * proj_centers,   // 
                       int * tag_vol,
                       float * f_array
                       )
{
  // the index of the volume cell
  int index_vol   = index3(threadIdx.x, blockIdx.y, blockIdx.x, blockDim.x);

  // the index of the correspoing item in the array
  int index_array = tag_vol[index_vol];

  if (index_array != 0)
  {
    // pixel on the image
    int u = proj_centers[n_view * 2 * (index_array-1)];
    int v = proj_centers[n_view * 2 * (index_array-1)+1];

    float f = 0.0;
    for (int uu = u - interval; uu <= u + interval; ++uu)
    {
      for (int vv = v - interval; vv <= v + interval; ++vv)
      {
        uchar4 rr4 = tex2D(render_result, float(uu), float(vv));
        uchar4 gt4 = tex3D(ground_truth, float(uu), float(vv), float(i_view));
        // USE ONLY R CHANNEL HERE...
        f += (rr4.x-gt4.x)*(rr4.x-gt4.x)/(255.0*255.0);
      }
    }
    f_array[ index_array ] = f;
  } // if (index_array != 0)
}


#endif //_F_CALCULATION_CU_