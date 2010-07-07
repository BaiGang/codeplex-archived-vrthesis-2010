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
                       int img_width,
                       int img_height,
                       int i_view,           // which view, relates to ofset in proj_centers...
                       int n_view,           // num of different views/cameras
                       //int n,                // num of items in array x, f, and g
                       int interval,         // the occupycation radius of projection
                       uint16 * proj_centers,   // 
                       int * tag_vol,
                       float * f_array,
                       clock_t * timer
                       )
{
  if (threadIdx.x == 0)
  {
    timer[blockIdx.x * gridDim.y + blockIdx.y] = clock();
  }

  // the index of the volume cell
  int index_vol   = index3(threadIdx.x, blockIdx.y, blockIdx.x, blockDim.x);

  // the index of the correspoing item in the array
  int index_array = tag_vol[index_vol];

  if (index_array != 0)
  {
    // pixel on the image
    int ptru = n_view * 2 * (index_array-1) + 2 * i_view;
    uint16 u = proj_centers[ ptru ];
    uint16 v = proj_centers[ ptru + 1];

    ///////////
    //data [v * img_width + u] = 1.0f;
    //////////

    float f = 0.0;
    for (int uu = u - interval; uu <= u + interval; ++uu)
    {
      for (int vv = v - interval; vv <= v + interval; ++vv)
      {
        float4 rr4 = tex2D(render_result, float(uu)+0.5, float(vv)+0.5);
        uchar4 gt4 = tex3D(ground_truth, float(uu)+0.5, float(img_height-1-vv)+0.5, float(i_view)+0.5);

        //float frr = rr4.x;
        float fgt = uint8_to_float(gt4.x);

        //// USE ONLY R CHANNEL HERE...
        //f += (frr - fgt)*(frr - fgt);
        f += (rr4.x - fgt) * (rr4.x - fgt);
      }
    }
    f_array[ index_array ] = f;


  } // if (index_array != 0)



  __syncthreads();
  if (threadIdx.x == 0)
  {
    timer[blockIdx.x * gridDim.y + blockIdx.y + gridDim.x * gridDim.y] = clock();
  }
}


#endif //_F_CALCULATION_CU_