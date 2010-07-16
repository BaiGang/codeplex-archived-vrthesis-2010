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
                       clock_t * timer1,
                       clock_t * timer2
                       //clock_t * timer3
                       )
{

  // time profiling for fetching global mem
  if (threadIdx.x == 0)
  {
    timer1[blockIdx.x * gridDim.y + blockIdx.y] = clock();
  }

  // the index of the volume cell
  int index_vol   = index3(threadIdx.x, blockIdx.y, blockIdx.x, blockDim.x);

  // the index of the correspoing item in the array
  int index_array = tag_vol[index_vol];

  // time profiling for fetching global mem
  __syncthreads();
  if (threadIdx.x == 0)
  {
    timer1[blockIdx.x * gridDim.y + blockIdx.y + gridDim.x * gridDim.y] = clock();
  }



  __syncthreads();
  if (threadIdx.x == 0)
  {
    timer2[blockIdx.x * gridDim.y + blockIdx.y] = clock();
  }


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
    for (int uu = u - 1; uu <= u + 1; ++uu)
    {
      for (int vv = v - 1; vv <= v + 1; ++vv)
      {

        float4 rr4 = tex2D(render_result, float(u), float(v));
        uchar4 gt4 = tex3D(ground_truth, float(u), float(img_height-1-v), float(i_view));

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
    timer2[blockIdx.x * gridDim.y + blockIdx.y + gridDim.x * gridDim.y] = clock();
  }
}


#endif //_F_CALCULATION_CU_