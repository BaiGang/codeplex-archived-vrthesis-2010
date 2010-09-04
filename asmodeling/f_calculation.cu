#ifndef _F_CALCULATION_CU_
#define _F_CALCULATION_CU_

#include <cuda_runtime.h>
#include <utils.cuh>

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Calc F on the cuda
//
/////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void calc_f_compact(
                               int i_view,
                               int img_height,
                               int range,
                               float * f_array
                               )
{
  int arr_index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

  ushort2 uv = tex3D(
    pcenters,
    __int2float_rn(arr_index & 0x1ff),
    __int2float_rn(arr_index >> 9),
    __int2float_rn(i_view)
    );

  int pu = uv.x;
  int pv = uv.y;

  float ff = 0.0f;
  for (int vv = pv - range; vv <= pv + range; ++vv)
  {
    for (int uu = pu - range; uu <= pu + range; ++uu)
    {
      float4 rr4 = tex2D(render_result, __int2float_rn(uu), __int2float_rn(vv));
      float4 gt4 = tex3D(ground_truth, __int2float_rn(uu), __int2float_rn(img_height-1-vv), __int2float_rn(i_view));

      float tmp = __fadd_rn(-rr4.x, gt4.x);
      ff += __fmul_rn(tmp, tmp);
    }
  }

  f_array[arr_index] = ff;

  //if (blockIdx.x == 0 && threadIdx.x == 0)
  //{
  //  f_array[0] = 0.0f;
  //}
}




#endif //_F_CALCULATION_CU_