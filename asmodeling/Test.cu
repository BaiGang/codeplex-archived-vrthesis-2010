#ifndef __TEST_CU__
#define __TEST_CU__

#include <cuda_runtime.h>
#include <utils.cuh>

__global__ void Output_rrprgt(
                              int i_view,
                              int img_width,
                              int img_height,
                              float * rr,
                              float * pr,
                              float * gt
                              )
{
  int v = blockDim.y * blockIdx.y + threadIdx.y;
  int u = blockDim.x * blockIdx.x + threadIdx.x;

  rr[v*img_width + u] = tex2D(render_result, __int2float_rn(u), __int2float_rn(v)).x;

  pr[v*img_width + u] = tex2D(perturbed_result, __int2float_rn(u), __int2float_rn(v)).x;

  gt[v*img_width + u] = tex3D(
    ground_truth, __int2float_rn(u),
    __int2float_rn(img_height-1-v), __int2float_rn(i_view)).x;
}

#endif //__TEST_CU__