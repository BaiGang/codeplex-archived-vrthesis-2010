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

__global__ void Output_RenderResult(
                                    int i_view,
                                    int img_width,
                                    int img_height,
                                    float * rr
                                    )
{
  int v = blockDim.y * blockIdx.y + threadIdx.y;
  int u = blockDim.x * blockIdx.x + threadIdx.x;

  if (u < img_width && v < img_height)
  {
    rr[v * img_width + u] = tex2D(render_result, __int2float_rn(u), __int2float_rn(v)).x;
  }
}

__global__ void Output_PerturbedResult(
                                       int i_view,
                                       int img_width,
                                       int img_height,
                                       float * pr
                                       )
{
  int v = blockDim.y * blockIdx.y + threadIdx.y;
  int u = blockDim.x * blockIdx.x + threadIdx.x;

  if (u < img_width && v < img_height)
  {
    pr[v*img_width + u] = tex2D(perturbed_result, __int2float_rn(u), __int2float_rn(v)).x;
  }
}

__global__ void Output_GroundTruth(
                                   int i_view,
                                   int img_width,
                                   int img_height,
                                   float * gt
                                   )
{
  int v = blockDim.y * blockIdx.y + threadIdx.y;
  int u = blockDim.x * blockIdx.x + threadIdx.x;

  if (u < img_width && v < img_height)
  {
    float4 gt4 = tex3D(
      ground_truth, float(u),
      float(img_height-1-v), float(i_view));

    gt[v*img_width + u] =  gt4.x ;
  }
}


__global__ void ThreadWarmUp(int a)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  float f = __int2float_rn(i) * __int2float_rn(j) / __int2float_rn(a);
}


// threadIdx.x : x
// blockIdx. x : y
// blockIdx. y : z
__global__ void Output_ProjectedCenters(
                                        int i_view,
                                        int img_width,
                                        int img_height,
                                        float * pc
                                        )
{
  ushort2 uv = tex3D(
    pcenters,
    __int2float_rn(threadIdx.x),
    __int2float_rn(blockIdx.x),
    __int2float_rn(i_view)
    );

  int uu = uv.x;
  int vv = uv.y;

  if ( uu < img_width && vv < img_height )
  {
    pc[vv * img_width + uu] = 1.0f;
  }

}

#endif //__TEST_CU__