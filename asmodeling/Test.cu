#ifndef __TEST_CU__
#define __TEST_CU__

#include <cuda_runtime.h>
#include <utils.cuh>


__global__ void test_kernel(
                            int width,
                            int height,
                            int iview,
                            float * data1,
                            float * data2
                            )
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < width && j < height)
  {
    float4 rr4 = tex2D(render_result, float(i), float(j));
    data1[j * width + i] = rr4.x;
    uchar4 gt4 = tex3D(ground_truth, float(i), float(height-1-j), float(iview));
    data2[j * width + i] = uint8_to_float(gt4.x);
  } // within the img
}

__global__ void test_g_kernel(
                              int width,
                              int height,
                              int iview,
                              float * data
                              )
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < width && j < height)
  {
    float4 pr4 = tex2D(perturbed_result, float(i), float(j));
    data[j*width + i] = pr4.x;
  } // within the img
}



// test projecting centers
__global__ void test_projected_centers(
                                       int width,
                                       int height,
                                       int iview,
                                       int nview,
                                       uint16 * pcenters,
                                       int * tag_vol,
                                       float * data
                                       )           
{
  // the index of the volume cell
  int index_vol   = index3(threadIdx.x, blockIdx.y, blockIdx.x, blockDim.x);

  // the index of the correspoing item in the array
  int index_array = tag_vol[index_vol];

  if (index_array != 0)
  {

    ////
    int ptru = nview * 2 * (index_array-1) + 2 * iview;

    uint16 u = pcenters[ ptru ];
    uint16 v = pcenters[ ptru + 1];

    if (u < width && v < height)
      data [v*width+u] = 1.0;

  } // if (index_array != 0)

}

#endif //__TEST_CU__