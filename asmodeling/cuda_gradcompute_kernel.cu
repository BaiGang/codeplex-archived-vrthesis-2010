#ifndef __CUDA_GRAD_COMPUTE_KERNEL_CU__
#define __CUDA_GRAD_COMPUTE_KERNEL_CU__

#include <cuda.h>

// Kernel function to calculate the sum-of-square-error
__global__ void calc_f(float * ground_truth, float * render_result, int n, float * f_pixels)
{
  //  calc index of the thread
  int thread_ind = threadIdx.y * blockDim.x + threadIdx.x;

  //  calc f for each pixel
  f_pixels[ thread_ind ] += (ground_truth[thread_ind] - render_result[thread_ind])
    * (ground_truth[thread_ind] - render_result[thread_ind]);

  __syncthreads();

  // reduce and sum
  //  the final result will be stored in f_pixels[0]
  for (int i = n; i > 0; i /= 2)
  {
    if (thread_ind < i)
    {
      f_pixels[thread_ind] = f_pixels[thread_ind] + f_pixels[thread_ind +i];
    }

    __syncthreads();
  }
}

// Kernel function to calculate the gradient
__global__ void calc_g(float * ground_truth,
                       float * render_res_imperturbed,
                       float * render_res_perturbed,
                       int *   index_pixels2arrayg,
                       float * o_g,
                       int n,
                       int n_pixels)
{

}


////////////////////////////////////////////////////////////////
//
//   Perturb the voxels to calc gradient
//   The serial edition of the algorithm
//
//First, calc the most close axis to the view direction. This axis
//is the principal axis.
//
//N = the resolution of the slice
//M = number of sub slices
//
//For each slice along the principal axis
//  for group_v = 0 to N/M-1
//    for group_u = 0 to N/M-1  // for each concurrently perturbed group
//    {
//      // firstly, perturb
//
//      for sub_v = 0 to M
//        for sub_u = 0 to M    // for each sub slice
//        {
//          if this voxel contains density
//            perturb it
//        }
//
//      // then, render
//      Render 
//
//      // Sum the gradient
//
//    }


#endif //__CUDA_GRAD_COMPUTE_KERNEL_CU__