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



#endif //__CUDA_GRAD_COMPUTE_KERNEL_CU__