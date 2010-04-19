#include <cuda.h>
#include <cuda_runtime.h>

// functions of cuda-raymarching
extern "C" 
{

}

// global variables
int n_views;                // num of views
int current_level;          // current level of volume
float ** intr_camera_para;  // camera parameters
float ** extr_camera_para;  // ...


/////////////////////////////////////////////////////////////
//
//       DEVICE CODE
//
/////////////////////////////////////////////////////////////

// Kernel function to calculate the sum-of-square-error
__global__ void calc_f(float * ground_truth, float * render_result, int n, float * f_pixels)
{
  ////  calc index of the thread
  //int thread_ind = threadIdx.y * blockDim.x + threadIdx.x;

  ////  calc f for each pixel
  //f_pixels[ thread_ind ] += (ground_truth[thread_ind] - render_result[thread_ind])
  //  * (ground_truth[thread_ind] - render_result[thread_ind]);

  //__syncthreads();

  //// reduce and sum
  ////  the final result will be stored in f_pixels[0]
  //for (int i = n; i > 0; i /= 2)
  //{
  //  if (thread_ind < i)
  //  {
  //    f_pixels[thread_ind] = f_pixels[thread_ind] + f_pixels[thread_ind +i];
  //  }

  //  __syncthreads();
  //}
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
  //
}

// perturb voxels
__global__ void perturb_voxels()
{
  //
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

/////////////////////////////////////////////////////////////
//
//       HOST CODE
//
/////////////////////////////////////////////////////////////

// set the level and the volume indicator
// so that we can construct the density volume
// on the GPU side
extern "C"
void set_density_indicator(int level, int * indicator)
{

}

extern "C"
void construct_volume(float * p_x)

// the main routine for grad computation
extern "C"
float cuda_grad_compute(float * p_host_x, float * p_host_g, int n)
{
  float * p_device_x;
  float * p_device_g;

  // construct volume


  return 0.0f;
}

extern "C"
void subdivide_volume_cuda(int prev_level, int next_level)
{

}