#include <cstdlib>
#include <cstdio>

#include <cuda_runtime.h>

#include "cuda_util.cuh"

// functions of cuda-raymarching
extern "C" 
{

}

/////////////////////////////////////////////////////////////
//
//       GLOBAL VARIABLES for Camera Parameters
//
/////////////////////////////////////////////////////////////
 int n_views[1];  // num of views, array just for copy
__constant__ float intr_camera_para[ 16 * 128 ];  // 16 per view, max 128 views, that's abundant
__constant__ float extr_camera_para[ 16 * 128 ];

int current_level;

cudaArray * indicator_volume = 0;
texture<int, 3, cudaReadModeElementType> indicator_tex;

cudaPitchedPtr density_volume_pptr;



/////////////////////////////////////////////////////////////
//
//       DEVICE CODE
//
/////////////////////////////////////////////////////////////

// construct the density volume
// the volume is a length^3 cube
// each z is dispatched to blocks
// x, y are within a block
// 
__global__ void construct_denvol_kernel(cudaPitchedPtr vol_pptr, float * device_x, int length)
{
  unsigned int z = blockIdx.x;
  unsigned int y = threadIdx.y;
  unsigned int x = threadIdx.x;

  if(z >= length || y >= length || x >= length)
  {
    return;
  }

  int texfch = tex3D(indicator_tex, x, y, z);

  size_t slice_pitch = vol_pptr.pitch * length;

  *(float*)((char*)vol_pptr.ptr + z*slice_pitch + y * vol_pptr.pitch +x) = device_x[texfch];
}


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
///////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////
//
//       HOST CODE FOR GLOBAL INIT
//
/////////////////////////////////////////////////////////////
extern "C"
bool set_cameras(int n_camera, float * intr_para, float * extr_para)
{
  int tmp_n_camera[1] = {n_camera};

  cudaMemcpyToSymbol("n_views", tmp_n_camera, sizeof(int));

  cudaMemcpyToSymbol("intr_camera_para", intr_para, n_camera*16*sizeof(float));
  cudaMemcpyToSymbol("extr_camera_para", extr_para, n_camera*16*sizeof(float));

  CUT_CHECK_ERROR( "==Set Cameras==" );

  return true;
}

/////////////////////////////////////////////////////////////
//
//       HOST CODE FOR CURRENT FRAME INIT
//
/////////////////////////////////////////////////////////////
// set the level and the volume indicator
// so that we can construct the density volume
// on the GPU side
extern "C"
void set_density_indicator(int level, int * indicator)
{
  current_level = level;
  int length = 1<<level;

  cudaExtent extent = make_cudaExtent(length, length, length);
  cudaChannelFormatDesc channel_desc = {0};
  channel_desc.f = cudaChannelFormatKindSigned;
  channel_desc.x = 32;
  channel_desc.y = 0;
  channel_desc.z = 0;
  channel_desc.w = 0;

  cudaMalloc3DArray(&indicator_volume, &channel_desc, extent);

  cudaMemcpy3DParms copy_to_indicator = {0};
  copy_to_indicator.srcPtr = make_cudaPitchedPtr((void*)indicator, length*sizeof(float), length, length);
  copy_to_indicator.dstArray = indicator_volume;
  copy_to_indicator.extent = extent;
  copy_to_indicator.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&copy_to_indicator);

  indicator_tex.normalized = false;
  indicator_tex.filterMode = cudaFilterModePoint;
  indicator_tex.addressMode[0] = cudaAddressModeWrap;
  indicator_tex.addressMode[1] = cudaAddressModeWrap;
  indicator_tex.addressMode[2] = cudaAddressModeWrap;

  cudaBindTextureToArray(&indicator_tex, indicator_volume, &channel_desc);

  // mem aloc for density volume
  cudaMalloc3D(&density_volume_pptr, extent);

  // get last error
}

extern "C"
void delete_density_indicator()
{
  cudaFreeArray( indicator_volume );
  cudaFree( &density_volume_pptr );
}

/////////////////////////////////////////////////////////////
//
//       HOST CODE FOR CURRENT LEVEL INIT
//
/////////////////////////////////////////////////////////////

// construct the volume on GPU side using
// the array x and volume indicator
extern "C"
void construct_volume_cuda(float * p_x)
{
  int length = 1<<current_level;

  dim3 grid_dim(length, 1, 1);
  dim3 block_dim(length, length, 1);

  // run the kernel to construct volume on device
  

}

extern "C"
void subdivide_volume_cuda(int prev_level, int next_level)
{

}

// the main routine for grad computation
extern "C"
float cuda_grad_compute(float * p_host_x, float * p_host_g, int n)
{
  float * p_device_x;
  float * p_device_g;

  size_t size = n * sizeof(float);

  cudaMalloc((void**)p_device_g, size);
  cudaMalloc((void**)p_device_x, size+1);
  cudaMemcpy(p_device_x, p_host_x, size+1, cudaMemcpyHostToDevice);

  // construct volume
  construct_volume_cuda( p_device_x );

  // render to image 1

  // calc f

  // perturb voxel

  // render to image 2

  // calc g[]

  cudaMemcpy(p_host_g, p_device_g, size, cudaMemcpyDeviceToHost);

  cudaFree(p_device_x);
  cudaFree(p_device_g);

  return 0.0f;
}

