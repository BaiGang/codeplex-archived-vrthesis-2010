#include <cstdlib>
#include <cstdio>
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


#include "cuda_util.cuh"


/////////////////////////////////////////////////////////////
//
//       GLOBAL VARIABLES on both DEVICE and HOST
//      "static" to prevent solution on namespace
//
/////////////////////////////////////////////////////////////

// the level of volume for current optimization pass
static int current_level;

// the cuda 3D array for tag volume
static cudaArray * tag_volume;
// the 3D texture reference for tag volume
static texture<int, 3, cudaReadModeElementType> voltag_tex;

// the pitched pointer to the density 3D volume
static cudaPitchedPtr density_volume_pptr;


/////////////////////////////////////////////////////////////
//
//       DEVICE CODE
//
/////////////////////////////////////////////////////////////

// construct the density volume
// the volume is a length^3 cube
// each z is dispatched to blocks
// x, y are within a block
__global__ void construct_density_volome(cudaPitchedPtr vol_pptr, float * device_x, int length)
{
  unsigned int z = blockIdx.x;
  unsigned int y = threadIdx.y;
  unsigned int x = threadIdx.x;

  if(z >= length || y >= length || x >= length)
  {
    return;
  }

  int texfch = tex3D(voltag_tex, x, y, z);

  size_t slice_pitch = vol_pptr.pitch * length;

  *(float*)((char*)vol_pptr.ptr + z*slice_pitch + y * vol_pptr.pitch +x) = device_x[texfch];
}



/////////////////////////////////////////////////////////////
//
//       HOST CODE FOR GLOBAL INIT
//
/////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////
//
//       HOST CODE FOR CURRENT FRAME INIT
//
/////////////////////////////////////////////////////////////
// set the level and the volume tags
// so that we can construct the density volume
// on the GPU side
extern "C"
void set_density_indicator(int level, int * tags)
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

  cudaMalloc3DArray(&tag_volume, &channel_desc, extent);

  cudaMemcpy3DParms copy_to_indicator = {0};
  copy_to_indicator.srcPtr = make_cudaPitchedPtr((void*)tags, length*sizeof(int), length, length);
  copy_to_indicator.dstArray = tag_volume;
  copy_to_indicator.extent = extent;
  copy_to_indicator.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&copy_to_indicator);

  voltag_tex.normalized = false;
  voltag_tex.filterMode = cudaFilterModePoint;
  voltag_tex.addressMode[0] = cudaAddressModeWrap;
  voltag_tex.addressMode[1] = cudaAddressModeWrap;
  voltag_tex.addressMode[2] = cudaAddressModeWrap;

  cudaBindTextureToArray(&voltag_tex, tag_volume, &channel_desc);

  // mem aloc for density volume
  cudaMalloc3D(&density_volume_pptr, extent);

  // get last error
}

extern "C"
void delete_density_indicator()
{
  cudaFreeArray( tag_volume );
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
  float * p_device_x ;
  float * p_device_g ;

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

  // copy g[] from device to host 
  cudaMemcpy(p_host_g, p_device_g, size, cudaMemcpyDeviceToHost);

  cudaFree(p_device_x);
  cudaFree(p_device_g);

  return 0.0f;
}

