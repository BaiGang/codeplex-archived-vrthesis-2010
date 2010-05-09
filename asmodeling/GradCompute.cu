#include <cuda_runtime.h>

// map (i,j,k) to 1D array index
__device__ int index3(int i, int j, int k, int length)
{
  return i + length * ( j + k * length );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Reconstruct the volume on the cuda
//
/////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void construct_volume(
                                 cudaPitchedPtr vol_pptr, 
                                 float * device_x, 
                                 int * tag_vol
                                 )
{
  // z = blockDim.x
  // y = threadDim.y
  // x = threadDim.x
  // slice_pitch = pitch * height
  // slice = ptr + z * slice_pitch
  // row = (type*)(slice + y*pitch)
  // elem = *(row + x)

  int index = index3(threadIdx.x, threadIdx.y, blockIdx.x, blockDim.x);

  char * slice = (char *)vol_pptr.ptr + blockIdx.x * vol_pptr.pitch * blockDim.x;

  *((float*)(slice+threadIdx.y*vol_pptr.pitch) + threadIdx.x)
    = device_x[ tag_vol[index] ];

}
void construct_volume_cuda(
                           float * device_x,
                           cudaPitchedPtr * density_vol,
                           cudaExtent extent,
                           int *   tag_vol
                           )
{

  dim3 grid_dim(extent.depth, 1, 1);
  dim3 block_dim(extent.width, extent.height, 1);

  construct_volume<<< grid_dim, block_dim >>>(
    *density_vol,
    device_x,
    tag_vol
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Upsample current level to max_level
//
/////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void upsample_volume(
                                cudaPitchedPtr pptr_lower,
                                cudaExtent     extent_lower,
                                cudaPitchedPtr pptr_higher
                                )
{
  // for higher level volume
  unsigned int k = blockIdx.x;
  unsigned int j = threadIdx.y;
  unsigned int i = threadIdx.x;

  int scale = blockDim.x / extent_lower.depth;

  char * slice = (char*)pptr_higher.ptr + k*pptr_higher.pitch*blockDim.x;
  float *p_higher = (float*)(slice + j*pptr_higher.pitch) + i;


  k /= scale;
  j /= scale;
  k /= scale;

  slice = (char*)pptr_lower.ptr + k*pptr_lower.pitch*extent_lower.depth;
  float *p_lower  = (float*)(slice + j*pptr_lower.pitch) + i;

  *p_higher = *p_lower;
}
void upsample_volume_cuda(
                          int level,
                          int max_level,
                          cudaPitchedPtr * lower_lev,
                          cudaPitchedPtr * upper_lev
                          )
{
  int length = 1 << level;
  int max_length = 1 << max_level;

  dim3 grid_dim(max_length, 1, 1);
  dim3 block_dim(max_length, max_length, 1);

  cudaExtent ext_low = make_cudaExtent(length, length, length);

  upsample_volume<<< grid_dim, block_dim >>>(
    *lower_lev,
    ext_low,
    *upper_lev
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Calc F on the cuda
//
/////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Calc G[] on the cuda
//
/////////////////////////////////////////////////////////////////////////////////////////////////////