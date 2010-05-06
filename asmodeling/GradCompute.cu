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
__global__ void construct_volume(float * density_vol, float * device_x, int * tag_vol)
{
  unsigned int k = blockIdx.x;
  unsigned int j = threadIdx.y;
  unsigned int i = threadIdx.x;

  int index = index3(i, j, k, blockDim.x);
  density_vol[index] = device_x[ tag_vol[index] ];
}
void construct_volume_cuda(
                           int level,
                           float * device_x,
                           float * density_vol,
                           int *   tag_vol
                           )
{
  int length = 1<<level;

  dim3 grid_dim(length, 1, 1);
  dim3 block_dim(length, length, 1);

  construct_volume<<< grid_dim, block_dim >>>(
    density_vol,
    device_x,
    tag_vol
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Upsample current level to max_level
//
/////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void upsample_volume(float * i_lower, float * o_upper, int scale)
{
  unsigned int k = blockIdx.x;
  unsigned int j = threadIdx.y;
  unsigned int i = threadIdx.x;

  unsigned int index_upper = index3(i, j, k, blockDim.x);
  unsigned int index_lower = index3( i/scale, j/scale, k/scale, blockDim.x/scale );

  o_upper[index_upper] = i_lower[index_lower];
}
void upsample_volume_cuda(
                          int level,
                          int max_level,
                          float * lower_lev,
                          float * upper_lev
                          )
{
  int scale = 1<<(max_level - level);
  int max_length = 1 << max_level;

  dim3 grid_dim(max_length, 1, 1);
  dim3 block_dim(max_length, max_length, 1);

  upsample_volume<<< grid_dim, block_dim >>>(
    lower_lev,
    upper_lev,
    scale
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