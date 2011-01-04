#include <cuda_runtime.h>

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
  int index = index3(threadIdx.x, blockIdx.y, blockIdx.x, blockDim.x);

  char * slice = (char *)vol_pptr.ptr + blockIdx.x * vol_pptr.pitch * blockDim.x;
  float *  row = (float*)(slice + blockIdx.y * vol_pptr.pitch);

  row[threadIdx.x] = device_x[ tag_vol[index] ];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Reconstruct the volume on the cuda
//    linear memory edition
//
/////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void construct_volume_linm(
                                      float * density_vol,
                                      float * device_x,
                                      int   * tag_vol
                                      )
{
  int index = index3(threadIdx.x, blockIdx.y, blockIdx.x, blockDim.x);

  density_vol[index] = device_x[ tag_vol[index] ];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
//
//    Get volume data from the pitched ptr
//
/////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void get_volume(cudaPitchedPtr vol_pptr,
                           //int   * tag_vol,
                           float * den_vol)
{
  //unsigned int i = threadIdx.x;
  //unsigned int j = blockIdx.y;
  //unsigned int k = blockIdx.x;
  //unsigned int size = blockDim.x;

  int vol_index = index3(threadIdx.x, blockIdx.y, blockIdx.x, blockDim.x);
  //int arr_index = tag_vol[ vol_index ];

  char * slice = (char*)vol_pptr.ptr + blockIdx.x * vol_pptr.pitch * blockDim.x;
  float *  row = (float*)(slice + blockIdx.y * vol_pptr.pitch);

  den_vol[vol_index] = row [threadIdx.x];
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Upsample current level to max_level
//
/////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void upsample_volume(
                                cudaPitchedPtr pptr_lower,
                                cudaExtent     extent_lower,
                                cudaPitchedPtr pptr_higher,
                                int scale
                                )
{
  // for higher level volume
  unsigned int k = blockIdx.x;
  unsigned int j = blockIdx.y;
  unsigned int i = threadIdx.x;

  char * slice = (char*)pptr_higher.ptr + k*pptr_higher.pitch*blockDim.x;
  float *p_higher = (float*)(slice + j*pptr_higher.pitch) + i;

  k /= scale;
  j /= scale;
  i /= scale;

  slice = (char*)pptr_lower.ptr + k*pptr_lower.pitch*extent_lower.depth;
  float *p_lower  = (float*)(slice + j*pptr_lower.pitch) + i;

  *p_higher = *p_lower;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Upsample previous level to current level
//
/////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void construct_volume_from_prev(
  cudaPitchedPtr vol_pptr,
  float * device_x,
  int * tag_vol )
{
  unsigned int i = threadIdx.x / 2;
  unsigned int j = blockIdx.y / 2;
  unsigned int k = blockIdx.x / 2;
  unsigned int size = blockDim.x / 2;

  int index = index3(i, j, k, size);

  char * slice = (char *)vol_pptr.ptr + blockIdx.x * vol_pptr.pitch * blockDim.x;

  *((float*)(slice+blockIdx.y*vol_pptr.pitch) + threadIdx.x)
    = device_x[ tag_vol[index] ];
}

__global__ void cull_empty_cells (cudaPitchedPtr vol_pptr,
                                  int * tag_vol)
{
  unsigned int i = threadIdx.x;
  unsigned int j = blockIdx.y;
  unsigned int k = blockIdx.x;
  unsigned int size = blockDim.x;

  int index = index3(i, j, k, size);

  if (tag_vol[index] == 0)
  {
    char * slice = (char*)vol_pptr.ptr + blockIdx.x * vol_pptr.pitch * blockDim.x;
    *((float*)(slice + blockIdx.y*vol_pptr.pitch) + threadIdx.x) = 0.0f;
  }
}

__global__ void get_guess_x (cudaPitchedPtr vol_pptr,
                             int *tag_vol,
                             float * guess_x)
{
  unsigned int i = threadIdx.x;
  unsigned int j = blockIdx.y;
  unsigned int k = blockIdx.x;
  unsigned int size = blockDim.x;

  int index = index3(i, j, k, size);
  int ind_array = tag_vol[ index ];

  if (ind_array != 0)
  {
    char * slice = (char*)vol_pptr.ptr + blockIdx.x * vol_pptr.pitch * blockDim.x;
    float value = *((float*)(slice + blockIdx.y*vol_pptr.pitch) + threadIdx.x);

    guess_x[ind_array] = value;
  }
}

