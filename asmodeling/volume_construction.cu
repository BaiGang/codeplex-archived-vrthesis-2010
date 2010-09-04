#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Reconstruct the volume on the cuda
//
/////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void construct_volume(
                                 cudaPitchedPtr vol_pptr, 
                                 float * device_x
                                 )
{
  int tag_ind =  tex3D(
    position_tag,
    __int2float_rn(threadIdx.x),
    __int2float_rn(blockIdx.y),
    __int2float_rn(blockIdx.x)
    );

  char * slice = (char *)vol_pptr.ptr + blockIdx.x * vol_pptr.pitch * blockDim.x;
  float *  row = (float*)(slice + blockIdx.y * vol_pptr.pitch);

  row[threadIdx.x] = device_x[ tag_ind ];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Reconstruct the volume on the cuda
//    linear memory edition
//
/////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void construct_volume_linm(
                                      float * density_vol,
                                      float * device_x
                                      )
{
  int index = index3(threadIdx.x, blockIdx.y, blockIdx.x, blockDim.x);

  int tag_ind =  tex3D(
    position_tag,
    __int2float_rn(threadIdx.x),
    __int2float_rn(blockIdx.y),
    __int2float_rn(blockIdx.x)
    );

  density_vol[index] = device_x[ tag_ind ];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
//
//    Get volume data from the pitched ptr
//
/////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void get_volume(
                           cudaPitchedPtr vol_pptr,
                           float * den_vol
                           )
{
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
  float * device_x
  )
{
  char * slice = (char *)vol_pptr.ptr + blockIdx.x * vol_pptr.pitch * blockDim.x;

  int tag_ind =  tex3D(
    position_tag,
    __int2float_rn(threadIdx.x>>1),
    __int2float_rn(blockIdx.y>>1),
    __int2float_rn(blockIdx.x>>1)
    );

  *((float*)(slice+blockIdx.y*vol_pptr.pitch) + threadIdx.x)
    = device_x[ tag_ind ];
}

__global__ void cull_empty_cells (cudaPitchedPtr vol_pptr)
{
  int tag_ind =  tex3D(
    position_tag,
    __int2float_rn(threadIdx.x),
    __int2float_rn(blockIdx.y),
    __int2float_rn(blockIdx.x)
    );

  if (tag_ind == 0)
  {
    char * slice = (char*)vol_pptr.ptr + blockIdx.x * vol_pptr.pitch * blockDim.x;
    *((float*)(slice + blockIdx.y*vol_pptr.pitch) + threadIdx.x) = 0.0f;
  }
}

__global__ void get_guess_x (cudaPitchedPtr vol_pptr,
                             float * guess_x)
{
  int tag_ind =  tex3D(
    position_tag,
    __int2float_rn(threadIdx.x),
    __int2float_rn(blockIdx.y),
    __int2float_rn(blockIdx.x)
    );

  if (tag_ind != 0)
  {
    char * slice = (char*)vol_pptr.ptr + blockIdx.x * vol_pptr.pitch * blockDim.x;
    float value = *((float*)(slice + blockIdx.y*vol_pptr.pitch) + threadIdx.x);

    guess_x[tag_ind] = value;
  }
}

