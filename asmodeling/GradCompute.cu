#include <cstdio>

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_math.h>

// parallel reduce kernel
#include "reduction_kernel.cu"

//////////////////////////////////////////////////////////////////////////////////////////////////////
//
//    GLOBAL VARIABLES
//
//////////////////////////////////////////////////////////////////////////////////////////////////////
texture<uchar4, 2, cudaReadModeElementType> render_result;
texture<uchar4, 2, cudaReadModeElementType> perturbed_result;

texture<uchar4, 3, cudaReadModeElementType> ground_truth;


//////////////////////////////////////////////////////////////////////////////////////////////////////
//
//    Bind CUDA arrays to texture references
//
//////////////////////////////////////////////////////////////////////////////////////////////////////

// bind cudaArray to texture reference
template<typename pixel_T, int dim>
void bind_tex(cudaArray* data_array, texture<pixel_T, dim, cudaReadModeElementType>& tex)
{
  // set texture parameters
  tex.normalized = false;                      // access with normalized texture coordinates
  tex.filterMode = cudaFilterModePoint;        // linear interpolation
  tex.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates
  if (dim>=2);
  tex.addressMode[1] = cudaAddressModeWrap;
  if (dim>=3)
    tex.addressMode[2] = cudaAddressModeWrap;
  if (dim>=4)
    tex.addressMode[3] = cudaAddressModeWrap;

  // channel descriptor
  cudaChannelFormatDesc channelDesc;
  cutilSafeCall( cudaGetChannelDesc(&channelDesc, data_array) );

  // bind array to 3D texture
  cutilSafeCall(cudaBindTextureToArray(tex, data_array, channelDesc));
}

// bind input array to render result tex ref
void bind_rrtex_cuda(cudaArray* data_array)
{
  bind_tex<uchar4,2>(data_array, render_result);
}

// bind input array to perturbed result tex ref
void bind_prtex_cuda(cudaArray* data_array)
{
  bind_tex<uchar4,2>(data_array, perturbed_result);
}

// bind input array to ground truth tex ref
void bind_gttex_cuda(cudaArray* data_array)
{
  bind_tex<uchar4,3>(data_array, ground_truth);
}


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
  //unsigned int z = blockIdx.x;
  //unsigned int y = blockIdx.y;
  //unsigned int x = threadIdx.x;
  // slice_pitch = pitch * height
  // slice = ptr + z * slice_pitch
  // row = (type*)(slice + y*pitch)
  // elem = *(row + x)

  int index = index3(threadIdx.x, blockIdx.y, blockIdx.x, blockDim.x);

  char * slice = (char *)vol_pptr.ptr + blockIdx.x * vol_pptr.pitch * blockDim.x;

  *((float*)(slice+blockIdx.y*vol_pptr.pitch) + threadIdx.x)
    = device_x[ tag_vol[index] ];
}
void construct_volume_cuda(
                           float * device_x,
                           cudaPitchedPtr * density_vol,
                           cudaExtent extent,
                           int *   tag_vol
                           )
{
  dim3 grid_dim(extent.depth, extent.height, 1);
  dim3 block_dim(extent.width/4, 1, 1);

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

  dim3 grid_dim(max_length, max_length, 1);
  dim3 block_dim(max_length, 1, 1);

  cudaExtent ext_low = make_cudaExtent(length, length, length);
  int scale = max_length / length;

  upsample_volume<<< grid_dim, block_dim >>>(
    *lower_lev,
    ext_low,
    *upper_lev,
    scale );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Calc F on the cuda
//
/////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void calc_f(
                       int i_view,           // which view, relates to ofset in proj_centers...
                       int n_view,           // num of different views/cameras
                       int n,                // num of items in array x, f, and g
                       int interval,         // the occupycation radius of projection
                       int * proj_centers,   // 
                       int * tag_vol,
                       float * f_array
                       )
{
  unsigned int i = threadIdx.x;
  unsigned int j = blockIdx.y;
  unsigned int k = blockIdx.x;

  int index_vol   = index3(i, j, k, blockDim.x);
  int index_array = tag_vol[index_vol];

  if (index_array != 0)
  {
    // pixel on the image
    int u = proj_centers[n_view * 2 * (index_array-1)];
    int v = proj_centers[n_view * 2 * (index_array-1)+1];

    float f = 0.0;
    for (int uu = u - interval; uu <= u + interval; ++u)
    {
      for (int vv = v - interval; vv <= v + interval; ++v)
      {
        uchar4 rr4 = tex2D(render_result, float(uu), float(vv));
        uchar4 gt4 = tex3D(ground_truth, float(uu), float(vv), float(i_view));
        // USE ONLY R CHANNEL HERE...
        f += (rr4.x-gt4.x)*(rr4.x-gt4.x)/(255.0*255.0);
      }
    }
    f_array[ index_array ] = f;
  } // if (index_array != 0)
}

extern "C"
void reduceSinglePass(int size, int threads, int blocks, float *d_idata, float *d_odata);
float calculate_f_cuda(
                       int    level, 
                       int    i_view, 
                       int    n_view,
                       int    n_nonzero_items,
                       int    interval,
                       int*   projected_centers, 
                       int*   vol_tag,
                       float* f_array,
                       float* sum_array
                       )
{
  int size = 1 << level;

  // calc f value for each non-zero cell
  dim3 grid_dim(size, size, 1);
  dim3 block_dim(size, 1, 1);
  calc_f<<< grid_dim, block_dim >>>(
    i_view,
    n_view,
    n_nonzero_items,
    interval,
    projected_centers,
    vol_tag,
    f_array );

  cutilSafeCall( cudaThreadSynchronize() );

  // copy to sum_array for sum
  reduceSinglePass(n_nonzero_items, 256,
    (n_nonzero_items/256)+((n_nonzero_items%256)?1:0), 
    f_array, sum_array);

  // copy and return result
  float result;
  cudaMemcpy(
    &result,
    sum_array,
    sizeof(float),
    cudaMemcpyDeviceToHost);
  return result;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Calc G[] on the cuda
//
/////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void calc_g()
{
}

void calculate_g_cuda(int level, int max_level, int* projected_centers, float* g_array)
{
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  set parameters on CUDA
//
/////////////////////////////////////////////////////////////////////////////////////////////////////
