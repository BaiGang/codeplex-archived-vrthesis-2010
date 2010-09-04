#ifndef _ASMODELING_UTILS_CUH_
#define _ASMODELING_UTILS_CUH_

#include <cuda_runtime.h>
#include <cstdio>

typedef unsigned short uint16;

//////////////////////////////////////////////////////////////////////////////////////////////////////
//
//    GLOBAL VARIABLES
//
//////////////////////////////////////////////////////////////////////////////////////////////////////
texture<float4, 2, cudaReadModeElementType> render_result;
texture<float4, 2, cudaReadModeElementType> perturbed_result;
texture<uchar4, 3, cudaReadModeNormalizedFloat> ground_truth;
//texture<uchar4, 3, cudaReadModeElementType> ground_truth;

texture<int, 3, cudaReadModeElementType> position_tag;

texture<ushort2, 3, cudaReadModeElementType> pcenters;

__constant__ float disturb_value = 0.00001;

//////////////////////////////////////////////////////////////////////////////////////////////////////
//
//    Bind CUDA arrays to texture references
//
//////////////////////////////////////////////////////////////////////////////////////////////////////

// bind cudaArray to texture reference
template<typename pixel_T, int dim, cudaTextureReadMode rMode>
void bind_tex(cudaArray* data_array, texture<pixel_T, dim, rMode>& tex)
{
  // set texture parameters
  tex.normalized = 0;                          // access with [0,N) coordinates
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

  //frpintf(stderr, "\n\n\n\t\t\t\t TExture heRE....\n\n");
  //fprintf(stderr, "normalized : %d \n  x y z w : %d %d %d %d\n", tex.normalized,
  //  channelDesc.x, channelDesc.y, channelDesc.z, channelDesc.w);

  // bind array to 3D texture
  cutilSafeCall(cudaBindTextureToArray(tex, data_array, channelDesc));
}

// bind input array to projected center tex ref
void bind_pcenters_cuda(cudaArray* data_array)
{
  bind_tex<ushort2, 3, cudaReadModeElementType>(data_array, pcenters);
}

// bind input array to position tag volume tex ref
void bind_postags_cuda(cudaArray* data_array)
{
  bind_tex<int, 3, cudaReadModeElementType>(data_array, position_tag);
}

// bind input array to render result tex ref
void bind_rrtex_cuda(cudaArray* data_array)
{
  bind_tex<float4,2, cudaReadModeElementType>(data_array, render_result);
}

// bind input array to perturbed result tex ref
void bind_prtex_cuda(cudaArray* data_array)
{
  bind_tex<float4,2, cudaReadModeElementType>(data_array, perturbed_result);
}

// bind input array to ground truth tex ref
void bind_gttex_cuda(cudaArray* data_array)
{
  bind_tex<uchar4,3, cudaReadModeNormalizedFloat>(data_array, ground_truth);
  //bind_tex<uchar4,3, cudaReadModeElementType>(data_array, ground_truth);
}

void unbind_rrtex_cuda()
{
  cutilSafeCall( cudaUnbindTexture(&render_result));
}
void unbind_prtex_cuda()
{
  cutilSafeCall( cudaUnbindTexture(&perturbed_result));
}
void unbind_gt_tex_cuda()
{
  cutilSafeCall( cudaUnbindTexture(&ground_truth));
}

void unbind_pcenter_tex_cuda()
{
  cudaUnbindTexture(&pcenters);
}
void unbind_postag_tex_cuda()
{
  cudaUnbindTexture(&position_tag);
}


//__host__ __device__ float __float_max(float a, float b)
//{
//  return (a > b)? a : b;
//}
//
//__host__ __device__  float __float_min(float a, float b)
//{
//  return (a < b)? a: b;
//}

__device__ inline uint float_to_uint8(float value)
{
    //return min(max(__float2int_rn((255 * value + 0.5f) / (1.0f + 1.0f/255.0f)), 0), 255);
  return __min(__max(0, 255 * value + 0.5), 255);
}

__device__ inline float uint8_to_float(unsigned char value)
{
    //return __saturatef(__uint2float_rn(value) / 255.0f);
  return min(max(value / 255.0f, 0.0f), 1.0f);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////
//
//    Util function to calculate vectorized position of a volume cell
//
//////////////////////////////////////////////////////////////////////////////////////////////////////
// map (i,j,k) to 1D array index
__device__ int index3(int i, int j, int k, int length)
{
  //return i + length * ( j + k * length );
  return i + __umul24(length, j + __umul24(k, length));
}

#endif //_ASMODELING_UTILS_CUH_