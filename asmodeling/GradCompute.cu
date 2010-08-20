
#include <cstdio>

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_math.h>

#include "devfun_def.h"

#include "utils.cuh"
#include "volume_construction.cu"
#include "f_calculation.cu"
#include "g_calculation.cu"

// for testing
#include "Test.cu"

void construct_volume_cuda (
                            float * device_x,
                            cudaPitchedPtr * density_vol,
                            cudaExtent extent
                            )
{
  dim3 grid_dim(extent.depth, extent.depth, 1);
  dim3 block_dim(extent.depth, 1, 1);

  //fprintf(stderr, "Lauching Kernel \"construct_volume\", <<<(%d %d %d),(%d %d %d)>>>",
  //  extent.depth, extent.depth, 1, extent.depth, 1, 1);
  cutilSafeCall( cudaGetLastError() );
  construct_volume<<< grid_dim, block_dim >>>(
    *density_vol,
    device_x
    );
  cutilSafeCall( cudaGetLastError() );
  // check if kernel execution generated an error
  cutilCheckMsg("Kernel execution failed");
}

void construct_volume_linm_cuda (
                                 int length,
                                 float *devic_x,
                                 float *density_vol
                                 )
{
  dim3 grid_dim(length, length, 1);
  dim3 block_dim(length, 1, 1);
  cutilSafeCall( cudaGetLastError() );
  construct_volume_linm<<<grid_dim, block_dim>>>(
    density_vol,
    devic_x );
  cutilSafeCall( cudaGetLastError() );
}

void upsample_volume_cuda (
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

  //fprintf(stderr, "<+> Lauching kernel \"upsampe_volume\",upper: %d, lower: %d, scale: %d\n",
  //  max_length, length, scale);
  cutilSafeCall( cudaGetLastError() );
  upsample_volume<<< grid_dim, block_dim >>>(
    *lower_lev,
    ext_low,
    *upper_lev,
    scale );
  cutilSafeCall( cudaGetLastError() );

}

void construct_volume_from_previous_cuda(
  float * device_x,
  cudaPitchedPtr * density_vol,
  cudaExtent extent
  )
{
  dim3 grid_dim(extent.depth, extent.depth, 1);
  dim3 block_dim(extent.depth, 1, 1);
  cutilSafeCall( cudaGetLastError() );
  construct_volume_from_prev<<<grid_dim, block_dim>>> (
    *density_vol,
    device_x
    );
  cutilSafeCall( cudaGetLastError() );
}

void cull_empty_cells_cuda (
                            cudaPitchedPtr * density_vol,
                            cudaExtent extent
                            )
{
  dim3 grid_dim(extent.depth, extent.depth, 1);
  dim3 block_dim(extent.depth, 1 ,1);

  cutilSafeCall( cudaGetLastError() );
  cull_empty_cells<<<grid_dim, block_dim>>> (
    *density_vol );
  cutilSafeCall( cudaGetLastError() );
}

void get_guess_x_cuda (
                       float * guess_x,
                       cudaPitchedPtr * density_vol,
                       cudaExtent extent
                       )
{
  dim3 grid_dim(extent.depth, extent.depth, 1);
  dim3 block_dim(extent.depth, 1, 1);

  cutilSafeCall( cudaGetLastError() );
  get_guess_x<<<grid_dim, block_dim>>> (
    *density_vol,
    guess_x );
  cutilSafeCall( cudaGetLastError() );
}

extern "C"
void reduceSinglePass (int size,
                       int threads,
                       int blocks,
                       float *d_idata,
                       float *d_odata );

extern "C"
void reduce (int size,
             int threads,
             int blocks,
             float *d_idata,
             float *d_odata );

float calculate_f_compact_cuda(
                               int i_view,
                               int img_height,
                               int range,
                               int n_items,       // num of non-zero voxels plus one zero indicator
                               float *f_array,
                               float *sum_array
                               )
{
  dim3 grid_dim( (n_items/256)+((n_items%256)?1:0) , 1, 1);
  dim3 block_dim(256, 1, 1);

  calc_f_compact<<<grid_dim, block_dim>>>(
    i_view,
    img_height,
    range,
    f_array
    );
  cutilSafeCall(cudaGetLastError());

  cudaMemset( f_array, 0, sizeof(float) );

  // parallel reduction
  reduceSinglePass(n_items, 256,
    (n_items/256)+((n_items%256)?1:0), 
    f_array, sum_array);

  // copy and return result
  float result = 0.0f;
  cutilSafeCall( cudaMemcpy(
    &result,
    sum_array,
    sizeof(float),
    cudaMemcpyDeviceToHost ));

  return result;

}



void calculate_g_cuda(
                      int      level,
                      int      img_height,
                      int      i_view, 
                      int      interval,    //
                      int      pt_tilesize,
                      int      pt_u,
                      int      pt_v,
                      char     facing,
                      int      slice,
                      float*   g_array
                      )
{
  int length = 1 << level;

  int num_tiles = length / pt_tilesize;

  dim3 dim_grid(num_tiles, 1, 1);
  dim3 dim_block(num_tiles, 1, 1);

  //cutilSafeCall( cudaGetLastError() );
  //
  // Lauch Kernels....
  //
  if (facing == 'X' || facing == 'x')
  {
    calc_g_x<<<dim_grid, dim_block>>> (
      img_height,
      i_view,
      interval,
      slice,
      pt_tilesize,
      pt_u,
      pt_v,
      g_array
      );
  }
  else if (facing == 'Y' || facing == 'y')
  {
    calc_g_y<<<dim_grid, dim_block>>> (
      img_height,
      i_view,
      interval,
      slice,
      pt_tilesize,
      pt_u,
      pt_v,
      g_array
      );
  }
  else // facing Z axis
  {
    calc_g_z<<<dim_grid, dim_block>>> (
      img_height,
      i_view,
      interval,
      slice,
      pt_tilesize,
      pt_u,
      pt_v,
      g_array
      );
  }

  cutilSafeCall( cudaGetLastError() );

}


////////////////////////////////////////
// read pptr volume to linear memory
////////////////////////////////////////
void get_volume_cuda(
                     int level,
                     cudaPitchedPtr vol_pptr,
                     float * den_vol
                     )
{
  int length = 1 << level;
  dim3 dim_grid(length, length, 1);
  dim3 dim_block(length, 1, 1);

  cutilSafeCall( cudaGetLastError() );
  get_volume<<<dim_grid, dim_block>>> (
    vol_pptr,
    den_vol
    );
  cutilSafeCall( cudaGetLastError() );

  fprintf(stderr, "..Finished kernel call of \"get_volume\"\n");
}

/////////////////////////////
//Testing

void test_rrprgt(
                 int i_view,
                 int img_width,
                 int img_height,
                 float * rr,
                 float * pr,
                 float * gt
                 )
{
  //float * d_rr;
  //float * d_pr;
  //float * d_gt;

  //cutilSafeCall(  );
  //cutilSafeCall();
  //cutilSafeCall();

  //cutilSafeCall();
  //cutilSafeCall();
  //cutilSafeCall();
}

////////

