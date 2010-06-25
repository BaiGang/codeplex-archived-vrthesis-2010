
#include <cstdio>

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_math.h>

#include "devfun_def.h"

#include "utils.cuh"
#include "volume_construction.cu"
#include "f_calculation.cu"
#include "g_calculation.cu"

typedef unsigned short uint16;

void construct_volume_cuda (float * device_x,
                            cudaPitchedPtr * density_vol,
                            cudaExtent extent,
                            int *   tag_vol )
{
  dim3 grid_dim(extent.depth, extent.depth, 1);
  dim3 block_dim(extent.depth, 1, 1);

  fprintf(stderr, "Lauching Kernel \"construct_volume\", <<<(%d %d %d),(%d %d %d)>>>",
    extent.depth, extent.depth, 1, extent.depth, 1, 1);

  construct_volume<<< grid_dim, block_dim >>>(
    *density_vol,
    device_x,
    tag_vol
    );

  // check if kernel execution generated an error
  cutilCheckMsg("Kernel execution failed");
}

void construct_volume_linm_cuda (int length,
                                 float *devic_x,
                                 float *density_vol,
                                 int * tag_vol )
{
  dim3 grid_dim(length, length, 1);
  dim3 block_dim(length, 1, 1);

  construct_volume_linm<<<grid_dim, block_dim>>>(
    density_vol,
    devic_x,
    tag_vol );

  // check if kernel execution generated an error
  cutilCheckMsg("Kernel execution failed");

}

void upsample_volume_cuda (int level,
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

  fprintf(stderr, "<+> Lauching kernel \"upsampe_volume\",upper: %d, lower: %d, scale: %d\n",
    max_length, length, scale);

  upsample_volume<<< grid_dim, block_dim >>>(
    *lower_lev,
    ext_low,
    *upper_lev,
    scale );

  // check if kernel execution generated an error
  cutilCheckMsg("Kernel execution failed");
}

void construct_volume_from_previous_cuda (
  float * device_x,
  cudaPitchedPtr * density_vol,
  cudaExtent extent,
  int * tag_vol
  )
{
  dim3 grid_dim(extent.depth, extent.depth, 1);
  dim3 block_dim(extent.depth, 1, 1);

  construct_volume_from_prev<<<grid_dim, block_dim>>> (
    *density_vol,
    device_x,
    tag_vol
    );

  cutilCheckMsg("Kernel execution failed");
}

void cull_empty_cells_cuda (cudaPitchedPtr * density_vol,
                            cudaExtent extent,
                            int * tag_vol )
{
  dim3 grid_dim(extent.depth, extent.depth, 1);
  dim3 block_dim(extent.depth, 1 ,1);

  cull_empty_cells<<<grid_dim, block_dim>>> (
    *density_vol,
    tag_vol );

  cutilCheckMsg("Kernel execution failed");
}

void get_guess_x_cuda (float * guess_x,
                       cudaPitchedPtr * density_vol,
                       cudaExtent extent,
                       int * tag_vol )
{
  dim3 grid_dim(extent.depth, extent.depth, 1);
  dim3 block_dim(extent.depth, 1, 1);

  get_guess_x<<<grid_dim, block_dim>>> (
    *density_vol,
    tag_vol,
    guess_x );

  cutilCheckMsg("Kernel execution failed");
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

float calculate_f_cuda (int      level, 
                        int      i_view, 
                        int      n_view,
                        int      n_nonzero_items,
                        int      powtwo_length,
                        int      interval,
                        uint16*  projected_centers, 
                        int*     vol_tag,
                        float*   f_array,
                        float*   sum_array )
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

  // check if kernel execution generated an error
  cutilCheckMsg("Kernel execution failed");


  // copy to sum_array for sum
  reduceSinglePass(n_nonzero_items, 256,
    (n_nonzero_items/256)+((n_nonzero_items%256)?1:0), 
    f_array, sum_array);

  // check if kernel execution generated an error
  cutilCheckMsg("Kernel execution failed");

  cutilSafeCall( cudaThreadSynchronize() );

  // copy and return result
  float result = 0.0f;
  cutilSafeCall( cudaMemcpy(
    &result,
    sum_array,
    sizeof(float),
    cudaMemcpyDeviceToHost ));

  cutilSafeCall( cudaGetLastError() );

  return result;
}


void calculate_g_cuda( int      level, 
                       int      i_view, 
                       int      n_view,
                       int      n_nonzero_items,
                       int      interval,
                       uint16*  projected_centers, 
                       int*     vol_tag,
                       float*   g_array )
{
  int length = 1 << level;
  dim3 dim_grid(length, length, 1);
  dim3 dim_block(length, 1, 1);

  calc_g <<<dim_grid, dim_block>>> (
    i_view,
    n_view,
    n_nonzero_items,
    interval,
    projected_centers,
    vol_tag,
    g_array
    );

  // check if kernel execution generated an error
  cutilCheckMsg("Kernel execution failed");
  cutilSafeCall( cudaGetLastError() );
}


////////////////////////////////////////
// read pptr volume to linear memory
////////////////////////////////////////
void get_volume_cuda( int level,
                      cudaPitchedPtr vol_pptr,
                      //int * tag_vol,
                      float * den_vol )
{
  int length = 1 << level;
  dim3 dim_grid(length, length, 1);
  dim3 dim_block(length, 1, 1);

  get_volume<<<dim_grid, dim_block>>> (
    vol_pptr,
    //tag_vol,
    den_vol
    );
  // check if kernel execution generated an error
  cutilCheckMsg("Kernel execution failed");
  cutilSafeCall( cudaGetLastError() );

  fprintf(stderr, "..Finished kernel call of \"get_volume\"\n");

}

/////////////////////////////