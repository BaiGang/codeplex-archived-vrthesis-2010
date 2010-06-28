
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

void construct_volume_cuda (float * device_x,
                            cudaPitchedPtr * density_vol,
                            cudaExtent extent,
                            int *   tag_vol )
{
  dim3 grid_dim(extent.depth, extent.depth, 1);
  dim3 block_dim(extent.depth, 1, 1);

  //fprintf(stderr, "Lauching Kernel \"construct_volume\", <<<(%d %d %d),(%d %d %d)>>>",
  //  extent.depth, extent.depth, 1, extent.depth, 1, 1);
  cutilSafeCall( cudaGetLastError() );
  construct_volume<<< grid_dim, block_dim >>>(
    *density_vol,
    device_x,
    tag_vol
    );
  cutilSafeCall( cudaGetLastError() );
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
  cutilSafeCall( cudaGetLastError() );
  construct_volume_linm<<<grid_dim, block_dim>>>(
    density_vol,
    devic_x,
    tag_vol );
  cutilSafeCall( cudaGetLastError() );
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

  //fprintf(stderr, "<+> Lauching kernel \"upsampe_volume\",upper: %d, lower: %d, scale: %d\n",
  //  max_length, length, scale);
  cutilSafeCall( cudaGetLastError() );
  upsample_volume<<< grid_dim, block_dim >>>(
    *lower_lev,
    ext_low,
    *upper_lev,
    scale );
  cutilSafeCall( cudaGetLastError() );
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
  cutilSafeCall( cudaGetLastError() );
  construct_volume_from_prev<<<grid_dim, block_dim>>> (
    *density_vol,
    device_x,
    tag_vol
    );
  cutilSafeCall( cudaGetLastError() );

  cutilCheckMsg("Kernel execution failed");
}

void cull_empty_cells_cuda (cudaPitchedPtr * density_vol,
                            cudaExtent extent,
                            int * tag_vol )
{
  dim3 grid_dim(extent.depth, extent.depth, 1);
  dim3 block_dim(extent.depth, 1 ,1);

  cutilSafeCall( cudaGetLastError() );
  cull_empty_cells<<<grid_dim, block_dim>>> (
    *density_vol,
    tag_vol );
  cutilSafeCall( cudaGetLastError() );
  cutilCheckMsg("Kernel execution failed");
}

void get_guess_x_cuda (float * guess_x,
                       cudaPitchedPtr * density_vol,
                       cudaExtent extent,
                       int * tag_vol )
{
  dim3 grid_dim(extent.depth, extent.depth, 1);
  dim3 block_dim(extent.depth, 1, 1);

  cutilSafeCall( cudaGetLastError() );
  get_guess_x<<<grid_dim, block_dim>>> (
    *density_vol,
    tag_vol,
    guess_x );

  cutilCheckMsg("Kernel execution failed");
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

float calculate_f_cuda (int      level, 
                        int      img_width,
                        int      img_height,
                        int      i_view, 
                        int      n_view,
                        int      n_nonzero_items,
                        int      powtwo_length,
                        int      interval,
                        uint16*  projected_centers, 
                        int*     vol_tag,
                        float*   f_array,
                        float*   sum_array/*,
                        float *  data*/)
{
  int size = 1 << level;

  // calc f value for each non-zero cell
  dim3 grid_dim(size, size, 1);
  dim3 block_dim(size, 1, 1);

  cutilSafeCall( cudaGetLastError() );

  calc_f<<< grid_dim, block_dim >>>(
    img_width,
    img_height,
    i_view,
    n_view,
    //n_nonzero_items,
    interval,
    projected_centers,
    vol_tag,
    f_array/*,
    data*/);

  // check if kernel execution generated an error
  cutilCheckMsg("Kernel execution failed");

  cutilSafeCall( cudaGetLastError() );

  // copy to sum_array for sum
  reduceSinglePass(n_nonzero_items, 256,
    (n_nonzero_items/256)+((n_nonzero_items%256)?1:0), 
    f_array, sum_array);

  // check if kernel execution generated an error
  cutilCheckMsg("Kernel execution failed");

  cutilSafeCall( cudaThreadSynchronize() );

  cutilSafeCall( cudaGetLastError() );
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
                      int      img_width,
                      int      img_height,
                      int      i_view, 
                      int      n_view,
                      //int      n_nonzero_items,
                      int      interval,
                      uint16*  projected_centers, 
                      int*     vol_tag,
                      float*   g_array )
{
  int length = 1 << level;
  dim3 dim_grid(length, length, 1);
  dim3 dim_block(length, 1, 1);

  cutilSafeCall( cudaGetLastError() );
  calc_g <<<dim_grid, dim_block>>> (
    img_width,
    img_height,
    i_view,
    n_view,
    //n_nonzero_items,
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

  cutilSafeCall( cudaGetLastError() );
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


////////
// TEST parts....


void test__(int width, int height, int iview, float * h_data1, float * h_data2)
{
  float * d_data1;  // render result
  float * d_data2;  // ground truth

  cutilSafeCall( cudaMalloc((void**)(&d_data1), width * height * sizeof(float) ) );
  cutilSafeCall( cudaMalloc((void**)(&d_data2), width * height * sizeof(float) ) );

  dim3 dim_grid(width/16 + ((width%16)?1:0), height/16+((height%16)?1:0), 1);
  dim3 dim_block(16, 16, 1);

  cutilSafeCall( cudaGetLastError() );
  test_kernel<<<dim_grid, dim_block>>>(width, height, iview, d_data1, d_data2);
  cutilSafeCall( cudaGetLastError() );

  cutilSafeCall( cudaMemcpy(h_data1, d_data1, width * height * sizeof(float),
    cudaMemcpyDeviceToHost) );

  cutilSafeCall( cudaMemcpy(h_data2, d_data2, width * height * sizeof(float),
    cudaMemcpyDeviceToHost) );

  cutilSafeCall( cudaFree(d_data1));
  cutilSafeCall( cudaFree(d_data2));

}

void tst_g(int width, int height, int iview, float * h_data)
{
  dim3 dim_grid(width/16 + ((width%16)?1:0), height/16+((height%16)?1:0), 1);
  dim3 dim_block(16, 16, 1);

  float * d_data;
  cutilSafeCall( cudaMalloc((void**)(&d_data), width* height * sizeof(float) ));

  cutilSafeCall( cudaGetLastError() );
  test_g_kernel<<<dim_grid, dim_block>>>
    (width, height, iview, d_data);
  cutilSafeCall( cudaGetLastError() );

  cutilSafeCall( cudaMemcpy(h_data, d_data, width*height*sizeof(float),
    cudaMemcpyDeviceToHost ) );

  cutilSafeCall( cudaFree((void*)d_data ) );
}

void tst_pcenters(int level, int width, int height, int iview, int nview,
                  uint16 * pcenters, int * tag_vol, float * h_data)
{
  int length = 1 << level;
  dim3 dim_grid(length, length, 1);
  dim3 dim_block(length, 1, 1);

  fprintf(stderr, "Testing pcenters.. \nwidth: %d\n height : %d\n iview: %d\n nviews: %d\n",
    width, height, iview, nview);

  float * d_data;
  cutilSafeCall( cudaMalloc((void**)(&d_data), width*height* sizeof(float) ));

  cutilSafeCall( cudaMemset((void**)(&d_data), 0, sizeof(float)*width*height) );

  cutilSafeCall( cudaGetLastError() );
  test_projected_centers<<<dim_grid, dim_block>>>
    (width, height, iview, nview,
    pcenters, tag_vol, d_data);
  cutilSafeCall( cudaGetLastError() );

  cutilSafeCall( cudaThreadSynchronize() );

  cutilSafeCall( cudaMemcpy(h_data, d_data, width*height* sizeof(float),
    cudaMemcpyDeviceToHost ) );

  cutilSafeCall( cudaFree((void*)d_data ) );
}
