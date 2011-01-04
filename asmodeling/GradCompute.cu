
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

void construct_volume_linm_cuda (
                                 int length,
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

  //// timer
  /////////////////////////////////////////////////////////////////////////////////////////////////////
  //clock_t * h_timer1 = new clock_t[2*size * size];
  //clock_t * d_timer1;
  //cutilSafeCall( cudaMalloc((void**)&d_timer1, 2*size*size*sizeof(clock_t)) );

  //clock_t * h_timer2 = new clock_t[2*size * size];
  //clock_t * d_timer2;
  //cutilSafeCall( cudaMalloc((void**)&d_timer2, 2*size*size*sizeof(clock_t)) );
  /////////////////////////////////////////////////////////////////////////////////////////////////////

  cutilSafeCall( cudaGetLastError() );
  calc_f<<< grid_dim, block_dim >>>(
    img_width,
    img_height,
    i_view,
    n_view,
    interval,
    projected_centers,
    vol_tag,
    f_array/*,
    d_timer1,
    d_timer2*/
    );
  cutilSafeCall( cudaGetLastError() );

  /////////////////////////////////////////////////////////////////////////////////////////////////////
  //cutilSafeCall(cudaMemcpy(h_timer1, d_timer1, 2*size * size *sizeof(clock_t), cudaMemcpyDeviceToHost));
  //clock_t min_begin = h_timer1[0];
  //clock_t max_end   = h_timer1[size*size];
  //for (int iitimer = 1; iitimer < size * size - 1; ++iitimer)
  //{
  //  if (min_begin > h_timer1[iitimer])
  //    min_begin = h_timer1[iitimer];
  //  if (max_end < h_timer1[size*size + iitimer])
  //    max_end = h_timer1[size*size + iitimer];
  //}
  //fprintf(stderr, "<TIMING> F calculation - fetching global memory - used %d clocks.\n", max_end - min_begin);
  //delete [] h_timer1;
  //cutilSafeCall( cudaFree(d_timer1) );

  //cutilSafeCall(cudaMemcpy(h_timer2, d_timer2, 2*size * size *sizeof(clock_t), cudaMemcpyDeviceToHost));
  //min_begin = h_timer2[0];
  //max_end   = h_timer2[size*size];
  //for (int iitimer = 1; iitimer < size * size - 1; ++iitimer)
  //{
  //  if (min_begin > h_timer2[iitimer])
  //    min_begin = h_timer2[iitimer];
  //  if (max_end < h_timer2[size*size + iitimer])
  //    max_end = h_timer2[size*size + iitimer];
  //}
  //fprintf(stderr, "<TIMING> F calculation - computation - used %d clocks.\n", max_end - min_begin);
  //delete [] h_timer1;
  //cutilSafeCall( cudaFree(d_timer1) );

  /////////////////////////////////////////////////////////////////////////////////////////////////////


  // copy to sum_array for sum
  reduceSinglePass(n_nonzero_items, 256,
    (n_nonzero_items/256)+((n_nonzero_items%256)?1:0), 
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


void calculate_g_cuda(int      level,
                      int      img_width,
                      int      img_height,
                      int      i_view, 
                      int      n_view,
                      int      interval,    //
                      int      pt_tilesize,
                      int      pt_u,
                      int      pt_v,
                      char     facing,
                      int      slice,
                      uint16*  projected_centers, 
                      int*     vol_tag,
                      float*   g_array )
{
  int length = 1 << level;

  int num_tiles = length / pt_tilesize;

  dim3 dim_grid(num_tiles, 1, 1);
  dim3 dim_block(num_tiles, 1, 1);

  //// timer
  /////////////////////////////////////////////////////////////////////////////////////////////////////
  //clock_t * h_timer = new clock_t[2*length];
  //clock_t * d_timer;
  //cutilSafeCall( cudaMalloc((void**)&d_timer, 2*length*sizeof(clock_t)) );
  /////////////////////////////////////////////////////////////////////////////////////////////////////

  cutilSafeCall( cudaGetLastError() );
  //
  // Lauch Kernels....
  //
  if (facing == 'X' || facing == 'x')
  {
    calc_g_x<<<dim_grid, dim_block>>> (
      img_width,
      img_height,
      i_view,
      n_view,
      interval,
      slice,
      pt_tilesize,
      pt_u,
      pt_v,
      projected_centers,
      vol_tag,
      g_array
      //d_timer
      );
  }
  else if (facing == 'Y' || facing == 'y')
  {
    calc_g_y<<<dim_grid, dim_block>>> (
      img_width,
      img_height,
      i_view,
      n_view,
      interval,
      slice,
      pt_tilesize,
      pt_u,
      pt_v,
      projected_centers,
      vol_tag,
      g_array
      //d_timer
      );
  }
  else // facing Z axis
  {
    calc_g_z<<<dim_grid, dim_block>>> (
      img_width,
      img_height,
      i_view,
      n_view,
      interval,
      slice,
      pt_tilesize,
      pt_u,
      pt_v,
      projected_centers,
      vol_tag,
      g_array
      //d_timer
      );
  }
  // check if kernel execution generated an error
  cutilSafeCall( cudaGetLastError() );


  /////////////////////////////////////////////////////////////////////////////////////////////////////
  //cutilSafeCall(cudaMemcpy(h_timer, d_timer, 2*length*sizeof(clock_t), cudaMemcpyDeviceToHost));
  //clock_t min_begin = h_timer[0];
  //clock_t max_end   = h_timer[length];
  //for (int iitimer = 1; iitimer < length - 1; ++iitimer)
  //{
  //  if (min_begin > h_timer[iitimer])
  //    min_begin = h_timer[iitimer];
  //  if (max_end < h_timer[length + iitimer])
  //    max_end = h_timer[length + iitimer];
  //}
  //fprintf(stderr, "<TIMING> G calculation used %d clocks.\n", max_end - min_begin);
  //delete [] h_timer;
  //cutilSafeCall( cudaFree(d_timer) );
  /////////////////////////////////////////////////////////////////////////////////////////////////////

}


////////////////////////////////////////
// read pptr volume to linear memory
////////////////////////////////////////
void get_volume_cuda( int level,
                     cudaPitchedPtr vol_pptr,
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

  cutilSafeCall( cudaMemcpy(h_data, d_data, width*height* sizeof(float),
    cudaMemcpyDeviceToHost ) );

  cutilSafeCall( cudaFree((void*)d_data ) );
}
