#ifndef __CUDA_BRIDGE_H__
#define __CUDA_BRIDGE_H__

#include <cuda_runtime.h>

/////////////////////////////////////////////
//   Forward declarations
/////////////////////////////////////////////
void construct_volume_cuda (float * device_x,
                            cudaPitchedPtr * density_vol,
                            cudaExtent extent,
                            int *   tag_vol );

void upsample_volume_cuda (int level,
                           int max_level,
                           cudaPitchedPtr * lower_lev,
                           cudaPitchedPtr * upper_lev );

void construct_volume_from_previous_cuda
(float * device_x,
 cudaPitchedPtr* density_vol,
 cudaExtent extent,
 int * tag_vol
 );

void cull_empty_cells_cuda (cudaPitchedPtr* density_vol,
                            cudaExtent extent,
                            int * tag_vol );

void get_guess_x_cuda (float * guess_x,
                       cudaPitchedPtr * density_vol,
                       cudaExtent extent,
                       int * tag_vol );

void bind_rrtex_cuda (cudaArray*);
void bind_prtex_cuda (cudaArray*);
void bind_gttex_cuda (cudaArray*);
void unbind_rrtex_cuda();
void unbind_prtex_cuda();
void unbind_gt_tex_cuda();

void change_image_layout_cuda (unsigned char * raw_image,
                               cudaPitchedPtr * image_pptr,
                               cudaExtent * extent,
                               int width,
                               int height,
                               int iview );

float calculate_f_cuda (int    level, 
                        int    i_view, 
                        int    n_view,
                        int    n_nonzero_items,
                        int    powtwo_length,
                        int    interval,
                        int*   projected_centers, 
                        int*   vol_tag,
                        float* f_array,
                        float* sum_array );

void calculate_g_cuda (int    level, 
                       int    i_view, 
                       int    n_view,
                       int    n_nonzero_items,
                       int    interval,
                       int*   projected_centers, 
                       int*   vol_tag,
                       float* g_array );

static inline int nearest_pow2(int a)
{
  int k = 1;
  while (k < a)
  {
    k = k << 1;
  }
  return k;
}

#endif //__CUDA_BRIDGE_H__