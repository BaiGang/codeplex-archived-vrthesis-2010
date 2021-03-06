#ifndef __CUDA_BRIDGE_H__
#define __CUDA_BRIDGE_H__


#include <cuda_runtime.h>

typedef unsigned short uint16;

/////////////////////////////////////////////
//   Forward declarations
/////////////////////////////////////////////
void construct_volume_cuda (float * device_x,
                            cudaPitchedPtr * density_vol,
                            cudaExtent extent,
                            int *   tag_vol );

void construct_volume_linm_cuda (int length,
                                 float *devic_x,
                                 float *density_vol,
                                 int * tag_vol );

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

////////////////////////////////////////
// read pptr volume to linear memory
////////////////////////////////////////
void get_volume_cuda( int level,
                      cudaPitchedPtr vol_pptr,
                      //int * tag_vol,
                      float * den_vol );

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

void change_image_layout_cuda (unsigned char * raw_image,
                               cudaPitchedPtr * image_pptr,
                               cudaExtent * extent,
                               int width,
                               int height,
                               int iview );

float calculate_f_cuda (int     level, 
                        int     img_width,
                        int     img_height,
                        int     i_view, 
                        int     n_view,
                        int     n_nonzero_items,
                        int     powtwo_length,
                        int     interval,
                        uint16* projected_centers, 
                        int*    vol_tag,
                        float*  f_array,
                        float*  sum_array/*,
                        float * data*/);

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
                      float*   g_array );

static inline int nearest_pow2(int a)
{
  int k = 1;
  while (k < a)
  {
    k = k << 1;
  }
  return k;
}


////////////// 
//  for testing
void test__(int width, int height, int iview, float * h_data1, float * h_data2);
void tst_g(int width, int height, int iview, float * h_data);
void tst_pcenters(int level, int width, int height, int iview, int nview,
                  unsigned short * pcenters, int * tag_vol, float * h_data);
#endif //__CUDA_BRIDGE_H__