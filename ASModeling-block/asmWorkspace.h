#ifndef __ASM_WORKSPACE_H_
#define __ASM_WORKSPACE_H_

#include <vector>
#include <map>

// 
#include <gl/glew.h>

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cuda_gl_interop.h>
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>

#include "asmTypes.h"
#include "asmOGL.h"
#include "asmBlock.h"
#include "asmConfigure.h"
#include "asmGEvaluator.h"

#include "../L-BFGS-B/ap.h"
#include "../L-BFGS-B/lbfgsb.h"
#include "../Utils/image/BMPImage.h"


namespace asmodeling_block
{


  class Workspace
  {
  public:
    //! Init the system
    bool Init(const char * conf_filename, const char * camera_filename);

    //! Process a frame -- the frame-th
    bool ProcessFrame(uint32_t frame);

  private:

    // methods
    //! Load captured images
    bool load_frame_images(uint32_t frame);

    //! init blocks and cells
    bool init_value_firstlevel(uint32_t level);
    bool init_value_upperlevel(uint32_t level);

    //! reconstruct density volume
    bool reconstruct(uint32_t level);

    //! store the volume data
    bool store_frame_result(uint32_t frame);

	//! render the volume straightforwdly
	void straight_render(int iview);
	
	//! render the perturbing group
	void perturb_render(int iview, int block_layer, int slice, int pu, int pv);
	
    // cuda methods,  in asmWorkspace.cu
    bool init_cuda_resources(void);		//! 
    void upload_captured_images(void);	//! upload each frame to 
	void zero_g();						//! memset g to 0

    //! temp data
    std::vector< Block >     blocks_cpu_;
    std::vector< Block_GPU > blocks_gpu_;

    //! block num per dimension
    const static uint32_t NBLOCK = 16;
    uint32_t block_length;

    //! store each layer's block id
    std::map<int, std::vector<int> > along_x_;
    std::map<int, std::vector<int> > along_y_;
    std::map<int, std::vector<int> > along_z_;

    //! captured images here, merging into one big block
    scoped_array<float> captured_image_;
	
	//! captured images, seperated files
	scoped_array<BMPImage> captured_BMPimages_;

    //! density data here
    ap::real_1d_array x_array_;

    //! lbfgsb data
    ap::integer_1d_array lbfgsb_nbd_;
    ap::real_1d_array    lbfgsb_l_;
    ap::real_1d_array    lbfgsb_u_;
    int lbfgsb_info_code_;

    //! Configure of the modeling system
    Configure conf_;

    // member data on device side

	//! gradients value
	float * device_g;
	uint32_t g_size;
	
	
    //! graphics resource for render result and perturbed result
    cudaGraphicsResource * resource_rr_;
    cudaGraphicsResource * resource_pr_;

    cudaArray * rr_tex_cudaArray;
    cudaArray * pr_tex_cudaArray;

    // 3d, z maps to different images
    cudaArray * gt_tex_cudaArray;
    cudaExtent  gt_cudaArray_extent;

  };

  // grad compute 
  extern Workspace * g_workspace;
  void GradCompute(const ap::real_1d_array& , ap::real_t& , ap::real_1d_array& );

} //namespace asmodeling_block


#endif //__ASM_WORKSPACE_H_