#ifndef __ASM_WORKSPACE_H_
#define __ASM_WORKSPACE_H_

#include <vector>
#include <map>

#include "../L-BFGS-B/apvt.h"

#include "asmTypes.h"
#include "asmOGL.h"
#include "asmBlock.h"
#include "asmConfigure.h"
#include "asmGEvaluator.h"

#include "../L-BFGS-B/ap.h"
#include "../L-BFGS-B/lbfgsb.h"


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
    bool init_value(void);

    //! reconstruct density volume
    bool reconstruct(void);

    //! store the volume data
    bool store_frame_result(uint32_t frame);

    //! temp data
    std::vector< Block >     blocks_cpu_;
    std::vector< Block_GPU > blocks_gpu_;

    //! store each layer's block id
    std::map<int, std::vector<int> > along_x_;
    std::map<int, std::vector<int> > along_y_;
    std::map<int, std::vector<int> > along_z_;

    //! captured images here, merging into one big block
    scoped_array<float> captured_image_;

    //! density data here
    ap::real_1d_array x_array_;

    //! Configure of the modeling system
    Configure conf_;
  };

  // grad compute 
  extern Workspace * g_workspace;
  void GradCompute(const ap::real_1d_array& , ap::real_t& , ap::real_1d_array& );

} //namespace asmodeling_block


#endif //__ASM_WORKSPACE_H_