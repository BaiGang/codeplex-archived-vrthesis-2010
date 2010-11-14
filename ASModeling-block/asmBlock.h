#ifndef __ASM_BLOCK_H_
#define __ASM_BLOCK_H_

#include <gl/glew.h>

#include "asmTypes.h"


namespace asmodeling_block
{

  //! An abstraction of sub volume of the field
  class Block
  {
  public:

    //! for conveniently access to member data
    friend class Block_GPU;

    //! position of the current block in the whole volume
    int i_block_;
    int j_block_;
    int k_block_;

    //! size of the current block (in terms of voxels)
    int length_;

    //! length of 
    float width_;
  };

  class Block_GPU
  {
  public:
    //! render the slice-th slice which is perpendicular to the orientation orn.
    void Render(int slice, Orientation orn);

    //! render perturbed slice
    void RenderPerturbed(int slice, Orientation orn, int pu, int pv, int tile_size);

    //! assign a cpu block to gpu block
    bool AssignBlock(Block & blk);

    Block_GPU();
    ~Block_GPU();

  private:

    //! 3D volume tex
    GLuint tex_id_;

    //! near-lower-left (original) corner of the box
    float nll_x_;
    float nll_y_;
    float nll_z_;

    //! length of the volume border (in world space)
    float width_;

    //! length (in tex space)
    float length_;
  };

} // namespace asmodeling_block

#endif //__ASM_BLOCK_H_