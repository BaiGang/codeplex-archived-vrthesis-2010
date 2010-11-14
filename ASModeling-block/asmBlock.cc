
#include "asmBlock.h"

using namespace asmodeling_block;

bool Block_GPU::AssignBlock(Block & blk)
{
  glGenTextures(1, &tex_id_);
  //glBindTexture()


  return true;
}

Block_GPU::~Block_GPU()
{
  glDeleteTextures(1, &tex_id_);
}