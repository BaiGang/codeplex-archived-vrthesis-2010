#include "asmWorkspace.h"

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_math.h>

using namespace asmodeling_block;

void Workspace::upload_captured_image(void)
{

}

bool Workspace::init_cuda_resources(void)
{
  cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );

  cudaMalloc3DArray(

  return true;
}