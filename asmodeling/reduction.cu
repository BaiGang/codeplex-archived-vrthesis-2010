#include <cuda_runtime.h>

#include "reduction_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
extern "C"
void reduce(int size, int threads, int blocks, float *d_idata, float *d_odata)
{
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

  // choose which of the optimized versions of reduction to launch
  if (isPow2(size))
  {
    switch (threads)
    {
    case 512:
      reduceMultiPass<512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 256:
      reduceMultiPass<256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 128:
      reduceMultiPass<128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 64:
      reduceMultiPass< 64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 32:
      reduceMultiPass< 32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 16:
      reduceMultiPass< 16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  8:
      reduceMultiPass<  8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  4:
      reduceMultiPass<  4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  2:
      reduceMultiPass<  2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  1:
      reduceMultiPass<  1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    }
  }
  else
  {
    switch (threads)
    {
    case 512:
      reduceMultiPass<512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 256:
      reduceMultiPass<256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 128:
      reduceMultiPass<128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 64:
      reduceMultiPass< 64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 32:
      reduceMultiPass< 32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 16:
      reduceMultiPass< 16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  8:
      reduceMultiPass<  8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  4:
      reduceMultiPass<  4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  2:
      reduceMultiPass<  2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  1:
      reduceMultiPass<  1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    }
  }
}

extern "C"
void reduceSinglePass(int size, int threads, int blocks, float *d_idata, float *d_odata)
{
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  int smemSize = threads * sizeof(float);

  // choose which of the optimized versions of reduction to launch
  if (isPow2(size))
  {
    switch (threads)
    {
    case 512:
      reduceSinglePass<512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 256:
      reduceSinglePass<256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 128:
      reduceSinglePass<128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 64:
      reduceSinglePass< 64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 32:
      reduceSinglePass< 32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 16:
      reduceSinglePass< 16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  8:
      reduceSinglePass<  8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  4:
      reduceSinglePass<  4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  2:
      reduceSinglePass<  2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  1:
      reduceSinglePass<  1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    }
  }
  else
  {
    switch (threads)
    {
    case 512:
      reduceSinglePass<512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 256:
      reduceSinglePass<256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 128:
      reduceSinglePass<128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 64:
      reduceSinglePass< 64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 32:
      reduceSinglePass< 32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 16:
      reduceSinglePass< 16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  8:
      reduceSinglePass<  8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  4:
      reduceSinglePass<  4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  2:
      reduceSinglePass<  2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  1:
      reduceSinglePass<  1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    }
  }
}