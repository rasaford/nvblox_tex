#include "nvblox/tex/masking.h"

namespace nvblox {
namespace tex {

// Adaped from TextureFusion
__global__ void erodeHoleKernel(float* output, float* input, int radius,
                                int img_width, int img_height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= img_width || y >= img_height) return;

  output[y * img_width + x] = 0.f;

  float sum = 0.f;

  for (int m = x - radius; m <= x + radius; m++) {
    for (int n = y - radius; n <= y + radius; n++) {
      if (m >= 0 && n >= 0 && m < img_width && n < img_height) {
        const float currentValue = input[n * img_width + m];
        sum += currentValue;
      }
    }
  }

  if (sum >= 0.95f) output[y * img_width + x] = 1.f;
}

void Eroder::erodeHole(Image<float> output, Image<float> input, int radius) {
  const int width = input.width();
  const int height = input.height();

  const dim3 grid_size((width + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                       (height + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  const dim3 block_size(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

  erodeHoleKernel<<<grid_size, block_size, 0, erosion_stream_>>>(
      output.dataPtr(), input.dataPtr(), radius, width, height);

  checkCudaErrors(cudaStreamSynchronize(erosion_stream_));
  checkCudaErrors(cudaPeekAtLastError());
}

__device__ inline float gaussian(const float sigma, const int x, const int y) {
  return expf(-((x * x + y * y) / (2.f * sigma * sigma)));
}

__global__ void gaussBlurKernel(float* output, float* input, float sigmaD,
                                int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  const int kernelRadius = (int)ceil(2.0 * sigmaD);

  output[y * width + x] = 0;

  float sum = 0.0f;
  float sumWeight = 0.0f;

  const float valueCenter = input[y * width + x];
  output[y * width + x] = valueCenter;

  for (int m = x - kernelRadius; m <= x + kernelRadius; m++) {
    for (int n = y - kernelRadius; n <= y + kernelRadius; n++) {
      if (m >= 0 && n >= 0 && m < width && n < height) {
        const float currentValue = input[n * width + m];

        const float weight = gaussian(sigmaD, m - x, n - y);
        sumWeight += weight;
        sum += weight * currentValue;
      }
    }
  }

  if (sumWeight > 0.0f) {
    output[y * width + x] = sum / sumWeight;
  }
}

void Eroder::gaussianBlur(Image<float> output, Image<float> input,
                          float sigma) {
  const int width = input.width();
  const int height = input.height();

  const dim3 grid_size((width + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                       (height + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  const dim3 block_size(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

  gaussBlurKernel<<<grid_size, block_size, 0, erosion_stream_>>>(
      output.dataPtr(), input.dataPtr(), sigma, width, height);

  checkCudaErrors(cudaStreamSynchronize(erosion_stream_));
  checkCudaErrors(cudaPeekAtLastError());
}

}  // namespace tex
}  // namespace nvblox