#include "nvblox/core/color.h"
#include "nvblox/core/image.h"

namespace nvblox {
namespace tex {

static constexpr int THREADS_PER_BLOCK = 8;

class Eroder {
 public:
  void erodeMask(Image<float> image, int iterations = 10, int radius = 1);

 protected:
  void initialize(Image<float>);
  void erodeHole(Image<float> output, Image<float> input, int radius);
  void gaussianBlur(Image<float> output, Image<float> input, float sigma);

  Image<float> tmp_image_;
  cudaStream_t erosion_stream_;
};

}  // namespace tex
}  // namespace nvblox