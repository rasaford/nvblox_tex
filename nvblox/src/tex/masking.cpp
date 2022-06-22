#include "nvblox/tex/masking.h"

namespace nvblox {
namespace tex {

void Eroder::initialize(Image<float> image) {
  if (tmp_image_.width() != image.width() ||
      tmp_image_.height() != image.height()) {
    tmp_image_ = Image<float>(image);
  }
}

void Eroder::erodeMask(Image<float> image, int iterations, int radius) {
  initialize(image);

  int even_iter = (iterations >> 1) << 1;
  int radius = 1;

  for (int i = 0; i < even_iter; i++) {
    if (i & 1) {
      erodeHole(image, tmp_image_, radius);
    } else {
      erodeHole(tmp_image_, image, radius);
    }
  }
}

}  // namespace tex
}  // namespace nvblox