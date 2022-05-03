#include "nvblox/core/voxels.h"

namespace nvblox {

// constexpr in template classes also need to be defined
template <typename ElementType, int PatchWidth>
constexpr int TexVoxelTemplate<ElementType, PatchWidth>::kPatchWidth;

// Definitions for template class TexVoxelTemplate
template class TexVoxelTemplate<Color, 4>;

}  // namespace nvblox