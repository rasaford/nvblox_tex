#include "nvblox/core/voxels.h"

namespace nvblox {

// constexpr in template classes also need to be defined
template <typename ElementType, int PatchWidth>
constexpr int TexVoxelTemplate<ElementType, PatchWidth>::kPatchWidth;

// Definitions for template class TexVoxelTemplate of various texture sizes.
// Using these, we only need to specify the actually used size once in voxels.h
template class TexVoxelTemplate<Color, 2>;
template class TexVoxelTemplate<Color, 4>;
template class TexVoxelTemplate<Color, 8>;
template class TexVoxelTemplate<Color, 16>;
template class TexVoxelTemplate<Color, 32>;
template class TexVoxelTemplate<Color, 64>;

}  // namespace nvblox