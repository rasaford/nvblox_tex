#include "nvblox/core/voxels.h"

namespace nvblox {

// constexpr in template classes also need to be defined
template <typename ElementType, int PatchWidth>
constexpr int TexVoxelTemplate<ElementType, PatchWidth>::kPatchWidth;

// Definitions for template class TexVoxelTemplate of various texture sizes.
// Using these, we only need to specify the actually used size once in voxels.h
#ifdef TEXEL_SIZE
template class TexVoxelTemplate<Color, TEXEL_SIZE>;
#else 
template class TexVoxelTemplate<Color, 8>;
#endif

}  // namespace nvblox