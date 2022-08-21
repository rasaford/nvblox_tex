# Developer's Guide

This guide briefly covers the implementation details of nvblox_tex from a developer's perspective. Its aim is to aid in extending our approach.
Since our approach is an extension of nvblox, which is a part of this codebase and still fully functional in its original form, we describe both functionality of nvblox and nvblox_tex in the following paragraphs.

- [Developer's Guide](#developers-guide)
  - [Executable Binaries](#executable-binaries)
  - [nvblox_lib](#nvblox_lib)
  - [Dataset Loaders](#dataset-loaders)
  - [Core](#core)
    - [Voxels](#voxels)
    - [Voxel Block](#voxel-block)
    - [Voxel Block Layer](#voxel-block-layer)
    - [Common Names](#common-names)
  - [Integrators](#integrators)
    - [Projective TSDF Integrator](#projective-tsdf-integrator)
    - [Projective Color Integrator](#projective-color-integrator)
    - [Projective Tex Integrator](#projective-tex-integrator)
    - [ESDF Integrator](#esdf-integrator)
    - [Mesh Integrator](#mesh-integrator)
  - [Texture Packing](#texture-packing)
  - [Unit Tests](#unit-tests)


## Executable Binaries

Executable binaries are built from the experiments defined in `nvblox/experiments/src/fuse_3dmatch.cpp` and `nvblox/experiments/src/tex_integration.cpp` 
which define the entry-points for reconstruction using nvblox and nvblox_tex respectively. 
Specifically, the class `Fuse3DMatch` performs reconstruction using nvblox and `TexFuse3DMatch` performs it using our extension nvblox_tex.

Both share many of the same underlying functions in `nvblox/experiments/src/common/fuse_3dmatch.cpp` which performs the actual function calls for integration. 

The following two functions are of particular interest, since they call the individual Integrators described in the next section. 

```cpp
bool Fuse3DMatch::integrateFrame(const int frame_number);
```

```cpp
bool TexFuse3DMatch::integrateFrame(const int frame_number);
```

Their binaries are build in the `nvblox/build/experiments` directory when using cmake as described in the installation section above.


## nvblox_lib
This is the main shared library with functionality of both nvblox and nvblox_tex. 
For an example of its usage see the above section on Executable Binaries.

Unless explicitly stated otherwise, all functionality described below is relevant for both nvblox and nvblox_tex. 

## Dataset Loaders

`nvblox/include/nvblox/datasets`

Here, the dataset loaders are defined for loading `.png` color and depth images as well as the pose files in the 3dmatch dataset format.

Individual images are loaded with functions in `nvblox/include/nvblox/datasets/image_loader.h`. The entire dataset is parsed with `nvblox/include/nvblox/datasets/parse_3dmatch.h`. 
Both nvblox and nvblox_tex use the same functionality from these headers.

Of note is that the image loaders in `parse_3dmatch.h` have both a single- and multi-threaded implementation. 
We prefer the multithreaded version for the fastest reconstruction speeds.

## Core

`nvblox/include/nvblox/core`

This folder contains core functionality for both nvblox and nvblox_tex most common types are defined here such as the implementations for voxels.


### Voxels

`nvblox/include/nvblox/core/voxels.h`

This header defines all different voxel types used throughout nvblox and nvblox_tex. Specifically, we use the following voxel types:

```cpp
struct TsdfVoxel;

struct EsdfVoxel;

struct ColorVoxel;

template <typename _ElementType, int _PatchWidth>
class TexVoxelTemplate;

struct FreespaceVoxel;
```

When adding a new voxel type here, care must be taken that each function on it is callable from CUDA using the `__host__ __device__` annotations. 
Also, since each of the above structs is individually stored in GPU memory the number of bytes used by each struct 
should be as small as possible while taking care to align memory access of the individual fields.


### Voxel Block

`nvblox/include/nvblox/core/blox.h`

Individual voxels are grouped by the `VoxelBlock` class into blocks of constant size (currently `8x8x8`) since such a grouping fits
 well with the thread block model of the GPU. Conceptually, each block is processed independently and in arbitrary order.

`VoxelBlock` is a template class whose specializations are defined in `common_names.h`.


### Voxel Block Layer

`nvblox/include/nvblox/core/layer.h`

`VoxelBlock`s are collected by a `VoxelBlockLayer` class of for every voxel type. Each layer manages all allocated voxels of a given type. 
This class a templated and independent of the concrete type used.  Specializations are defined in `common_names.h`


### Common Names

`nvblox/include/nvblox/core/common_names.h`

As described in the previous sections this file provides `using` type aliases for the different `VoxelBlock` and `VoxelBlockLayer` types used throughout nvblox_lib.


## Integrators

`nvblox/include/nvblox/integrators`

The main functionality of this library is provided by the individual integrators listed below, which take in different images of the 
dataset and integrate them into the different representations in nvblox_tex, i.e. TSDF Voxels, Color Voxels, Tex Voxels, ...


### Projective TSDF Integrator
`nvblox/include/nvblox/integrators/projective_tsdf_integrator.h`

This integrator performs per-frame integration of depth images with associated pose estimates into the TSDF Voxel grid.
The key function for integration of a new depth frame is:

```cpp
void ProjectiveTsdfIntegrator::integrateFrame;
```

It dynamically allocates all required GPU and system memory and performs the integration into the TSDF Grid on the GPU.

This functionality is used by both nvblox and nvblox_tex.


### Projective Color Integrator
`nvblox/include/nvblox/integrators/projective_color_integrator.h`

*Requires TSDF Integration to be performed beforehand*

This integrator integrates a color image into the Color Voxel grid by back-projecting each voxel into the new color image. 
The single color value at this voxel is determined by linearly interpolating the projected voxel location in the color image.

```cpp
ProjectiveColorIntegrator::integrateFrame
```

This function takes in the color image, pose and TSDF layer and performs the integration into the Color Voxel field on the GPU. 


### Projective Tex Integrator
`nvblox/include/nvblox/integrators/projective_tex_integrator.h`

*Requires TSDF Integration to be performed beforehand*

This integrator contains the majority of the additions of nvblox_tex.
It performs integration of a color frame into the `TexVoxelLayer` by back-projecting each Texel and interpolating the color at that point.

Based on the TSDF surface integration, it integrates a new frame using the following function:

```cpp
void ProjectiveTexIntegrator::integrateFrame;
```

This function performs the following steps in-order:

**Block Allocation**

```cpp
void allocateBlocksWhereRequired;
```

Based on the current field of view of the camera new blocks might need to be allocated. This is done by dynamically allocating the required memory on the GPU.

**Neighbor Updates**

```cpp
void ProjectiveTexIntegrator::updateNeighborIndicies;
```

Since different voxel blocks do not natively contain adjacency information 
among themselves, this function updates a 2D adjacency index for each allocated block.

**Normal Directions**

```cpp
void ProjectiveTexIntegrator::updateVoxelNormalDirections;
```

This function computes the normal for each Voxel on the TSDF grid and quantizes them into discrete `TexVoxel` directions which are then written to each voxel. 
The computation is done on the GPU.

The relevant function call starting the Kernel is in `/nvblox/src/tex/tex_integrator_kernels.cu`:

```cpp
void nvblox::tex::updateTexVoxelDirectionsGPU(
    device_vector<const TsdfBlock*> neighbor_blocks,
    const device_vector<Index3D> block_indices,
    device_vector<TexBlock*>& tex_block_ptrs, const int num_blocks,
    const cudaStream_t stream, const float block_size, const float voxel_size);
```

For more details on how this computation is performed, see the project report.

**Depth Image rendering**

For fast querying of the distance from the surface to the current camera position, a depth image is rendered at 
low resolution. Each pixel value in it is the distance in meters to the camera. 
Rendering is done on the GPU and the resulting image is also stored in GPU memory. 

The relevant header is `/nvblox/include/nvblox/ray_tracing/sphere_tracer.h` and rendering is performed by:

```cpp
std::shared_ptr<const DepthImage> renderImageOnGPU(
      const Camera& camera, const Transform& T_S_C, const TsdfLayer& tsdf_layer,
      const float truncation_distance_m,
      const MemoryType output_image_memory_type = MemoryType::kDevice,
      const int ray_subsampling_factor = 1);
```

**Surface Distance Estimation**

For each Texel that is colored by an input image, we orthogonally project it the the TSDF surface and 
then project back to the image plane. More details on this projection are given in the report. 

To perform the projection, we define the following function in `/nvblox/src/integrators/cuda/projective_tex_integrator.cu`:

```cpp
__device__ bool nvblox::raycastToSurface(const TsdfBlock** blocks,
                                 const float voxel_size,
                                 const Vector3f& position,
                                 const TexVoxel::Dir& direction,
                                 float* distance) {
```
Internally, this function performs a neighbor lookup of adjacent voxels to the given Texel position and computes the 
TSDF value at that position by linear interpolation. The estimated distance to the surface along the given direction 
is written into the `distance` function argument. 

This function is only callable from GPU kernel code as it reads the TSDF Voxel values, which are stored in GPU memory. 

**Texel Coloring**

After Surface Estimation a measured color is computed for each Texel by projecting its estimated surface location on the 
image plane and linearly interpolating the color at that image location. For each measurement a weight is given by the following function:

```cpp
__device__ float computeMeasurementWeight(const TexVoxel* tex_voxel,
                                          const Transform& T_C_L,
                                          const Vector3f& voxel_center,
                                          const Vector2f& u_px,
                                          const float u_px_depth);
```
Details on the definition of this weight are given in the report.

From this measurement weight, measured color and previous Voxel color and weight, the updated color is computed by weighted averaging of the two colors in:  

```cpp
__device__ inline void updateTexel(const Color& color_measured,
                                   TexVoxel* tex_voxel,
                                   const Index2D& texel_idx,
                                   const float measurement_weight,
                                   const float max_weight)
```


### ESDF Integrator

`nvblox/include/nvblox/integrators/esdf_integrator.h`

The ESDF integrator creates an ESDF grid from the TSDF grid. This is mostly used for free space estimation in planning applications. 

nublox…tex does not change this functionality,


### Mesh Integrator


## Texture Packing 


## Unit Tests
`nvblox/tests`
