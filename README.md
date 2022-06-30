# nvblox_tex
*Texel based mapping on the GPU!*

![nvlbox_tex overview](docs/images/nvblox_tex.jpg)

### Abstract 
Teleoperation can be an important stepping stone to the autonomy of robotic systems. 
However, outsourcing long-term decision-making to a human operator
requires providing them with detailed information on the robot's state in real-time.  
In this project, we, therefore propose using a real-time 3D mapping approach leveraging independent geometry and surface texture resolution 
reconstruction to give the operator a detailed sense of the robot's surroundings while providing real-time performance and optimizing for the size
of the underlying output map representation. To this end, we implement an existing decoupled surface resolution mapping approach, TextureFusion,
on top of a proven scene scale reconstruction framework, voxblox. Using our implementation, nvblox_tex, we are able to reconstruct whole indoor environments in real-time
while improving robustness over TextureFusion as well as visual quality in comparison to voblox.
This enables our implementation to be a building block of future Teleoperation systems on legged robots such as ANYmal. 

# Native Installation
If you want to build natively, please follow these instructions. 
<!-- Instructions for docker are [further below](#docker). -->

## Install dependencies
We depend on:
- gtest
- glog
- gflags (to run experiments)
- CUDA 10.2 - 11.5 (others might work but are untested)
- Eigen (no need to explicitly install, a recent version is built into the library)
Please run
```
sudo apt-get install -y libgoogle-glog-dev libgtest-dev libgflags-dev python3-dev
cd /usr/src/googletest && sudo cmake . && sudo cmake --build . --target install
```

## Build all executables
In the project root do:

```bash
mkdir build
cd build
cmake --no-warn-unused-cli \
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE \
    -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo \
    -DCMAKE_C_COMPILER:FILEPATH=/bin/x86_64-linux-gnu-gcc-9 \
    -DCMAKE_CXX_COMPILER:FILEPATH=/bin/x86_64-linux-gnu-g++-9 \
    -S ../nvblox \
    -B . \
    -G Ninja
cmake --build . \
      --config RegWithDebInfo \
      --target tex_integration
```

## Run an example
In this example we fuse data from the [3DMatch dataset](https://3dmatch.cs.princeton.edu/). First let's grab the dataset. Here I'm downloading it to my dataset folder `~/dataset/3dmatch`.
```bash
wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets/sun3d-mit_76_studyroom-76-1studyroom2.zip -P ~/datasets/3dmatch
unzip ~/datasets/3dmatch/sun3d-mit_76_studyroom-76-1studyroom2.zip -d ~/datasets/3dmatch
```
Navigate to and run the `tex_integration` binary. From the nvblox_tex base folder run
```bash
cd nvblox/build/experiments
./tex_integration ~/datasets/3dmatch/sun3d-mit_76_studyroom-76-1studyroom2/  --mesh_output_path 3dmatch.ply --texture_output_path 3dmatch.png
```
Once it's done we can view the output mesh using the Open3D viewer (only displays the mesh) or imported to blender / meshlab.
```
pip3 install open3d
python3 ../../visualization/visualize_mesh.py 3dmatch.ply
```
you should see a mesh of a room:

![nvblox_tex_3dmatch](docs/images/nvblox_tex_3dmatch.jpg)


# License
NVIDIA source code is under an [open-source license](LICENSE) (Apache 2.0). :)
