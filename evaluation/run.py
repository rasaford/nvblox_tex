from copy import copy
from pickle import BUILD
import subprocess
import argparse
import os
import shutil
from tracemalloc import start
import yaml

BUILD_DIR = "build2"
BASE_EXPERIMENT_DIR = "results"


def build_experiment(run_id: str, target: str, texel_size: int = 8, out_dir: str = "bin", build_dir: str = os.path.join(os.getcwd(), f"../{BUILD_DIR}"), jobs: int = 16, rebuild=False) -> str:
    bin_file = os.path.join(out_dir, run_id)
    if os.path.exists(bin_file) and not rebuild:
        return bin_file

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(build_dir, exist_ok=True)

    # call cmake to build the given target
    subprocess.call(
        ["cmake", "../nvblox", f"-DTEXEL_SIZE={texel_size}"], cwd=build_dir)
    # call directly to make instead of cmake --build . since the latter option
    # as some issues with being called from python
    subprocess.call(["make", target, f"-j{jobs}"], cwd=build_dir)
    # copy the resulting binary to out_dir
    shutil.copy2(os.path.join(build_dir, "experiments", target), bin_file)
    return bin_file


def run_experiment(run_id: str, binary_path: str, dataset_path: str, out_dir: str, voxel_size: float = 0.05, num_frames: int = -1, start_frame: int = 0, cuda_device_id: int = 1):
    os.makedirs(out_dir, exist_ok=True)

    config_file = os.path.join(out_dir, f"{run_id}.config.yaml")
    mesh_file = os.path.join(out_dir, f"{run_id}.ply")
    timing_file = os.path.join(out_dir, f"{run_id}.timings.txt")
    tex_file = os.path.join(out_dir, f"{run_id}.png")
    config = {
        "direct": {
            "binary_path": binary_path,
            "dataset_path": dataset_path
        },
        "params": {
            "mesh_output_path": mesh_file,
            "timing_output_path": timing_file,
            "voxel_size": voxel_size,
            "num_frames": num_frames,
            "start_frame": start_frame
        },
        "env": {
            "CUDA_VISIBLE_DEVICES": cuda_device_id
        }
    }

    # only add texture_output_path were running an nvblox_tex run
    if "NVT" in run_id:
        config["params"]["texture_output_path"] = tex_file

    with open(config_file, "w") as f:
        yaml.dump(config, f)

    subprocess.call(
        list(config["direct"].values()) +
        [f"--{k}={v}" for k, v in config["params"].items()])
    # TODO: add env vars back to subprocess command
    # env={**os.environ, **config["env"]})


def experiment_1(dataset_root: str, rebuild=False):
    # build the required binaries
    CORRIDOR_HANDHELD_L515 = os.path.join(
        dataset_root, "corridor_handheld_L515")
    OUT_DIR = os.path.join(BASE_EXPERIMENT_DIR, "Q1")

    NVT_Q1_01_BIN = build_experiment("NVT_Q1_01", target="tex_integration",
                                     texel_size=8, rebuild=rebuild)
    run_experiment("NVT_Q1_01", NVT_Q1_01_BIN, CORRIDOR_HANDHELD_L515,
                   OUT_DIR, voxel_size=0.05, num_frames=10)

    NV_Q1_01_BIN = build_experiment("NV_Q1_01", target="fuse_3dmatch",
                                    rebuild=rebuild)

    run_experiment("NV_Q1_01", NV_Q1_01_BIN,
                   CORRIDOR_HANDHELD_L515, OUT_DIR, voxel_size=0.05)
    run_experiment("NV_Q1_02", NV_Q1_01_BIN,
                   CORRIDOR_HANDHELD_L515, OUT_DIR, voxel_size=0.02)
    run_experiment("NV_Q1_03", NV_Q1_01_BIN,
                   CORRIDOR_HANDHELD_L515, OUT_DIR, voxel_size=0.01)
    run_experiment("NV_Q1_05", NV_Q1_01_BIN,
                   CORRIDOR_HANDHELD_L515, OUT_DIR, voxel_size=0.005)


def experiment_2(dataset_root: str, rebuild=False):
    texel_sizes = [1, 4, 8, 16]
    voxel_sizes = [0.2, 0.1, 0.05, 0.02]
    DATASET = os.path.join(dataset_root, "corridor_handheld_L515")
    OUT_DIR = os.path.join(BASE_EXPERIMENT_DIR, "Q2")

    for texel_size in texel_sizes:
        bin_file = build_experiment(f"NVT_Q2_{texel_size}x{texel_size}",
                                    target="tex_integration", texel_size=texel_size, rebuild=rebuild)

        for voxel_size in voxel_sizes:
            run_id = f"NVT_Q2_{texel_size}x{texel_size}_{voxel_size}"
            run_experiment(run_id, bin_file, DATASET,
                           OUT_DIR, voxel_size=voxel_size)


def experiment_3(dataset_root: str, rebuild=False):
    run_id = "NVT_Q3_01"
    NVT_Q3_01_BIN = build_experiment(run_id, target="tex_integration",
                                     texel_size=4, rebuild=rebuild)
    DATASET = os.path.join(dataset_root, "corridor_ANYmal_L515")
    OUT_DIR = os.path.join(BASE_EXPERIMENT_DIR, "Q3")
    run_experiment(run_id, NVT_Q3_01_BIN, DATASET, OUT_DIR, voxel_size=0.05)


def experiment_4(dataset_root: str, rebuild=False):
    raise NotImplementedError(
        "Not yet implemented, since I dont not have an NVIDIA Jetson to test on at the moment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, required=True,
                        help="ID of the experiment we want to run")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="root dir of the datasets in 3dmatch format")
    parser.add_argument("--rebuild", action="store_true",
                        help="If the required binaries for the run should be rebuilt")
    args = parser.parse_args()

    if args.experiment == 1:
        experiment_1(args.dataset_root, args.rebuild)
    elif args.experiment == 2:
        experiment_2(args.dataset_root, args.rebuild)
    elif args.experiment == 3:
        experiment_3(args.dataset_root, args.rebuild)
    elif args.experiment == 4:
        experiment_4(args.dataset_root, args.rebuild)
    else:
        raise RuntimeError(f"Invalid experiment ID: {args.experiment}")
