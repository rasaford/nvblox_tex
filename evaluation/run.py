from logging import warning
import subprocess
import argparse
import os
import shutil
import json
import GPUtil
from threading import Thread
import time

BUILD_DIR = "build2"
BASE_EXPERIMENT_DIR = "results"
# in seconds the frequency with which we sample the current gpu usage
GPU_MONITOR_DELAY = 1


class GPUMonitor(Thread):
    def __init__(self, delay: float, gpu_id: int):
        super(GPUMonitor, self).__init__()
        self._stopped = False
        self._delay = delay
        self._gpu_id = gpu_id
        self.samples = []
        self.start()

    def run(self):
        while not self._stopped:
            try:
                GPUs = GPUtil.getGPUs()
                # find the selected gpu
                GPU = next(gpu for gpu in GPUs if gpu.id == self._gpu_id)
                self.samples.append({**GPU.__dict__, **{"time": time.time()}})
            except StopIteration:
                warning(f"The GPU with id: {self._gpu_id} could not be found")
                pass

    def stop(self):
        self._stopped = True


def build_experiment(run_id: str, target: str, texel_size: int = 8, out_dir: str = "bin", build_dir: str = os.path.join(os.getcwd(), f"../{BUILD_DIR}")) -> str:
    bin_file = os.path.join(out_dir, run_id)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(build_dir, exist_ok=True)

    # call cmake to build the given target
    subprocess.call(["cmake", "--no-warn-unused-cli", "-DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE",
                     "-DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo", "-DCMAKE_C_COMPILER:FILEPATH=/bin/x86_64-linux-gnu-gcc-9",
                     "-DCMAKE_CXX_COMPILER:FILEPATH=/bin/x86_64-linux-gnu-g++-9",
                     f"-S{os.path.abspath('../nvblox')}", f"-B{os.path.abspath(build_dir)}", f"-DTEXEL_SIZE={texel_size}", "-G", "Ninja"])

    # call directly to make instead of cmake --build . since the latter option
    # as some issues with being called from python
    subprocess.call(["cmake", f"--build", os.path.abspath(build_dir),
                    "--config", "RelWithDebInfo", "--target", target])
    # copy the resulting binary to out_dir
    shutil.copy2(os.path.join(build_dir, "experiments", target), bin_file)
    return bin_file


def run_experiment(run_id: str, binary_path: str, dataset_path: str, out_dir: str, voxel_size: float = 0.05, texel_size: int = 8, num_frames: int = -1, start_frame: int = 0, cuda_device_id: int = 1):
    os.makedirs(out_dir, exist_ok=True)

    config_file = os.path.join(out_dir, f"{run_id}.config.json")
    mesh_file = os.path.join(out_dir, f"{run_id}.ply")
    if os.path.exists(config_file):
        print(f"Skipping run {run_id}, since it already exists")
        return

    timing_file = os.path.join(out_dir, f"{run_id}.timings.txt")
    gpu_usage_file = os.path.join(out_dir, f"{run_id}.gpu_usage.json")
    stdout_file = os.path.join(out_dir, f"{run_id}.stdout.txt")
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
        },
        "bin": {
            "texel_size": texel_size
        }
    }

    # only add texture_output_path were running an nvblox_tex run
    if "NVT" in run_id:
        config["params"]["texture_output_path"] = tex_file

    with open(config_file, "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)

    print(f"Running experiment {run_id}")
    print(json.dumps(config, indent=4, sort_keys=True))

    monitor = GPUMonitor(GPU_MONITOR_DELAY, cuda_device_id)
    try:
        command = list(config["direct"].values()) + \
            [f"--{k}={v}" for k, v in config["params"].items()]

        print(f"Executing command {' '.join(command)}")
        process_output = subprocess.check_output(
            command, stderr=subprocess.STDOUT).decode()
        # TODO: add env vars back to subprocess command
        # env={**os.environ, **config["env"]})
    except subprocess.CalledProcessError as e:
        # ignore errors as they are already in stdout / -err.txt
        print(e)
    print("Done writing result files")
    monitor.stop()

    with open(gpu_usage_file, "w") as f:
        json.dump(monitor.samples, f, indent=4, sort_keys=True)
    with open(stdout_file, "w") as f:
        f.write(str(process_output))


def experiment_1(dataset_root: str):
    # build the required binaries
    CORRIDOR_HANDHELD_L515 = os.path.join(
        dataset_root, "corridor_handheld_L515")
    OUT_DIR = os.path.join(BASE_EXPERIMENT_DIR, "Q1")

    NVT_Q1_01_BIN = build_experiment("NVT_Q1_01", target="tex_integration",
                                     texel_size=8)
    run_experiment("NVT_Q1_01", NVT_Q1_01_BIN, CORRIDOR_HANDHELD_L515,
                   OUT_DIR, voxel_size=0.04, texel_size=8)
    
    NVT_Q1_01_BIN = build_experiment("NVT_Q1_02", target="tex_integration",
                                     texel_size=16)
    run_experiment("NVT_Q1_02", NVT_Q1_01_BIN, CORRIDOR_HANDHELD_L515,
                   OUT_DIR, voxel_size=0.04, texel_size=8)

    NV_Q1_01_BIN = build_experiment("NV_Q1_01", target="fuse_3dmatch")

    run_experiment("NV_Q1_01", NV_Q1_01_BIN,
                   CORRIDOR_HANDHELD_L515, OUT_DIR, voxel_size=0.04, texel_size=1)
    run_experiment("NV_Q1_02", NV_Q1_01_BIN,
                   CORRIDOR_HANDHELD_L515, OUT_DIR, voxel_size=0.02, texel_size=1)
    run_experiment("NV_Q1_03", NV_Q1_01_BIN,
                   CORRIDOR_HANDHELD_L515, OUT_DIR, voxel_size=0.01, texel_size=1)
    run_experiment("NV_Q1_04", NV_Q1_01_BIN,
                   CORRIDOR_HANDHELD_L515, OUT_DIR, voxel_size=0.009, texel_size=1)


def experiment_2(dataset_root: str):
    texel_sizes = [2, 4, 8, 16]
    voxel_sizes = [0.16, 0.08, 0.04, 0.02, 0.01]
    ignore = {
        (0.01, 8),
        (0.01, 16)
    }

    DATASET = os.path.join(dataset_root, "corridor_handheld_L515")
    OUT_DIR = os.path.join(BASE_EXPERIMENT_DIR, "Q2")

    for texel_size in texel_sizes:
        bin_file = build_experiment(f"NVT_Q2_{texel_size}x{texel_size}",
                                    target="tex_integration", texel_size=texel_size)

        for voxel_size in voxel_sizes:
            run_id = f"NVT_Q2_{texel_size}x{texel_size}_{voxel_size}"

            if any(i[0] == voxel_size and i[1] == texel_size for i in ignore):
                print(f"Skipping run {run_id}, since it's on the ignore list")
                continue

            run_experiment(run_id, bin_file, DATASET,
                           OUT_DIR, voxel_size=voxel_size, texel_size=texel_size)


def experiment_3(dataset_root: str):
    DATASET = os.path.join(dataset_root, "corridor_ANYmal_L515")
    OUT_DIR = os.path.join(BASE_EXPERIMENT_DIR, "Q3")

    NVT_Q3_01_BIN = build_experiment(
        "NVT_Q3_01", target="tex_integration", texel_size=4)
    run_experiment("NVT_Q3_01", NVT_Q3_01_BIN, DATASET, OUT_DIR,
                   voxel_size=0.04, start_frame=1300, num_frames=2570, texel_size=4)
    run_experiment("NVT_Q3_02", NVT_Q3_01_BIN, DATASET, OUT_DIR,
                   voxel_size=0.02, start_frame=1300, num_frames=2570, texel_size=4)
    run_experiment("NVT_Q3_03", NVT_Q3_01_BIN, DATASET, OUT_DIR,
                   voxel_size=0.01, start_frame=1300, num_frames=2570, texel_size=4)


def experiment_4(dataset_root: str):
    raise NotImplementedError(
        "Not yet implemented, since I dont not have an NVIDIA Jetson to test on at the moment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    valid_experiments = ["1", "2", "3", "4", "all"]
    parser.add_argument("--experiment", type=str, required=True,
                        help=f"ID of the experiment we want to run. Possible values {' | '.join(valid_experiments)}")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="root dir of the datasets in 3dmatch format")
    parser.add_argument("--rebuild", action="store_true",
                        help="Required all binaries for the executed runs")
    # TODO: add rerun functionality if needed
    parser.add_argument("--rerun", type=str, default="",
                        help="Rebuild the run(s) with the given ids. "
                        "Needs to be passed as a comma separated list with no spaces.")
    args = parser.parse_args()

    if args.experiment == "1":
        experiment_1(args.dataset_root)
    elif args.experiment == "2":
        experiment_2(args.dataset_root)
    elif args.experiment == "3":
        experiment_3(args.dataset_root)
    elif args.experiment == "4":
        experiment_4(args.dataset_root)
    elif args.experiment == "all":
        experiment_1(args.dataset_root)
        experiment_2(args.dataset_root)
        experiment_3(args.dataset_root)
        experiment_4(args.dataset_root)
    else:
        raise RuntimeError(f"Invalid experiment ID: {args.experiment}")
