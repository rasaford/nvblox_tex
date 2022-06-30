import glob
import json
import numbers
import os
import pprint
import re
from collections import defaultdict
from multiprocessing.sharedctypes import Value
from pathlib import Path
from sre_constants import IN_IGNORE

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.style.use('ggplot')

TIME_SCALE = 1000.
BAR_WIDTH = 0.75


def plot_gpu_usage(data_dir: str):
    plt.figure(figsize=(6, 3))

    # keep = [
    #     "NVT_Q1_01",
    #     "NV_Q1_01",
    #     "NV_Q1_04",
    # ]
    # keep = [
    #     "NVT_Q1_02",
    #     "Tf_Q1_01",
    #     # "NV_Q1_01",
    #     # "NV_Q1_04",
    # ]
    keep = [
        "NVT_Q2_2x2_0.01",
        "NVT_Q2_2x2_0.02",
        "NVT_Q2_2x2_0.04",
        "NVT_Q2_2x2_0.08",
        "NVT_Q2_4x4_0.01",
        "NVT_Q2_4x4_0.02",
        "NVT_Q2_4x4_0.04",
        "NVT_Q2_4x4_0.08",
        "NVT_Q2_8x8_0.01",
        "NVT_Q2_8x8_0.02",
        "NVT_Q2_8x8_0.04",
        "NVT_Q2_8x8_0.08",
        "NVT_Q2_16x16_0.01",
        "NVT_Q2_16x16_0.02",
        "NVT_Q2_16x16_0.04",
        "NVT_Q2_16x16_0.08",
    ]
    # names = {
    #     "NVT_Q1_01": "nvblox_tex $\\nu = 0.04m, \\tau = 8 \\times 8 px$",
    #     "NVT_Q1_02": "nvblox_tex $\\nu=0.01m , \\tau =4 \\times 4px$",
    #     "TF_Q1_01": "TextureFusion $\\nu = 0.01m, \\tau = 4 \\times 4 px$",
    #     "NV_Q1_01": "nvblox $\\nu = 0.04m$",
    #     "NV_Q1_02": "nvblox v=0.02",
    #     "NV_Q1_03": "nvblox v=0.01",
    #     "NV_Q1_04": "nvblox $\\nu = 0.009m$",
    # }
    names = {
        "NVT_Q2_2x2_0.01": "$\\nu=0.01, \\tau=2 \\times 2 px$",
        "NVT_Q2_2x2_0.02": "$\\nu=0.02, \\tau=2 \\times 2 px$",
        "NVT_Q2_2x2_0.04": "$\\nu=0.04, \\tau=2 \\times 2 px$",
        "NVT_Q2_2x2_0.08": "$\\nu=0.08, \\tau=2 \\times 2 px$",
        "NVT_Q2_2x2_0.16": "$\\nu=0.16, \\tau=2 \\times 2 px$",
        "NVT_Q2_4x4_0.01": "$\\nu=0.01, \\tau=4 \\times 4 px$",
        "NVT_Q2_4x4_0.02": "$\\nu=0.02, \\tau=4 \\times 4 px$",
        "NVT_Q2_4x4_0.04": "$\\nu=0.04, \\tau=4 \\times 4 px$",
        "NVT_Q2_4x4_0.08": "$\\nu=0.08, \\tau=4 \\times 4 px$",
        "NVT_Q2_4x4_0.16": "$\\nu=0.16, \\tau=4 \\times 4 px$",
        "NVT_Q2_8x8_0.01": "$\\nu=0.01, \\tau=8 \\times 8 px$",
        "NVT_Q2_8x8_0.02": "$\\nu=0.02, \\tau=8 \\times 8 px$",
        "NVT_Q2_8x8_0.04": "$\\nu=0.04, \\tau=8 \\times 8 px$",
        "NVT_Q2_8x8_0.08": "$\\nu=0.08, \\tau=8 \\times 8 px$",
        "NVT_Q2_8x8_0.16": "$\\nu=0.16, \\tau=8 \\times 8 px$",
        "NVT_Q2_16x16_0.01": "$\\nu=0.01, \\tau=16 \\times 16 px$",
        "NVT_Q2_16x16_0.02": "$\\nu=0.02, \\tau=16 \\times 16 px$",
        "NVT_Q2_16x16_0.04": "$\\nu=0.04, \\tau=16 \\times 16 px$",
        "NVT_Q2_16x16_0.08": "$\\nu=0.08, \\tau=16 \\times 16 px$",
        "NVT_Q2_16x16_0.16": "$\\nu=0.16, \\tau=16 \\times 16 px$",
    }

    for path in sorted(glob.glob(os.path.join(data_dir, "*.gpu_usage.json"))):

        run_id = Path(path).stem.replace(".gpu_usage", "")

        if run_id not in keep:
            print(f"Skipping {run_id}, since it's ignored")
            continue

        with open(path, "r") as f:
            measurements = json.load(f)

        times = np.array([sample["time"] for sample in measurements])
        times -= times[0]

        load = np.array([sample["load"] for sample in measurements])
        memory_free = np.array([sample["memoryFree"]
                               for sample in measurements])
        memory_total = np.array([sample["memoryTotal"]
                                for sample in measurements])
        memory_used = np.array([sample["memoryUsed"]
                               for sample in measurements])
        memory_util = np.array([sample["memoryUtil"]
                               for sample in measurements])
        temperature = np.array([sample["temperature"]
                               for sample in measurements])

        plt.plot(times/60., memory_used, label=names[run_id])

    plt.xlabel("$min$")
    plt.ylabel("MiB")
    plt.title(f"GPU Memory Usage")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./plot_gpu1.pdf")
    plt.show()

    plt.figure(figsize=(6, 3))
    plt.title("Max Memory Usage")

    labels = []
    memory = []

    for path in reversed(sorted(glob.glob(os.path.join(data_dir, "*.gpu_usage.json")))):

        run_id = Path(path).stem.replace(".gpu_usage", "")

        if run_id not in keep:
            print(f"Skipping {run_id}, since it's ignored")
            continue

        with open(path, "r") as f:
            measurements = json.load(f)

        times = np.array([sample["time"] for sample in measurements])
        times -= times[0]

        load = np.array([sample["load"] for sample in measurements])
        memory_free = np.array([sample["memoryFree"]
                               for sample in measurements])
        memory_total = np.array([sample["memoryTotal"]
                                for sample in measurements])
        memory_used = np.array([sample["memoryUsed"]
                               for sample in measurements])
        memory_util = np.array([sample["memoryUtil"]
                               for sample in measurements])
        temperature = np.array([sample["temperature"]
                               for sample in measurements])

        labels.append(names[run_id])
        memory.append(memory_used.max())

    plt.bar(labels, memory, width=BAR_WIDTH)
    plt.ylabel("MiB")
    plt.xticks(rotation=90.)
    plt.tight_layout()
    plt.savefig("./plot_gpu2.pdf")
    plt.show()


def parse_texture_fusion_timings(path: str):
    with open(path, "r") as f:
        lines = f.readlines()

    timings = defaultdict(dict)
    frame_idx = 0
    reading_lines = False

    # read timing blocks
    for line in lines:
        line = line.strip()
        is_frame = line.startswith("decompressing frame")
        is_empty = len(line.strip()) == 0
        if is_empty:
            reading_lines = False
        if is_frame:
            reading_lines = True

        if not is_frame and not reading_lines:
            continue

        if is_frame:
            frame_idx = int(line.replace("decompressing frame", "").strip())
            continue

        try:
            key, value = line.split(": ")
            value = value.strip()
            # parsing priorities: float -> int -> str
            parsed = None
            try:
                parsed = float(value)
            except ValueError:
                try:
                    parsed = int(value)
                except ValueError:
                    parsed = value

            timings[frame_idx][key.strip()] = parsed
        except ValueError:
            continue

    grouped_times = defaultdict(list)
    for frame_idx, times in timings.items():
        for name, time in times.items():
            if not isinstance(time, numbers.Number):
                continue
            grouped_times[name].append(time)

    entries = {}
    for name, samples in grouped_times.items():
        s = np.asarray(samples)
        entries[name] = {
            "num_samples": len(s),
            "total": np.sum(s),
            "mean": np.mean(s),
            "stddev": np.std(s),
            "min": np.min(s),
            "max": np.max(s)
        }
    return entries


def parse_nvblox_timings(path: str):
    with open(path, "r") as f:
        lines = f.readlines()

    entries = {}
    # skip the first two lines since they are headers
    for line in lines[2:]:
        line = re.sub(r"\(|\)|\[|\]|\+-|,|\t", " ", line)
        cols = [col.strip() for col in line.split(" ") if col.strip()]
        entries[cols[0]] = {
            "num_samples": int(cols[1]),
            "total": float(cols[2]),
            "mean": float(cols[3]),
            "stddev": float(cols[4]),
            "min": float(cols[5]),
            "max": float(cols[6])
        }
    return entries


def plot_processing_time(data_dir: str):
    plt.figure(figsize=(12, 3))

    # ignore = [
    #     "NVT_Q1_01",
    #     "NV_Q1_01",
    #     "NV_Q1_02",
    #     "NV_Q1_03",
    #       "NV_Q1_04",
    #     "TF_Q1_02",
    #     # "TF_Q1_01"
    # ]
    # ignore = ["NVT_Q1_01", "NV_Q1_01", "NV_Q1_02",
    #           "NV_Q1_03", "NV_Q1_03", "NV_Q1_04", "TF_Q1_02"]
    # names = {
    #     "NVT_Q1_01": "nvblox_tex $\\nu = 0.04m, \\tau = 8 \\times 8 px$",
    #     "NVT_Q1_02": "nvblox_tex $\\nu = 0.01m, \\tau = 4 \\times 4 px$",
    #     "TF_Q1_01": "TextureFusion $\\nu = 0.01m, \\tau = 4 \\times 4 px$",
    #     "NV_Q1_01": "nvblox $\\nu=0.04m$",
    #     "NV_Q1_04": "nvblox $\\nu=0.009m$"
    # }
    ignore = [
        "NVT_Q2_2x2_0.16",
        "NVT_Q2_4x4_0.16",
        "NVT_Q2_8x8_0.16",
        "NVT_Q2_16x16_0.16",
    ]
    names = {
        "NVT_Q2_2x2_0.01": "$\\nu=0.01$ \n $\\tau=2 \\times 2 px$",
        "NVT_Q2_2x2_0.02": "$\\nu=0.02$ \n $\\tau=2 \\times 2 px$",
        "NVT_Q2_2x2_0.04": "$\\nu=0.04$ \n $\\tau=2 \\times 2 px$",
        "NVT_Q2_2x2_0.08": "$\\nu=0.08$ \n $\\tau=2 \\times 2 px$",
        "NVT_Q2_2x2_0.16": "$\\nu=0.16$ \n $\\tau=2 \\times 2 px$",
        "NVT_Q2_4x4_0.01": "$\\nu=0.01$ \n $\\tau=4 \\times 4 px$",
        "NVT_Q2_4x4_0.02": "$\\nu=0.02$ \n $\\tau=4 \\times 4 px$",
        "NVT_Q2_4x4_0.04": "$\\nu=0.04$ \n $\\tau=4 \\times 4 px$",
        "NVT_Q2_4x4_0.08": "$\\nu=0.08$ \n $\\tau=4 \\times 4 px$",
        "NVT_Q2_4x4_0.16": "$\\nu=0.16$ \n $\\tau=4 \\times 4 px$",
        "NVT_Q2_8x8_0.01": "$\\nu=0.01$ \n $\\tau=8 \\times 8 px$",
        "NVT_Q2_8x8_0.02": "$\\nu=0.02$ \n $\\tau=8 \\times 8 px$",
        "NVT_Q2_8x8_0.04": "$\\nu=0.04$ \n $\\tau=8 \\times 8 px$",
        "NVT_Q2_8x8_0.08": "$\\nu=0.08$ \n $\\tau=8 \\times 8 px$",
        "NVT_Q2_8x8_0.16": "$\\nu=0.16$ \n $\\tau=8 \\times 8 px$",
        "NVT_Q2_16x16_0.01": "$\\nu=0.01$ \n $\\tau=16 \\times 16 px$",
        "NVT_Q2_16x16_0.02": "$\\nu=0.02$ \n $\\tau=16 \\times 16 px$",
        "NVT_Q2_16x16_0.04": "$\\nu=0.04$ \n $\\tau=16 \\times 16 px$",
        "NVT_Q2_16x16_0.08": "$\\nu=0.08$ \n $\\tau=16 \\times 16 px$",
        "NVT_Q2_16x16_0.16": "$\\nu=0.16$ \n $\\tau=16 \\times 16 px$",
    }

    labels = []
    int_color = []
    int_color_std = []
    int_esdf = []
    int_esdf_std = []
    int_tsdf = []
    int_tsdf_std = []
    int_mesh = []
    int_mesh_std = []

    total_time = []
    total_time_std = []

    for path in reversed(sorted(glob.glob(os.path.join(data_dir, "*.timings.txt")))):

        run_id = Path(path).stem.replace(".timings", "")
        if run_id in ignore:
            print(f"ingnoring run {run_id}")
            continue

        labels.append(names[run_id])

        if run_id.startswith("NV"):

            timings = parse_nvblox_timings(path)
            int_color.append(
                timings["3dmatch/integrate_color"]["mean"] * TIME_SCALE)
            int_color_std.append(
                timings["3dmatch/integrate_color"]["stddev"] * TIME_SCALE)

            int_esdf.append(
                timings["3dmatch/integrate_esdf"]["mean"] * TIME_SCALE)
            int_esdf_std.append(
                timings["3dmatch/integrate_esdf"]["stddev"] * TIME_SCALE)

            int_tsdf.append(
                timings["3dmatch/integrate_tsdf"]["mean"] * TIME_SCALE)
            int_tsdf_std.append(
                timings["3dmatch/integrate_tsdf"]["stddev"] * TIME_SCALE)

            int_mesh.append(
                timings["3dmatch/mesh"]["mean"] * TIME_SCALE)
            int_mesh_std.append(
                timings["3dmatch/mesh"]["stddev"] * TIME_SCALE)

            total_time.append(0)
            total_time_std.append(0)

            print(
                f'{run_id}: {timings["3dmatch/time_per_frame"]["mean"]} +- {timings["3dmatch/time_per_frame"]["stddev"]}')
        elif run_id.startswith("TF"):
            int_color.append(0)
            int_color_std.append(0)
            int_esdf.append(0)
            int_esdf_std.append(0)
            int_tsdf.append(0)
            int_tsdf_std.append(0)
            int_mesh.append(0)
            int_mesh_std.append(0)

            timings = parse_texture_fusion_timings(path)
            total_mean_ms = 0
            total_std_ms = 0
            for k, v in timings.items():
                if v["mean"] < 100:
                    total_mean_ms += v["mean"]
                    total_std_ms += v["stddev"] ** 2

            total_std_ms = np.sqrt(total_std_ms)
            print(f"{run_id}: {total_mean_ms} +- {total_std_ms}")

            total_time.append(total_mean_ms)
            total_time_std.append(total_std_ms)
        else:
            raise RuntimeError(f"unknown run_id format: {run_id}")

    int_color = np.asarray(int_color)
    int_color_std = np.asarray(int_color_std)
    int_esdf = np.asarray(int_esdf)
    int_esdf_std = np.asarray(int_esdf_std)
    int_tsdf = np.asarray(int_tsdf)
    int_tsdf_std = np.asarray(int_tsdf_std)
    int_mesh = np.asarray(int_mesh)
    int_mesh_std = np.asarray(int_mesh_std)

    plt.bar(labels, int_tsdf, width=BAR_WIDTH,
            yerr=int_tsdf_std, label="TSDF Integration")

    plt.bar(labels, int_esdf, bottom=int_tsdf, width=BAR_WIDTH,
            yerr=int_esdf_std, label="ESDF Integration")

    plt.bar(labels, int_mesh, bottom=int_tsdf + int_esdf, width=BAR_WIDTH,
            yerr=int_mesh_std, label="Meshing")

    plt.bar(labels, int_color, bottom=int_tsdf + int_esdf + int_mesh, width=BAR_WIDTH,
            yerr=int_color_std, label="Color Integration")

    plt.bar(labels, total_time, bottom=int_tsdf + int_esdf + int_mesh + int_color, width=BAR_WIDTH,
            yerr=total_time_std, label="Total Time")

    plt.ylabel("$ms$")
    plt.title("Mean integration time per frame")
    plt.legend()
    # plt.xticks(rotation=90.)
    plt.tight_layout()
    plt.savefig("./plot.pdf")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        help="Directory of the sample")
    args = parser.parse_args()

    # plot_gpu_usage(args.data_dir)
    plot_processing_time(args.data_dir)
