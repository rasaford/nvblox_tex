import glob
import json
import numbers
import os
import pprint
import re
from collections import defaultdict
from multiprocessing.sharedctypes import Value
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.style.use('ggplot')


def plot_gpu_usage(data_dir: str):
    plt.figure(figsize=(10, 5))

    for path in glob.glob(os.path.join(data_dir, "*.gpu_usage.json")):

        run_id = Path(path).stem.replace(".gpu_usage", "")

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

        plt.plot(times, memory_used, label=run_id)

    plt.xlabel("$s$")
    plt.ylabel("MiB")
    plt.title(f"GPU Memory Usage")
    plt.legend()
    plt.tight_layout()
    plt.show()


def parse_texture_fusion_timings_file(path: str):
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


def parse_nvblox_timings_file(path: str):
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
    plt.figure(figsize=(10, 5))

    labels = []
    int_times = []
    int_times_stddev = []
    file_times = []
    file_times_stddev = []
    TIME_SCALE = 100.
    BAR_WIDTH = 0.75
    for path in sorted(glob.glob(os.path.join(data_dir, "*.timings.txt"))):

        run_id = Path(path).stem.replace(".timings", "")

        if run_id.startswith("NV"):
            labels.append(run_id)
            timings = parse_nvblox_timings_file(path)
            int_times.append(
                timings["3dmatch/time_per_frame"]["mean"] * TIME_SCALE)
            int_times_stddev.append(
                timings["3dmatch/time_per_frame"]["stddev"] * TIME_SCALE)
            file_times.append(timings["3dmatch/file_loading"]["mean"] * TIME_SCALE)
            file_times_stddev.append(
                timings["3dmatch/file_loading"]["stddev"] * TIME_SCALE)
        elif run_id.startswith("TF"):
            timings = parse_texture_fusion_timings_file(path)
            pprint.pprint(timings)
        else:
            raise RuntimeError(f"unknown run_id format: {run_id}")


    plt.bar(labels, file_times, width=BAR_WIDTH,
            yerr=file_times_stddev, label="File loading")
    plt.bar(labels, int_times, bottom=file_times, width=BAR_WIDTH,
            yerr=int_times_stddev, label="Integration")
    plt.ylabel("$ms$")
    plt.xticks(rotation=45,)
    plt.title("Mean integration time per frame")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        help="Directory of the sample")
    args = parser.parse_args()

    # plot_gpu_usage(args.data_dir)
    plot_processing_time(args.data_dir)
