#!/usr/bin/env python3
"""Benchmark: wall-clock vs audio length, peak VRAM, and whether VRAM is released.

⛔ A CRASHED RUN MUST NOT BE ABLE TO REPORT A SPEED. An earlier version of this
script timed a run that failed on an unusable path and cheerfully reported
"0.37x real-time — FASTER than real-time", because it divided elapsed time by
audio duration without checking that anything had been produced. A benchmark
that reports a win on a failure is worse than no benchmark. Exit status and
output existence are checked first, and a failure suppresses the ratio entirely.

Usage:
    python scripts/bench.py --audio example/audio.wav --source example/image.png \
        [--steps 5] [--tail-flush 3.0] [--out out/bench.mp4]
"""
import argparse
import os
import subprocess
import sys
import threading
import time


def gpu_used_mib():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def audio_seconds(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


def frame_count(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames",
                        "-select_streams", "v", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="example/audio.wav")
    ap.add_argument("--source", default="example/image.png")
    ap.add_argument("--out", default="out/bench.mp4")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--tail-flush", type=float, default=3.0)
    ap.add_argument("--data-root", default="./checkpoints/ditto_pytorch")
    ap.add_argument("--cfg", default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch_online.pkl")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    dur = audio_seconds(a.audio)
    base = gpu_used_mib()

    peak = [base]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], gpu_used_mib())
            time.sleep(0.5)

    t = threading.Thread(target=sample, daemon=True)
    t.start()

    cmd = [sys.executable, "inference_stream.py",
           "--data_root", a.data_root, "--cfg_pkl", a.cfg,
           "--audio_path", a.audio, "--source_path", a.source,
           "--output_path", a.out, "--tail_flush_s", str(a.tail_flush)]
    if a.steps is not None:
        cmd += ["--sampling_timesteps", str(a.steps)]

    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    wall = time.time() - t0
    stop.set()

    time.sleep(8)
    after = gpu_used_mib()
    ok = rc == 0 and os.path.exists(a.out)
    frames = frame_count(a.out) if ok else -1

    print()
    print(f"  audio            {dur:.2f}s")
    print(f"  exit code        {rc}")
    if not ok:
        print("  ⛔ RUN FAILED — no timing reported (a failure must never look fast)")
        sys.exit(1)
    print(f"  wall clock       {wall:.2f}s")
    print(f"  ratio            {wall/dur:.2f}x real-time")
    print(f"  frames out       {frames}  (expected {int(dur*25 + 0.999)})")
    print(f"  VRAM baseline    {base} MiB")
    print(f"  VRAM peak        {peak[0]} MiB  (delta {peak[0]-base} MiB)")
    print(f"  VRAM after +8s   {after} MiB  "
          f"({'released' if after <= base + 300 else 'STILL HELD'})")


if __name__ == "__main__":
    main()
