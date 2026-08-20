#!/usr/bin/env python3
"""Streaming inference with a flushed tail — the PyTorch/no-TensorRT path.

TWO BUGS THIS FIXES, both measured on an RTX 5090.

1. ⛔ THE STREAMING PATH TRUNCATES THE END OF EVERY CLIP.
   inference.py's online branch feeds chunks and calls close() immediately.
   The pipeline carries internal lookahead/smoothing state, so the last frames
   are still in flight and are lost. Measured on the shipped 15.75 s example:

       offline path                      394 frames  (correct)
       streaming, no tail flush          340 frames  (54 short = 2.16 s)
       streaming + 0.6 s of silence      355 frames
       streaming + 1.2 s of silence      370 frames
       streaming + 2.4 s of silence      400 frames

   Exactly 25 frames recovered per second of padding, 1:1 with the output
   framerate — a fixed pipeline depth, not a rate mismatch. The symptom in a
   finished video is lipsync drifting off in the finalseconds, which is easy to
   misread as model quality rather than a dropped tail.

   We do NOT hardcode a magic flush length. The depth is not derivable from the
   published config (chunksize (3,5,2) does not predict 54), so guessing a
   formula would be a fossil the moment upstream changes a buffer. Instead:
   flush GENEROUSLY, then TRIM to the frame count the audio actually implies.
   Over-flushing costs a little compute; under-flushing loses content.

2. inference.py performs the audio mux with ffmpeg AFTER SDK.close(), outside
   the SDK. Call close() yourself and you get a silent <output>.tmp.mp4 and no
   final file — with no error. Handled here.

Also exposes --sampling_timesteps, which upstream leaves at 50 and which
dominates runtime (see BLACKWELL.md: 1.90x -> 1.03x real-time at 5 steps).

Usage:
    python inference_stream.py \
        --data_root ./checkpoints/ditto_pytorch \
        --cfg_pkl   ./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch_online.pkl \
        --audio_path in.wav --source_path face.png --output_path out.mp4 \
        --sampling_timesteps 5
"""
import argparse
import math
import os
import subprocess
import sys
import time

import librosa
import numpy as np

from stream_pipeline_online import StreamSDK

CHUNKSIZE = (3, 5, 2)
FPS = 25


def stream(SDK, audio, chunksize=CHUNKSIZE):
    """Feed audio in ~200 ms hops, exactly as a live TTS would deliver it."""
    hop = chunksize[1] * 640                                  # 3200 samples @16k = 200 ms
    split_len = int(sum(chunksize) * 0.04 * 16000) + 80        # 6480 = 405 ms window
    padded = np.concatenate(
        [np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
    for i in range(0, len(padded), hop):
        chunk = padded[i:i + split_len]
        if len(chunk) < split_len:
            chunk = np.pad(chunk, (0, split_len - len(chunk)), mode="constant")
        SDK.run_chunk(chunk, chunksize)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./checkpoints/ditto_pytorch")
    ap.add_argument("--cfg_pkl",
                    default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch_online.pkl")
    ap.add_argument("--audio_path", required=True)
    ap.add_argument("--source_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--sampling_timesteps", type=int, default=None,
                    help="diffusion steps for audio2motion (upstream default 50)")
    ap.add_argument("--tail_flush_s", type=float, default=3.0,
                    help="silence appended to flush in-flight frames; over-flush "
                         "then trim. Raise it if the frame count still falls short.")
    a = ap.parse_args()

    t0 = time.time()
    SDK = StreamSDK(a.cfg_pkl, a.data_root)
    t_load = time.time() - t0

    setup_kwargs = {}
    if a.sampling_timesteps is not None:
        setup_kwargs["sampling_timesteps"] = a.sampling_timesteps

    t0 = time.time()
    SDK.setup(a.source_path, a.output_path, **setup_kwargs)
    t_setup = time.time() - t0

    audio, _ = librosa.core.load(a.audio_path, sr=16000)
    duration = len(audio) / 16000
    want_frames = math.ceil(duration * FPS)
    SDK.setup_Nd(N_d=want_frames)

    t0 = time.time()
    stream(SDK, np.concatenate(
        [audio, np.zeros((int(a.tail_flush_s * 16000),), dtype=np.float32)], 0))
    SDK.close()
    t_gen = time.time() - t0

    tmp = SDK.tmp_output_path
    if not os.path.exists(tmp):
        sys.exit(f"FAILED: no intermediate video at {tmp}")

    # Trim to the frames the audio implies, then mux. -frames:v is applied to the
    # decoded stream, so this cuts the over-flushed tail without re-encoding video.
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-i", a.audio_path,
           "-frames:v", str(want_frames), "-map", "0:v", "-map", "1:a",
           "-c:v", "copy", "-c:a", "aac", a.output_path]
    if subprocess.run(cmd).returncode != 0:
        sys.exit("FAILED: ffmpeg mux/trim returned non-zero")

    print(f"model load   {t_load:6.2f}s   (once per process)")
    print(f"avatar setup {t_setup:6.2f}s   (once per face)")
    print(f"generation   {t_gen:6.2f}s   for {duration:.2f}s audio = "
          f"{t_gen / duration:.2f}x real-time")
    print(f"output       {a.output_path}  ({want_frames} frames expected)")


if __name__ == "__main__":
    main()
