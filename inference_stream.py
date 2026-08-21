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
import soundfile as sf

from stream_pipeline_online import StreamSDK

CHUNKSIZE = (3, 5, 2)
FPS = 25

# Measured on an RTX 5090: the streaming path emits the right NUMBER of frames once
# padded, but the final 44 frames carry no motion -- the same 44 on a 15.75s clip
# (motion dies at frame 350 of 394) and a 21.74s clip (500 of 544). Constant, so it
# is a fixed pipeline depth, not a rate mismatch.
PAD_IS_NOT_SILENCE = """
The end of a clip needs padding before generation, and the padding must NOT be
digital silence.

Measured: motion stops ~1.9s before the SPEECH ends, and appending silence does not
recover it -- it extends the frozen region instead. Reproducible at both lengths:

    speech 15.00s -> motion dies at 13.0s   (2.00s short)
    speech 21.74s -> motion dies at 20.0s   (1.74s short)

The cause is the motion model's window, not a dropped buffer. seq_frames=80 (3.2s),
valid_clip_len=70 (2.8s), so the final ~2s of real speech sits in a window whose
FUTURE half is the silence you appended -- and the model correctly renders a mouth
coming to rest. It is reacting to the silence.

Padding with speech-SHAPED audio instead (the tail reversed: right spectrum, no
words) animates straight through. Frames 325-375, the previously dead zone, go from
0.05 to 0.302 against a body rate of 0.313 -- a ratio of 0.96.

⇒ This is a FILE-MODE artefact only. A live stream never contains an artificial
silence cliff, so it needs no padding: the mouth comes to rest when the speaker
actually stops, which is correct. Do not port this to a live path.
"""


# ⛔ THE PIPELINE SWALLOWS ITS FIRST ~53 FRAMES, AT THE FRONT, NOT THE TAIL.
# Emitted frame count is always fed_frames - 53, measured across lengths:
#   fed 437 -> 385   fed 618 -> 565   fed 743 -> 690
# and an onset probe (2s silence, 2s speech, 2s silence) shows mouth activity at
# 0.0-2.8s against speech at 2.0-4.0s -- i.e. emitted frame 0 corresponds to audio
# at ~2.12s, so pairing it with t=0 makes the video LEAD by two seconds.
#
# This is what "lip sync is way off" was, and an earlier revision of this file
# misdiagnosed the same 53 frames as a TAIL loss and "fixed" it by trimming, which
# preserved the frame count and left the shift in place.
#
# So: pad the FRONT by exactly the lead, and let the swallow consume filler.
LEAD_FRAMES = 53


def speech_filler(audio, pad_s, sr=16000):
    """Speech-shaped tail filler. Reversed audio: same spectrum, no intelligible words."""
    n = int(pad_s * sr)
    if len(audio) == 0:
        return np.zeros((n,), dtype=np.float32)
    src = audio[-min(n, len(audio)):][::-1]
    return np.tile(src, int(np.ceil(n / len(src))))[:n].astype(np.float32)


def verify_tail(path, speech_frames):
    """Assert the last second of SPEECH actually moves. Loud, not silent."""
    try:
        import imageio.v2 as iio
    except ImportError:
        print("verify: imageio unavailable, tail NOT checked")
        return
    prev, d = None, []
    for f in iio.get_reader(path):
        f = f.astype(np.int16)
        if prev is not None:
            d.append(np.abs(f - prev).mean())
        prev = f
    d = np.array(d)
    if len(d) < speech_frames:
        print(f"verify: FAILED -- {len(d)+1} frames for {speech_frames} of speech")
        return
    body = d[int(len(d) * 0.2):int(len(d) * 0.7)].mean()
    tail = d[max(0, speech_frames - FPS):speech_frames].mean()
    ratio = tail / body if body else 0.0
    verdict = "OK" if ratio > 0.5 else "⛔ FROZEN OVER SPEECH -- raise --tail_pad_s"
    print(f"verify tail   last 1s of speech moves {ratio:.2f}x the body rate  {verdict}")


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
    ap.add_argument("--tail_pad_s", type=float, default=3.0,
                    help="Seconds of SPEECH-SHAPED filler appended before generation "
                         "and trimmed after. Must not be silence -- see PAD_IS_NOT_SILENCE.")
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
    speech_s = len(audio) / 16000
    speech_frames = math.ceil(speech_s * FPS)

    lead = speech_filler(audio, LEAD_FRAMES / FPS)
    fed = np.concatenate([lead, audio, speech_filler(audio, a.tail_pad_s)], 0)
    SDK.setup_Nd(N_d=math.ceil(len(fed) / 16000 * FPS))

    t0 = time.time()
    stream(SDK, fed)
    SDK.close()
    t_gen = time.time() - t0

    tmp = SDK.tmp_output_path
    if not os.path.exists(tmp):
        sys.exit(f"FAILED: no intermediate video at {tmp}")

    # Trim to the frames the audio implies, then mux. -frames:v is applied to the
    # decoded stream, so this cuts the over-flushed tail without re-encoding video.
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-i", a.audio_path,
           "-frames:v", str(speech_frames), "-map", "0:v", "-map", "1:a",
           "-c:v", "copy", "-c:a", "aac", a.output_path]
    if subprocess.run(cmd).returncode != 0:
        sys.exit("FAILED: ffmpeg mux/trim returned non-zero")

    # ⛔ VERIFY THE SPEECH IS ANIMATED, do not assume the padding was enough.
    # A frozen face over live speech is the failure this whole argument exists for,
    # and it is invisible in the frame COUNT -- which is why it survived one round
    # of "fixed" already. Measure and say so.
    verify_tail(a.output_path, speech_frames)

    print(f"model load   {t_load:6.2f}s   (once per process)")
    print(f"avatar setup {t_setup:6.2f}s   (once per face)")
    print(f"generation   {t_gen:6.2f}s   for {speech_s:.2f}s audio = "
          f"{t_gen / speech_s:.2f}x real-time")
    print(f"output       {a.output_path}  ({speech_frames} frames)")


if __name__ == "__main__":
    main()
