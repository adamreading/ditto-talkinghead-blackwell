# Ditto on NVIDIA Blackwell (RTX 50-series) — measured notes

Everything here was measured on an **RTX 5090** (sm_120, 32,607 MiB, driver 591.86)
under **WSL2**, on the **PyTorch path with no TensorRT installed**. Numbers are from
the shipped `example/audio.wav` (15.75 s) unless stated. Where something is inferred
rather than measured it says so.

Upstream issues [#66](https://github.com/antgroup/ditto-talkinghead/issues/66)
("TensorRT for BLACKWELL?") and
[#88](https://github.com/antgroup/ditto-talkinghead/issues/88) ("Run
ditto-talkinghead in a Docker with 5090") were both open and unanswered when this
work was done.

---

## 1. It does not run as documented, and the reason is not obvious

`environment.yaml` pins `torch 2.5.1+cu121` and `tensorrt 8.6.1`. Neither supports
Blackwell — cu121 wheels ship no sm_120 kernels, and TensorRT 8.6.1 predates the
architecture. The prebuilt engines are built at
`hardware-compatibility-level=Ampere_Plus`, which **reads as inclusive but is not**:
they were produced by a TensorRT that has never seen an sm_120 device.

**You do not need TensorRT.** HuggingFace ships `ditto_pytorch`, which is
self-contained — both the `.pth` weights and the ONNX detector/landmark/audio
models. That is 2.31 GB of the 6.93 GB repo; the 2.08 GB of `ditto_trt_Ampere_Plus`
and 2.53 GB of `ditto_onnx` are not required.

Working stack (see `requirements-cu130.txt`):

```
torch 2.11.0+cu130   torchvision 0.26.0   torchaudio 2.11.0   Python 3.10
arch_list: ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']
```

`torchaudio` caps at 2.11.0 on the cu130 index while `torch` goes to 2.13.0, so
2.11.0 is the ceiling for a coherent triple.

⛔ **Do not install `onnxruntime-gpu`.** It needs a complete CUDA 12 runtime beside a
CUDA 13 torch and fails to load its provider, falling back to CPU **silently** — the
error names whichever library is missing first (`libcublasLt.so.12`, then
`libcufft.so.11`, …), so it is easy to install two more packages and still be on CPU.
We measured it both ways and there was **no throughput difference**, because the ONNX
stages are not the bottleneck.

## 2. Streaming works without TensorRT — and that is undocumented

Three configs ship. Only one enables streaming:

| config | `online_mode` |
|---|---|
| `v0.4_hubert_cfg_pytorch.pkl` | False |
| `v0.4_hubert_cfg_trt.pkl` | False |
| `v0.4_hubert_cfg_trt_online.pkl` | **True** — TensorRT only |

So streaming *appears* to require TensorRT. It does not. `online_mode` is a plain
flag in `default_kwargs`, `StreamSDK.run_chunk` is backend-agnostic Python, and the
streaming audio encoder it needs (`aux_models/hubert_streaming_fix_kv.onnx`) is
already in `ditto_pytorch`. `scripts/make_pytorch_online_cfg.py` flips it.

Measured in streaming mode, 200 ms hops:

```
model load      4.72 s    once per process
avatar setup    1.08 s    once per face  (cacheable if the avatar is fixed)
chunk submit    109.8 ms mean, 173-202 ms p95, per 200 ms hop
feed loop       0.55x real-time          ← audio ingestion is ~2x faster than speech
```

`setup_Nd(N_d)` wants a total frame count, which a live stream does not know. Reading
it, `N_d` only drives end-of-clip eye-open and fade alphas — **cosmetic**. An estimate
is safe.

## 3. ⛔ The streaming path silently truncates the end of every clip

`inference.py`'s online branch feeds chunks then calls `close()` immediately, losing
the frames still in flight:

| tail padding | frames out |
|---|---|
| none | 340 (**54 short**) |
| 0.6 s | 355 |
| 1.2 s | 370 |
| 2.4 s | 400 |

Exactly 25 frames recovered per second of padding — 1:1 with the framerate, so a fixed
pipeline depth rather than a rate mismatch. The visible symptom is **lipsync drifting
off in the final ~2 s**, which reads as a model-quality problem and is not one.

`chunksize=(3,5,2)` does not predict 54, and the depth is not derivable from the
published config, so `inference_stream.py` does not hardcode a formula. It
**over-flushes and then trims** to the frame count the audio implies. Verified: 394
frames out for a 15.75 s clip, matching the offline path exactly.

The flush costs time (18.75 s of audio processed for 15.75 s of video → 1.03x becomes
1.42x). In continuous conversation the *next* utterance flushes the previous one, so
only the final utterance pays.

Separately: `inference.py` muxes audio with ffmpeg **after** `SDK.close()`, outside the
SDK. Call `close()` yourself and you get a silent `<output>.tmp.mp4` and no final file,
with no error raised.

## 4. Where the time actually goes

Per-stage, threaded, so shares exceed 100% (stages run concurrently):

| stage | total | calls | ms/call | share of wall |
|---|---|---|---|---|
| **audio2motion** | **29.32 s** | 6 | 4886 | **95.3%** |
| decode_f3d | 23.51 s | 340 | 69.1 | 76.4% |
| warp_f3d | 14.86 s | 340 | 43.7 | 48.3% |
| motion_stitch | 1.67 s | 340 | 4.9 | 5.4% |
| putback | 1.15 s | 340 | 3.4 | 3.7% |
| writer | 0.82 s | 340 | 2.4 | 2.7% |

`audio2motion` is the critical path and it is a **diffusion model** at
`sampling_timesteps = 50`. That single config value dominates everything:

| steps | wall | vs real-time | mean pixel diff vs 50-step | motion retained |
|---|---|---|---|---|
| 50 | 29.95 s | 1.90x | 0.00 | 2.45 |
| 20 | 21.99 s | 1.40x | 5.24 | 2.34 |
| 10 | 19.34 s | 1.23x | 5.97 | 2.36 |
| 5 | 16.27 s | **1.03x** | 6.45 | 2.31 |

Divergence **plateaus**: 50→20 costs 5.24, and 20→5 adds only 1.21 more. Frame-to-frame
motion magnitude holds at 2.31–2.45 throughout, so the face does not go stiff at low
step counts — it is a different sample of similar liveliness, not a degraded one.
Judge it by eye; the metric only says "not collapsed".

**Implication for optimisation work:** attention kernels, `torch.compile` and TRT would
all have targeted `decode_f3d`, which was never the constraint. Below ~5 steps the
floor becomes `decode_f3d` (69 ms/frame) and `warp_f3d` (44 ms/frame); 25 fps needs
40 ms/frame, so that is a ~1.7x ask, not a 10x one.

## 5. Two things that are better than expected

**Output resolution is free.** 512×512, 768×768 and 1432×1432 all cost the same
(37.0 s / 36.2 s / 36.2 s). The pipeline works at a fixed internal resolution and the
warp/decode to source size is cheap. A large avatar costs nothing.

**VRAM is released cleanly.** Peak `+2.4 GB` over baseline, back to baseline within
8 s of every run, with no special handling. Useful if the GPU is shared.

## 6. Not verified here

- TensorRT on Blackwell. Engines would need rebuilding from `ditto_onnx` with a
  TensorRT that supports sm_120. Not attempted.
- `torch.compile`, fp16/bf16 or fused-attention variants on `decode_f3d`/`warp_f3d`.
- Lipsync *quality* at low step counts. Only measured as pixel divergence and motion
  magnitude, which cannot tell you whether the mouth matches the phonemes.
- Anything on Windows, or on any GPU other than a 5090.
