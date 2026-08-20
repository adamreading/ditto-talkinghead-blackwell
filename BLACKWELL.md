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

**Implication for optimisation work:** at 50 steps `audio2motion` dominates and nothing
else is worth touching. At the 5 steps this fork defaults to, the floor becomes
`decode_f3d` and `warp_f3d` — and that ~1.7x ask is what sections 7 and 8 close.

## 5. Two things that are better than expected

**Output resolution is free.** 512×512, 768×768 and 1432×1432 all cost the same
(37.0 s / 36.2 s / 36.2 s). The pipeline works at a fixed internal resolution and the
warp/decode to source size is cheap. A large avatar costs nothing.

**VRAM is released cleanly.** Peak `+2.4 GB` over baseline, back to baseline within
8 s of every run, with no special handling. Useful if the GPU is shared.

## 6. Not verified here

- TensorRT on Blackwell. Engines would need rebuilding from `ditto_onnx` with a
  TensorRT that supports sm_120. Not attempted.
- `torch.compile` and fused-attention variants. (CUDA graphs on
  `warp_f3d`+`decode_f3d` ARE now measured — see section 8.)
- Lipsync *quality* at low step counts. Only measured as pixel divergence and motion
  magnitude, which cannot tell you whether the mouth matches the phonemes.
- Anything on Windows, or on any GPU other than a 5090.

---

## 7. ⛔ Two dependency traps that make every measurement a lie

**`nvidia-cudnn-cu12` overwrites torch's cu13 cuDNN, in place.** Both wheels install to
`nvidia/cudnn/lib/`, so the cu12 one replaces the files torch 2.11.0+cu130 shipped
with. This is not a library-search-order problem — `LD_LIBRARY_PATH` cannot fix it,
the files are gone. Symptom: `torch.backends.cudnn.version()` reports `92400` under a
cu130 build, and nothing warns. Anything you install that wants CUDA 12 (most
commonly `onnxruntime-gpu` on Python 3.10) drags it in.

Recovery, measured — note torch will not import between the two commands:

```bash
uv pip uninstall nvidia-cudnn-cu12
uv pip install --reinstall-package nvidia-cudnn-cu13 nvidia-cudnn-cu13==9.19.0.56
```

Wall time was *unaffected* by the contamination here (17.42 s clean vs 17.45 s dirty),
so this is a correctness trap for your measurements rather than a performance one.

**`onnxruntime-gpu` on Python 3.10 silently runs on the CPU.** The last cp310 wheel is
1.23.2, a CUDA 12 build; against a cu13 torch it fails to load its provider and falls
back without raising. The warning text changes depending on which library it misses
first, which makes it easy to read as noise.

⇒ **Use Python >= 3.11.** `onnxruntime-gpu` 1.26+ is itself a CUDA 13 build
(`nvidia-cuda-runtime~=13.0`, `nvidia-cufft~=12.0`, `nvidia-cudnn-cu13~=9.0`), so it
shares torch cu130's stack with nothing to clobber:

```bash
uv venv --python 3.12 && uv pip install "onnxruntime-gpu[cuda,cudnn]==1.29.0"
```

Verify it took, rather than trusting the absence of an error:

```python
sdk.wav2feat.w2f.hubert.model.session.get_providers()
# ['CUDAExecutionProvider', 'CPUExecutionProvider']   <- not just ['CPUExecutionProvider']
```

## 8. What actually made it real-time (1.85x, no TensorRT)

### The pipeline was CPU-dispatch-bound, not GPU-bound

Profiled per frame of `warp_f3d`+`decode_f3d` on the clean cu13 stack:

```
GPU kernel time                      ~8 ms
CPU time                            20.18 ms
wall                                20.64 ms
distinct GPU op invocations/frame     1985
nvidia-smi utilisation during a run   24-41%
```

CPU time equals wall time. ~2000 individual op launches per frame, and Python/aten
dispatch is the cost. This is why a TensorRT engine or a GridSample3D plugin would
have helped: not because the kernels are slow, but because collapsing the launches
is. **`grid_sample` itself is 9.3 ms out of 3.9 s — 0.2%.**

### Fix 1 — hubert was inline on the critical path

`wav2feat` (the streaming ONNX audio encoder) ran inside `run_chunk`, on the caller's
thread, ahead of every queue. On the CPU provider that is 63.9 ms per 200 ms chunk.
Worse, it costs **11.10 s interleaved with the GPU workers against 5.24 s batched
alone** — under contention it runs at half speed. Now in its own worker, so a live
audio producer is never blocked by inference. On CUDA it drops to 13.6 ms.

### Fix 2 — a per-frame host round trip between two GPU stages

`warp_f3d` ended with `.float().cpu().numpy()`; `decode_f3d` began with
`torch.from_numpy(...).to(device)`. The tensor between them is
`(1,32,16,64,64)` — 8.4 MB, pulled to the host and pushed straight back, per frame,
after an fp32 upcast that doubled it. Worth 1.13x on its own; the bigger cost was the
implied device sync, which drains the GPU between stages.

### Fix 3 — the coordinate grid was rebuilt on the host every forward

`make_coordinate_grid` did `torch.arange(...)` on the **CPU** then copied to device,
every call, for a grid that depends only on `(spatial_size, dtype, device)` — all
constant for a run. Now built on device and cached (786 KB, held for the process).
`dense_motion`'s background `zeros` had the same shape of bug. Worth 1.31x, and it is
what made the next fix possible at all: `torch.cuda.graph` refuses to capture through
a non-pinned host->device copy.

### Fix 4 — warp+decode behind one CUDA graph

`core/atomic_components/warp_decode_fused.py`. One replay instead of ~2000 launches,
one worker thread instead of two, one queue hop fewer.

Per frame, warp+decode:

| path | ms/frame |
|---|---|
| original (host round trip) | 24.1 |
| device-resident + cached grid | 18.4 |
| **CUDA graph replay** | **14.2** |

Output matches the eager path to `atol=2/255` on identical inputs.

⛔ **Capture eagerly, from `setup()`.** Capturing lazily on the first frame fails
intermittently with `cudaErrorStreamCaptureInvalidated` — by then `wav2feat`
(onnxruntime CUDA) and `audio2motion` are issuing work on the same device from other
threads. The failure surfaces asynchronously, so a `try/except` around the capture
does not reliably catch it and the run dies instead of falling back. At `setup()` time
every worker is parked on an empty queue.

### End to end

15.75 s of audio, 394 frames, 5 sampling steps, interleaved A/B (alternating runs, because
runs get monotonically faster as the GPU warms and a blocked A/B would prove whatever ran
second):

| | run 1 | run 2 | run 3 | mean |
|---|---|---|---|---|
| upstream path (`DITTO_FUSE_WARP_DECODE=0`) | 16.55 s | 13.87 s | 18.65 s | **16.36 s** |
| fused (default) | 9.59 s | 9.57 s | 7.36 s | **8.84 s** |

**1.85x, and 41-54 fps against the 25 fps real-time bar.** The fused path is also far
more stable — 7.4-9.6 s against 13.9-18.7 s — because once it is no longer
dispatch-bound the GIL scheduling noise goes with it.

Offline path (`inference.py`, 50 steps, includes the 4.7 s model load): 26.71 s -> 18.41 s, 1.45x.

On a real 1672x941 source: **8.99 s wall, 43.8 fps, 0.570x real-time.**

⚠️ **Frame-exact comparison between runs is meaningless here** and it nearly produced a
false regression report. `audio2motion` is a diffusion sampler with no fixed seed, so
two *identical* unfused runs differ by mean 2.451 / max 190 — **more** than fused vs
unfused (mean 2.087 / max 185). Any correctness claim needs the same-run-twice control,
or deterministic single-stage inputs as used for the atol figure above.

### Still open

- `writer` is 12.3 ms/frame of CPU (ffmpeg pipe at source resolution). Not on the
  critical path yet, but it is next if the rest gets faster. NVENC untried.
- `audio2motion` at 6.54 s for 6 batched calls becomes the limiter once warp+decode
  is graphed. Not attacked.
- `channels_last` is a dead end as written: `dense_motion.py:100` does
  `prediction.view(bs, -1, h, w)`, which raises on a non-contiguous layout. Untested
  beyond that, and low priority now that the stage is not compute-bound.
