> ### Blackwell (RTX 50-series) fork — plus working lip sync and a live streaming server
>
> A fork of [antgroup/ditto-talkinghead](https://github.com/antgroup/ditto-talkinghead) that
> runs on **sm_120 with no TensorRT**, streams, is **faster than real time** on one RTX 5090,
> and — the part that took longest — has **lip sync that is actually in sync**.
>
> **1.85x faster: 8.84 s for 15.75 s of audio (41-54 fps)**, against 16.36 s upstream. All
> stock PyTorch: no engine rebuild, no custom CUDA plugin. The premise was wrong rather than
> the kernels — the pipeline is **CPU-dispatch-bound** (~1985 GPU op launches per frame,
> 20 ms of CPU against 8 ms of kernel time, GPU 24-41% idle). `grid_sample`, the thing
> everyone writes a TensorRT plugin for, is 0.2% of the run.
>
> **Lip sync was ~1.7 s out, and every duration check said it was fine.** Frame counts
> matched, video and audio durations matched, `ffprobe` was clean. What found it was
> measuring lip aperture per frame against the audio envelope: correlation **0.52 at a lag
> of -43 frames** — the mouth confidently doing the right thing at the wrong moment. Now
> **0 frames**. How to spot it in ten seconds, plus the two mechanisms I convinced myself of
> that were both wrong, are in [BLACKWELL.md §3](BLACKWELL.md).
>
> **`examples/live_stream_server.py`** is a working live pipeline: mic -> STT -> streamed LLM
> -> streaming TTS -> Ditto -> fragmented MP4 in the browser, frames pushed out as the voice
> is produced rather than rendered to a file first. Zero dependencies beyond the model stack
> and ffmpeg.
>
> **Read [BLACKWELL.md](BLACKWELL.md)** for every measurement, the method, the two dependency
> traps that silently invalidate benchmarks, and an explicit list of what is *not* verified.
>
> Quick start: `requirements-cu130.txt`, **Python >= 3.11**.
> `DITTO_FUSE_WARP_DECODE=0` restores the upstream two-worker path for A/B.
>
> Upstream's README follows unchanged.

---

<h2 align='center'>Ditto: Motion-Space Diffusion for Controllable Realtime Talking Head Synthesis</h2>

<div align='center'>
    <a href=""><strong>Tianqi Li</strong></a>
    ·
    <a href=""><strong>Ruobing Zheng</strong></a><sup>†</sup>
    ·
    <a href=""><strong>Minghui Yang</strong></a>
    ·
    <a href=""><strong>Jingdong Chen</strong></a>
    ·
    <a href=""><strong>Ming Yang</strong></a>
</div>
<div align='center'>
Ant Group
</div>
<br>
<div align='center'>
    <a href='https://arxiv.org/abs/2411.19509'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
    <a href='https://digital-avatar.github.io/ai/Ditto/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
    <a href='https://huggingface.co/digital-avatar/ditto-talkinghead'><img src='https://img.shields.io/badge/Model-HuggingFace-yellow'></a>
    <a href='https://github.com/antgroup/ditto-talkinghead'><img src='https://img.shields.io/badge/Code-GitHub-purple'></a>
    <!-- <a href='https://github.com/antgroup/ditto-talkinghead'><img src='https://img.shields.io/github/stars/antgroup/ditto-talkinghead?style=social'></a> -->
    <a href='https://colab.research.google.com/drive/19SUi1TiO32IS-Crmsu9wrkNspWE8tFbs?usp=sharing'><img src='https://img.shields.io/badge/Demo-Colab-orange'></a>
</div>
<br>
<div align="center">
    <video style="width: 95%; object-fit: cover;" controls loop src="https://github.com/user-attachments/assets/ef1a0b08-bff3-4997-a6dd-62a7f51cdb40" muted="false"></video>
    <p>
    ✨  For more results, visit our <a href="https://digital-avatar.github.io/ai/Ditto/"><strong>Project Page</strong></a> ✨ 
    </p>
</div>


## 📌 Updates
* [2025.11.12] 🔥🔥 We noticed the community's enthusiasm for open-source training code. [Training code](https://github.com/antgroup/ditto-talkinghead/tree/train) is now available, since there have been multiple versions and limited time to organize, it may differ slightly from the paper version.
* [2025.07.11] 🔥 The [PyTorch model](#-pytorch-model) is now available.
* [2025.07.07] 🔥 Ditto is accepted by ACM MM 2025.
* [2025.01.21] 🔥 We update the [Colab](https://colab.research.google.com/drive/19SUi1TiO32IS-Crmsu9wrkNspWE8tFbs?usp=sharing) demo, welcome to try it. 
* [2025.01.10] 🔥 We release our inference [codes](https://github.com/antgroup/ditto-talkinghead) and [models](https://huggingface.co/digital-avatar/ditto-talkinghead).
* [2024.11.29] 🔥 Our [paper](https://arxiv.org/abs/2411.19509) is in public on arxiv.

 
 ## 🔍 Overview
<!-- This is the **train branch**, containing code for **training the model**. For inference code, please switch to the [`main`](https://github.com/antgroup/ditto-talkinghead) branch. -->

This is the **inference branch**. For training code, please switch to the [`train`](https://github.com/antgroup/ditto-talkinghead/tree/train) branch.



## 🛠️ Installation

Tested Environment  
- System: Centos 7.2  
- GPU: A100  
- Python: 3.10  
- tensorRT: 8.6.1  


Clone the codes from [GitHub](https://github.com/antgroup/ditto-talkinghead):  
```bash
git clone https://github.com/antgroup/ditto-talkinghead
cd ditto-talkinghead
```

### Conda
Create `conda` environment:
```bash
conda env create -f environment.yaml
conda activate ditto
```

### Pip
If you have problems creating a conda environment, you can also refer to our [Colab](https://colab.research.google.com/drive/19SUi1TiO32IS-Crmsu9wrkNspWE8tFbs?usp=sharing). 
After correctly installing `pytorch`, `cuda` and `cudnn`, you only need to install a few packages using pip:
```bash
pip install \
    tensorrt==8.6.1 \
    librosa \
    tqdm \
    filetype \
    imageio \
    opencv_python_headless \
    scikit-image \
    cython \
    cuda-python \
    imageio-ffmpeg \
    colored \
    polygraphy \
    numpy==2.0.1
```

If you don't use `conda`, you may also need to install `ffmpeg` according to the [official website](https://www.ffmpeg.org/download.html).


## 📥 Download Checkpoints

Download checkpoints from [HuggingFace](https://huggingface.co/digital-avatar/ditto-talkinghead) and put them in `checkpoints` dir:
```bash
git lfs install
git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints
```

The `checkpoints` should be like:
```text
./checkpoints/
├── ditto_cfg
│   ├── v0.4_hubert_cfg_trt.pkl
│   └── v0.4_hubert_cfg_trt_online.pkl
├── ditto_onnx
│   ├── appearance_extractor.onnx
│   ├── blaze_face.onnx
│   ├── decoder.onnx
│   ├── face_mesh.onnx
│   ├── hubert.onnx
│   ├── insightface_det.onnx
│   ├── landmark106.onnx
│   ├── landmark203.onnx
│   ├── libgrid_sample_3d_plugin.so
│   ├── lmdm_v0.4_hubert.onnx
│   ├── motion_extractor.onnx
│   ├── stitch_network.onnx
│   └── warp_network.onnx
└── ditto_trt_Ampere_Plus
    ├── appearance_extractor_fp16.engine
    ├── blaze_face_fp16.engine
    ├── decoder_fp16.engine
    ├── face_mesh_fp16.engine
    ├── hubert_fp32.engine
    ├── insightface_det_fp16.engine
    ├── landmark106_fp16.engine
    ├── landmark203_fp16.engine
    ├── lmdm_v0.4_hubert_fp32.engine
    ├── motion_extractor_fp32.engine
    ├── stitch_network_fp16.engine
    └── warp_network_fp16.engine
```

- The `ditto_cfg/v0.4_hubert_cfg_trt_online.pkl` is online config
- The `ditto_cfg/v0.4_hubert_cfg_trt.pkl` is offline config


## 🚀 Inference 

Run `inference.py`:

```shell
python inference.py \
    --data_root "<path-to-trt-model>" \
    --cfg_pkl "<path-to-cfg-pkl>" \
    --audio_path "<path-to-input-audio>" \
    --source_path "<path-to-input-image>" \
    --output_path "<path-to-output-mp4>" 
```

For example:

```shell
python inference.py \
    --data_root "./checkpoints/ditto_trt_Ampere_Plus" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl" \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./tmp/result.mp4" 
```

❗Note:

We have provided the tensorRT model with `hardware-compatibility-level=Ampere_Plus` (`checkpoints/ditto_trt_Ampere_Plus/`). If your GPU does not support it, please execute the `cvt_onnx_to_trt.py` script to convert from the general onnx model (`checkpoints/ditto_onnx/`) to the tensorRT model.

```bash
python scripts/cvt_onnx_to_trt.py --onnx_dir "./checkpoints/ditto_onnx" --trt_dir "./checkpoints/ditto_trt_custom"
```

Then run `inference.py` with `--data_root=./checkpoints/ditto_trt_custom`.


## ⚡ PyTorch Model
*Based on community interest and to better support further development, we are now open-sourcing the PyTorch version of the model.*


We have added the PyTorch model and corresponding configuration files to the [HuggingFace](https://huggingface.co/digital-avatar/ditto-talkinghead). Please refer to [Download Checkpoints](#-download-checkpoints) to prepare the model files.

The `checkpoints` should be like:
```text
./checkpoints/
├── ditto_cfg
│   ├── ...
│   └── v0.4_hubert_cfg_pytorch.pkl
├── ...
└── ditto_pytorch
    ├── aux_models
    │   ├── 2d106det.onnx
    │   ├── det_10g.onnx
    │   ├── face_landmarker.task
    │   ├── hubert_streaming_fix_kv.onnx
    │   └── landmark203.onnx
    └── models
        ├── appearance_extractor.pth
        ├── decoder.pth
        ├── lmdm_v0.4_hubert.pth
        ├── motion_extractor.pth
        ├── stitch_network.pth
        └── warp_network.pth
```

To run inference, execute the following command:

```shell
python inference.py \
    --data_root "./checkpoints/ditto_pytorch" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl" \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./tmp/result.mp4" 
```


## 📧 Acknowledgement
Our implementation is based on [S2G-MDDiffusion](https://github.com/thuhcsi/S2G-MDDiffusion) and [LivePortrait](https://github.com/KwaiVGI/LivePortrait). Thanks for their remarkable contribution and released code! If we missed any open-source projects or related articles, we would like to complement the acknowledgement of this specific work immediately.

## ⚖️ License
This repository is released under the Apache-2.0 license as found in the [LICENSE](LICENSE) file.

## 📚 Citation
If you find this codebase useful for your research, please use the following entry.
```BibTeX
@article{li2024ditto,
    title={Ditto: Motion-Space Diffusion for Controllable Realtime Talking Head Synthesis},
    author={Li, Tianqi and Zheng, Ruobing and Yang, Minghui and Chen, Jingdong and Yang, Ming},
    journal={arXiv preprint arXiv:2411.19509},
    year={2024}
}
```


## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/ditto-talkinghead&type=Date)](https://www.star-history.com/#antgroup/ditto-talkinghead&Date)

---

## Live streaming server — configuration

`examples/live_stream_server.py`. Everything is environment-driven; nothing is hardcoded to
a particular host.

| variable | default | notes |
|---|---|---|
| `WHISPER_URL` | `http://127.0.0.1:9000/v1/audio/transcriptions` | OpenAI-shaped STT |
| `STT_API_KEY` | *(empty)* | **whisper returns 401 without one.** A typed turn skips STT, so this failure is invisible until someone speaks |
| `STT_PROMPT` | *(empty)* | vocabulary hint; without it proper nouns get mangled and it reads as a bad model |
| `TTS_URL` | `http://127.0.0.1:8881/v1/audio/speech` | must support `stream:true` + `response_format:pcm` |
| `TTS_API_KEY` | *(empty)* | |
| `TTS_VOICE` | `en-Finn_man` | |
| `LLM_URL` | `http://127.0.0.1:8090/v1/chat/completions` | OpenAI-shaped, streaming |
| `LLM_MODEL` | `local-model` | |
| `FACE_IMAGE` | `example/image.png` | head-and-shoulders, mouth closed, facing forward |
| `STEPS` | `15` | diffusion steps; nearly free in wall-clock, see BLACKWELL.md |
| `LEAD_FRAMES` | `53` (constant) | alignment lead — **empirically tuned, not derived** |
| `AV_OFFSET_MS` | `240` | residual video delay; positive = delay video |
| `DITTO_GAIN` | `0` (off) | normalise audio into the model. Measured: no effect (r 0.322 -> 0.325) |
| `OUTDIR` | `examples/out/live` | scratch for mic uploads + the audio FIFO; point it outside a checkout |
| `AVATAR_TOKEN` | *(empty = open)* | required as `X-Avatar-Token` on `/turn` and `/say`. **Set it before exposing this anywhere.** |
| `MAX_CONCURRENT_TURNS` | `1` | further requests get 429 rather than queueing GPU work |
| `OUT_HEIGHT` | `0` (native) | scale the output; **1672x940 measures 4.63 Mbit/s, 432p measures 1.39** and lip sync is unaffected |
| `PORT` | `7870` | binds loopback only |

Run it from an env file rather than a private fork of the code — that is the whole point of
the table above. A launcher that keeps credentials off disk:

```bash
set -a; . ./my-avatar.env; set +a          # everything except the secrets
export TTS_API_KEY="$(get-secret tts)"     # from your own secret store, at launch
export STT_API_KEY="$(get-secret stt)"
exec python examples/live_stream_server.py
```

⚠️ The env file is **sourced by bash**, so quote any value containing an apostrophe —
`SYSTEM_PROMPT` with a possessive in it will otherwise fail with `unexpected EOF`.

### Exposing it — read this first

`AVATAR_TOKEN` gates the routes that start work. Without it, anyone who finds the URL can
make your avatar talk and occupy your GPU indefinitely; `MAX_CONCURRENT_TURNS` bounds that
to one turn at a time. Verified: no token 401, wrong token 401, correct token 200, second
concurrent turn 429.

With a token set, **the page itself requires `?t=<token>`** — so a link you share works
and a bare hostname visit gets 403. Verified: bare 403, `?t=` 200.

**`/stream/<id>` and `/log/<id>` are deliberately NOT token-gated.** A browser front end
has to fetch them, and any secret embedded in a page is readable by everyone who loads that
page. They rely on the turn id being an unguessable 96-bit capability instead.

⛔ **The token the built-in page carries is therefore public to that page's viewers.** It is
fine on loopback or a private network (Tailscale, LAN) and is *not* a way to protect a
public deployment. A public front end must authenticate its own viewer — for a Discord
Activity, that means the Embedded App SDK's OAuth handshake, verified server-side against
the guild and channel you expect, which is also the only channel restriction that cannot be
bypassed by guessing a URL.

### Things that cost real time to learn

**Use a reasoning model and it will think instead of speaking.** A qwen3-class model
streamed 120 `reasoning_content` deltas and never emitted one word of `content`. Needs
`chat_template_kwargs: {"enable_thinking": false}`; with it, first content at 0.43 s. Read
`content` only — never `reasoning_content`, or the avatar reads its own monologue aloud.

**Give audio and video their own writer threads.** ffmpeg will not drain the video pipe
until it can also read audio. If audio is pushed by the same thread that feeds the model,
and the model's writer blocks on the video pipe, all three deadlock: ffmpeg waits for audio,
audio waits for the model, the model waits for ffmpeg. It wedges silently.

**Budget frames against arrived audio.** The speech-shaped tail padding generates frames
too. Letting them into the muxer put the video 1.2 s ahead of the audio on a three-sentence
reply, accumulating per sentence. Emitting a frame only when the delivered audio justifies
it makes drift structurally impossible rather than merely absent.

**iPad Safari will not play a chunked fMP4 from `<video src>`** — its media loader wants
byte-range requests and abandons a `Transfer-Encoding: chunked` response. Use MediaSource.
And the codec string must match the bitstream: `-preset ultrafast` silently overrides
`-profile:v main` and emits **Constrained Baseline**, so `avc1.4D4028` fails to append while
`isTypeSupported` still returns true. Check with `ffprobe`, don't assume.

**Frame-exact comparison between two runs is meaningless.** The motion model is a diffusion
sampler with no fixed seed: two *identical* runs differ by mean 2.45 / max 190, which is
MORE than a real change measured at 2.09 / 185. Any correctness claim needs a
same-input-twice control first.
