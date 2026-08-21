#!/usr/bin/env python3
"""LIVE streaming talking head: you speak, it answers, frames arrive as the voice does.

THE POINT, AND WHY THE EARLIER GRADIO ATTEMPT WAS WRONG. Gradio renders a complete mp4
and hands you a file. That is file playback with extra steps and it cannot answer "how
does live feel". Here the audio and the frames are pushed out AS THEY ARE PRODUCED.

HOW IT STREAMS, end to end:

  mic ──► whisper ──► qwen3.5-4b (streamed, per SENTENCE)
                            │
                            ▼
                     VibeVoice  stream=true, response_format=pcm
                     (measured live: TTFB 0.485s, RTF 0.54, PCM16 mono 24kHz)
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
        Ditto.run_chunk            ffmpeg audio pipe
        (200 ms hops)                     │
              │                           │
        SDK.writer intercept              │
              ▼                           ▼
        ffmpeg video pipe ────► fragmented MP4 ────► <video> in the browser

WHY ONE ffmpeg AND NOT TWO STREAMS. Video and audio go into a single fragmented-MP4
muxer, so the container carries the timestamps and lip sync CANNOT drift. Two
independent streams (MJPEG + <audio>) would drift, and drift is the one defect that
makes a talking head worthless.

WHY THE 4.4 s LIVE-MIC FLOOR DOES NOT BITE. Ditto's motion model has a 3.2 s window, so
animating a live microphone runs ~4.4 s behind it. But we never animate the USER's voice --
we animate the REPLY, and VibeVoice generates that at RTF 0.54, i.e. 1.85x faster than
real time. So Ditto is fed faster than playback and keeps pulling ahead after the
initial window fills. Numbers in BLACKWELL.md section 3.

ZERO NEW DEPENDENCIES: stdlib http.server + urllib + ffmpeg. Deliberate -- reaching for
a web framework here is what produced the wrong thing twice.

Configure entirely by environment -- see the table at the bottom of README.md.

Run:  python examples/live_stream_server.py
      then open http://127.0.0.1:7870
"""
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soxr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("DITTO_REPO", os.path.dirname(HERE))
sys.path.insert(0, REPO)

WHISPER_URL = os.environ.get("WHISPER_URL", "http://127.0.0.1:9000/v1/audio/transcriptions")
TTS_URL = os.environ.get("TTS_URL", "http://127.0.0.1:8881/v1/audio/speech")
TTS_VOICE = os.environ.get("TTS_VOICE", "en-Finn_man")
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8090/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "local-model")
FACE = os.environ.get("FACE_IMAGE", os.path.join(REPO, "example", "image.png"))
PORT = int(os.environ.get("PORT", "7870"))
# Residual A/V shift. Positive = delay the VIDEO by holding its first frame (use when
# the mouth runs early). Measured +6 frames early at LEAD_FRAMES=53 on fixed audio.
#
# ⛔ NOT ffmpeg -itsoffset: on a piped rawvideo input it produced 3521 frames for a
# ~15s utterance and destroyed the correlation (r 0.34 -> 0.06). Repeating the first
# frame is the boring version that works.
AV_OFFSET_MS = int(os.environ.get("AV_OFFSET_MS", "240"))
# Target peak for the audio fed to Ditto (0 = off). Playback loudness is handled
# separately by speechnorm in the muxer; this is purely what the MODEL hears.
DITTO_GAIN = float(os.environ.get("DITTO_GAIN", "0"))
STEPS = int(os.environ.get("STEPS", "15"))

TTS_SR = 24000          # VibeVoice streaming PCM16 mono
DITTO_SR = 16000        # what hubert wants
FPS = 25
CHUNK = (3, 5, 2)
HOP = CHUNK[1] * 640                                   # 3200 samples @16k = 200 ms
WINDOW = int(sum(CHUNK) * 0.04 * DITTO_SR) + 80        # 6480

# ⛔ THE PIPELINE SWALLOWS ITS FIRST ~53 FRAMES — AT THE FRONT, NOT THE TAIL.
# Emitted frames are always fed_frames - 53 (measured: 437->385, 618->565, 743->690),
# and an onset probe (2s silence / 2s speech / 2s silence) put mouth activity at
# 0.0-2.8s against speech at 2.0-4.0s. So emitted frame 0 corresponds to audio at
# ~2.12s: pairing it with t=0 makes the video LEAD BY TWO SECONDS. That was the
# "lip sync is way off". Feeding this much filler FIRST makes the swallow eat filler
# instead of speech. Verified after the fix: aperture 0.222 during speech vs 0.101
# during silence, a 2.20x ratio, previously inverted.
# TUNED EMPIRICALLY, not derived. With 53 the residual offset measured +5 frames
# (+200ms, video early); 48 targets zero. The mechanism story is NOT settled -- an
# onset probe suggested the video led, cross-correlation on real speech says it
# lagged, and the two disagree. What IS measured: this parameter moves the offset
# linearly, and 43 frames of misalignment (r=0.52) is what "lip sync way off" was.
LEAD_FRAMES = 53

SYSTEM = os.environ.get("SYSTEM_PROMPT",
    "You are a helpful assistant speaking aloud through a video avatar. Answer in at "
    "most three short spoken sentences. No markdown, no lists, no emoji, no stage "
    "directions -- every word is read aloud.")

# Scratch space for mic uploads and the audio FIFO. Keep it OUT of the repo when
# running from a checkout.
OUTDIR = os.environ.get("OUTDIR", os.path.join(HERE, "out", "live"))
os.makedirs(OUTDIR, exist_ok=True)


# Credentials and the STT vocabulary hint come from the ENVIRONMENT. Whisper returns
# 401 with no key, and because a typed/verbatim turn skips STT entirely that failure is
# invisible until someone actually speaks -- so it is worth checking on startup.
KEY = os.environ.get("TTS_API_KEY", "")
STT_KEY = os.environ.get("STT_API_KEY", "")
# Optional. A comma-separated vocabulary hint passed to whisper as `prompt`; without it
# whisper mangles proper nouns, which reads as a bad model rather than a missing param.
STT_PROMPT = os.environ.get("STT_PROMPT", "")

from stream_pipeline_online import StreamSDK          # noqa: E402
from inference_stream import speech_filler            # noqa: E402

print("[ditto] loading…", flush=True)
_t = time.time()
SDK = StreamSDK(os.path.join(REPO, "checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch_ONLINE.pkl"),
                os.path.join(REPO, "checkpoints/ditto_pytorch"))
print(f"[ditto] loaded in {time.time()-_t:.2f}s", flush=True)
SDK_LOCK = threading.Lock()     # one pipeline, one turn at a time


# --------------------------------------------------------------------------
# frame sink: intercepts Ditto's writer and pushes raw RGB into ffmpeg
# --------------------------------------------------------------------------
class FrameSink:
    """Replaces SDK.writer. Ditto calls this per finished frame.

    Must survive SDK.close() (which calls writer.close()) because one reply spans
    several sentences and therefore several setup()/close() cycles, all feeding ONE
    ffmpeg process and one video stream.
    """

    def __init__(self, turn):
        self.turn = turn

    def __call__(self, frame_rgb, fmt="rgb"):
        # ⛔ DROP THE PADDING FRAMES. Each sentence is fed 3s of speech-shaped filler so
        # the mouth does not come to rest early (BLACKWELL.md section 3), and Ditto
        # renders frames for it. Letting those into the muxer put the video 1.2s ahead
        # of the audio on a 3-sentence reply -- and because it accumulates per sentence,
        # lip sync drifts further with every one. Measured: video 13.64s vs audio 12.44s.
        #
        # The rule is "only emit a frame the arrived AUDIO justifies". Ditto lags the
        # TTS anyway, so in practice this only ever trims filler.
        if self.turn.frames_out >= self.turn.frames_allowed():
            return
        self.turn.frames_out += 1
        self.turn.push_video(np.ascontiguousarray(frame_rgb, dtype=np.uint8).tobytes(),
                             frame_rgb.shape)

    def close(self):
        pass            # the turn owns ffmpeg's lifetime, not the SDK

    def update(self, *a, **k):
        pass


class Pbar:
    def update(self, *a, **k):
        pass

    def close(self):
        pass


# --------------------------------------------------------------------------
# one turn = one HTTP video stream
# --------------------------------------------------------------------------
class Turn:
    def __init__(self, heard):
        self.id = uuid.uuid4().hex[:12]
        self.heard = heard
        self.reply = ""
        self.out = queue.Queue(maxsize=512)   # muxed fMP4 bytes for the browser
        self.log = []
        self.t0 = time.time()
        self.first_frame_at = None
        self.done = False
        self._ff = None
        self._ffv = None
        self._ffa = None
        self._size = None
        self._pump = None
        self._afifo = None
        self._abuf = bytearray()
        self.vq = queue.Queue(maxsize=64)
        self.aq = queue.Queue(maxsize=512)
        self.pcm_bytes = 0     # 24k PCM16 mono delivered to the muxer so far
        self.frames_out = 0
        self._hold = 0

    def frames_allowed(self):
        """Frames the audio delivered so far entitles us to emit."""
        return int(self.pcm_bytes / 2 / TTS_SR * FPS)

    def note(self, s):
        self.log.append(f"{time.time()-self.t0:6.2f}s  {s}")
        print(f"[turn {self.id}] {s}", flush=True)

    # --- ffmpeg, started lazily so the frame size comes from a real frame ---
    def _start_ffmpeg(self, shape):
        """Video on stdin, audio on a FIFO.

        NOT pipe:3/pipe:4 with pass_fds: pass_fds keeps the descriptors open but does
        NOT renumber them, so ffmpeg's pipe:3 would refer to whatever fd 3 happens to
        be in the child, not our pipe. And preexec_fn to dup2 is unsafe here -- this is
        a threaded server and fork-in-a-thread is exactly where that bites. One stdin
        plus one FIFO needs neither.
        """
        h, w = shape[0], shape[1]
        h -= h % 2
        w -= w % 2
        self._size = (w, h)
        self._afifo = os.path.join(OUTDIR, f"{self.id}.apcm")
        if os.path.exists(self._afifo):
            os.remove(self._afifo)
        os.mkfifo(self._afifo)
        cmd = ["ffmpeg", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", str(FPS), "-i", "pipe:0",
               "-f", "s16le", "-ar", str(TTS_SR), "-ac", "1", "-i", self._afifo,
               "-map", "0:v", "-map", "1:a",
               "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
               # Profile PINNED to what -preset ultrafast actually emits. Asking for
               # main and getting "Constrained Baseline" is what happened, and the MSE
               # codec string must match the bitstream or appendBuffer fails on Safari
               # while isTypeSupported still says yes. Baseline is also the widest
               # Safari support. 1672x940 exceeds level 3.1, hence 4.0 (0x28).
               "-profile:v", "baseline", "-level:v", "4.0",
               "-pix_fmt", "yuv420p", "-g", "25",
               # VibeVoice output is quiet in absolute terms -- measured peak 0.336 /
               # rms 0.039 (~-28 dBFS) where speech normally sits near -18, with ~3x
               # headroom before clipping. NOT the documented streaming-normalisation
               # drop: the streaming path is actually LOUDER than the buffered one
               # (rms 0.0387 vs 0.0302), so that hypothesis was wrong. speechnorm is
               # purpose-built and causal (no lookahead), so it costs no latency:
               # measured peak 0.950 / rms 0.108. dynaudnorm did nothing here (0.0336).
               "-af", "speechnorm=e=12.5:r=0.0001:l=1",
               "-c:a", "aac", "-b:a", "96k",
               "-f", "mp4",
               "-movflags", "frag_keyframe+empty_moov+default_base_moof+omit_tfhd_offset",
               "-frag_duration", "200000",
               "pipe:1"]
        self._ff = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self._ffv = self._ff.stdin
        # opening the FIFO for write blocks until ffmpeg opens it for read, which it
        # only does after input 0 -- so do it off-thread and buffer meanwhile
        def open_audio():
            self._ffa = open(self._afifo, "wb")
        threading.Thread(target=open_audio, daemon=True).start()

        def pump():
            try:
                while True:
                    b = self._ff.stdout.read(16384)
                    if not b:
                        break
                    self.out.put(b)
            finally:
                self.out.put(None)
        self._pump = threading.Thread(target=pump, daemon=True)
        self._pump.start()
        self.note(f"ffmpeg up, {w}x{h}")

    def _drain(self, q, get_fh, name):
        """Own thread per stream. This is what breaks the deadlock.

        ffmpeg will not drain the video pipe until it can also read audio (the muxer
        interleaves). If audio is pushed by the same thread that feeds Ditto, and
        Ditto's writer blocks on the video pipe, nothing can move: ffmpeg waits for
        audio, audio waits for Ditto, Ditto waits for ffmpeg. Measured -- it wedged.
        Separate threads mean PCM reaches the muxer the moment TTS emits it, whatever
        Ditto is doing.
        """
        while True:
            b = q.get()
            if b is None:
                break
            fh = get_fh()
            while fh is None and not self.done:
                time.sleep(0.01)
                fh = get_fh()
            if fh is None:
                break
            try:
                fh.write(b)
                fh.flush()
            except (BrokenPipeError, ValueError, OSError):
                break
        try:
            fh = get_fh()
            fh and fh.close()
        except Exception:
            pass

    def _start_drains(self):
        threading.Thread(target=self._drain, args=(self.vq, lambda: self._ffv, "v"),
                         daemon=True).start()
        threading.Thread(target=self._drain, args=(self.aq, lambda: self._ffa, "a"),
                         daemon=True).start()

    def push_video(self, raw, shape):
        if self._ff is None:
            self._start_ffmpeg(shape)
            self._start_drains()
            if self._abuf:
                self.aq.put(bytes(self._abuf))
                self._abuf = bytearray()
        if self.first_frame_at is None:
            self.first_frame_at = time.time() - self.t0
            self._hold = max(0, round(AV_OFFSET_MS / 1000.0 * FPS))
            self.note(f"\u25c4\u25c4 FIRST FRAME  ({self.first_frame_at:.2f}s)")
        w, h = self._size
        if (shape[1], shape[0]) != (w, h):
            a = np.frombuffer(raw, dtype=np.uint8).reshape(shape)[:h, :w]
            raw = np.ascontiguousarray(a).tobytes()
        if self._hold:
            for _ in range(self._hold):      # delay the video by holding frame 0
                self.vq.put(raw)
                self.frames_out += 1
            self._hold = 0
        self.vq.put(raw)

    def push_audio(self, pcm):
        """Called straight from the TTS read loop -- never gated on Ditto."""
        self.pcm_bytes += len(pcm)
        if self._ff is None:
            self._abuf.extend(pcm)
        else:
            self.aq.put(bytes(pcm))

    def finish(self):
        if self._ff is None:
            self.out.put(None)
            self.done = True
            return
        if self._abuf:
            self.aq.put(bytes(self._abuf))
            self._abuf = bytearray()
        self.vq.put(None)
        self.aq.put(None)
        try:
            self._ff.wait(timeout=120)
        except Exception:
            self._ff.kill()
        if self._afifo and os.path.exists(self._afifo):
            try:
                os.remove(self._afifo)
            except OSError:
                pass
        self.done = True


TURNS = {}


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
def _post(url, data, headers, timeout=300):
    req = urllib.request.Request(url, data=data, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


def stt(wav_path):
    boundary = "----ditto" + uuid.uuid4().hex
    with open(wav_path, "rb") as f:
        blob = f.read()
    def field(name, value):
        return (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\""
                f"\r\n\r\n{value}\r\n").encode()

    body = field("model", "whisper-1") + field("language", "en")
    if STT_PROMPT:
        body += field("prompt", STT_PROMPT)
    body += (f"--{boundary}\r\nContent-Disposition: form-data; "
             f"name=\"file\"; filename=\"in.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
             ).encode() + blob + f"\r\n--{boundary}--\r\n".encode()
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if STT_KEY:
        headers["Authorization"] = f"Bearer {STT_KEY}"
    r = _post(WHISPER_URL, body, headers, timeout=120)
    return (json.loads(r.read()).get("text") or "").strip()


SENTENCE_END = re.compile(r"(?<=[.!?])(\s+|$)")


def llm_sentences(user_text):
    """Stream the reply, yielding each SENTENCE the moment its terminator arrives.

    Per-sentence is what makes it feel live: sentence 1 is already being spoken and
    animated while the model is still writing sentence 2.
    """
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text}]
    # ⛔ enable_thinking=False IS LOAD-BEARING. qwen3.5 is a reasoning model: without
    # this it streams `reasoning_content` deltas and never emits `content` at all
    # within a sane token budget (measured: 120 reasoning deltas, zero speech). With
    # it: first content at 0.43s. Thinking out loud is also latency we cannot spend.
    body = json.dumps({"model": LLM_MODEL, "messages": msgs, "stream": True,
                       "temperature": 0.7, "max_tokens": 200,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = _post(LLM_URL, body, {"Content-Type": "application/json"}, timeout=120)
    buf = ""
    for raw in r:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            delta = json.loads(payload)["choices"][0].get("delta", {})
        except Exception:
            continue
        # take `content` ONLY -- never `reasoning_content`. Speaking a model's
        # internal monologue aloud is worse than saying nothing.
        buf += delta.get("content") or ""
        while True:
            m = SENTENCE_END.search(buf)
            if not m:
                break
            s, buf = buf[:m.start()].strip(), buf[m.end():]
            if s:
                yield s
    if buf.strip():
        yield buf.strip()


def tts_stream(text):
    """Open VibeVoice's streaming PCM endpoint and yield chunks as they arrive."""
    body = json.dumps({"model": "tts-1", "voice": TTS_VOICE, "input": text,
                       "stream": True, "response_format": "pcm"}).encode()
    t0 = time.time()
    r = _post(TTS_URL, body, {"Content-Type": "application/json",
                              "Authorization": f"Bearer {KEY}"})
    ttfb = None
    while True:
        b = r.read(8192)
        if not b:
            break
        if ttfb is None:
            ttfb = time.time() - t0
            yield ("ttfb", ttfb)
        yield ("pcm", b)


class Session:
    """ONE Ditto session for the whole turn, not one per sentence.

    Per-sentence setup()/close() cost ~1s each AND applied the 53-frame front
    correction per sentence. One session per turn means the swallow happens once, the
    lead padding is applied once, and the resampler keeps its filter state across the
    whole reply instead of restarting mid-speech at every sentence boundary.
    """

    def __init__(self, turn):
        self.turn = turn
        self.rs = soxr.ResampleStream(TTS_SR, DITTO_SR, 1, dtype="float32", quality="VHQ")
        self.all16 = np.zeros((0,), dtype=np.float32)   # kept for the tail filler
        self.peak = 0.0
        SDK.setup(FACE, os.path.join(OUTDIR, f"{turn.id}.mp4"), sampling_timesteps=STEPS)
        SDK.writer = FrameSink(turn)
        SDK.writer_pbar = Pbar()
        SDK.setup_Nd(N_d=int(600 * FPS))       # cosmetic only; an estimate is safe
        self.feed = np.concatenate([
            np.zeros((CHUNK[0] * 640,), dtype=np.float32),                     # priming
            np.zeros((int(LEAD_FRAMES / FPS * DITTO_SR),), dtype=np.float32)],  # the lead
            0)
        self._drain()

    def _drain(self):
        while len(self.feed) >= WINDOW:
            SDK.run_chunk(np.ascontiguousarray(self.feed[:WINDOW]), CHUNK)
            self.feed = self.feed[HOP:]

    def add(self, pcm24):
        a = np.frombuffer(pcm24, dtype=np.int16).astype(np.float32) / 32768.0
        # Ditto's hubert was hearing -28 dBFS: speechnorm runs in the MUXER, i.e. after
        # the pipeline, so it only ever fixed playback. hubert is trained on normal
        # speech levels, so quiet input is a plausible source of mushy visemes. Causal
        # gain toward a target peak, tracked across chunks (a per-chunk normalise would
        # pump). Costs nothing measurable.
        if DITTO_GAIN:
            self.peak = max(self.peak * 0.995, float(np.abs(a).max()) if len(a) else 0.0)
            if self.peak > 1e-4:
                a = np.clip(a * min(DITTO_GAIN / self.peak, 8.0), -1.0, 1.0)
        x = self.rs.resample_chunk(a).astype(np.float32)
        self.feed = np.concatenate([self.feed, x], 0)
        self.all16 = np.concatenate([self.all16, x], 0)
        self._drain()

    def close(self):
        self.feed = np.concatenate(
            [self.feed, self.rs.resample_chunk(np.zeros((0,), np.float32), last=True)], 0)
        # speech-shaped tail, never silence -- silence brings the mouth to rest early
        self.feed = np.concatenate([self.feed, speech_filler(self.all16, 3.0, DITTO_SR)], 0)
        self._drain()
        SDK.close()


def run_turn(turn, user_text, verbatim=False):
    """verbatim=True speaks the text as given, no LLM.

    Exists so alignment can be tuned against IDENTICAL audio -- otherwise every
    measurement lands on a different LLM reply and you are fitting noise.
    """
    sess = None
    try:
        with SDK_LOCK:
            sess = Session(turn)
            spoken = []
            src = [t.strip() for t in re.split(SENTENCE_END, user_text) if t.strip()] \
                if verbatim else llm_sentences(user_text)
            for i, sentence in enumerate(src, 1):
                turn.note(f"s{i}: {sentence}")
                spoken.append(sentence)
                turn.reply = " ".join(spoken)
                # TTS in its own thread. Read-then-feed in ONE thread made the two
                # costs ADD instead of overlap: measured 1.46x real-time at both 10
                # and 15 diffusion steps, i.e. steps were never the bottleneck -- the
                # serialisation was. Overlapped, per-sentence time is max(TTS, Ditto).
                q = queue.Queue(maxsize=64)

                def produce(sent=sentence, n=i):
                    try:
                        for kind, payload in tts_stream(sent):
                            if kind == "ttfb":
                                turn.note(f"s{n} TTS first audio {payload:.2f}s")
                                continue
                            turn.push_audio(payload)   # straight to the muxer
                            q.put(payload)
                    except Exception as e:
                        turn.note(f"s{n} TTS FAILED: {type(e).__name__}: {e}")
                    finally:
                        q.put(None)

                th = threading.Thread(target=produce, daemon=True)
                th.start()
                while True:
                    payload = q.get()
                    if payload is None:
                        break
                    sess.add(payload)            # Ditto, concurrent with TTS
                th.join(timeout=5)
            sess.close()
            sess = None
            turn.note(f"reply complete, {turn.frames_out} frames, "
                      f"first frame at {turn.first_frame_at}")
    except Exception as e:
        turn.note(f"FAILED: {type(e).__name__}: {e}")
        if sess is not None:
            try:
                SDK.close()
            except Exception:
                pass
    finally:
        turn.finish()


# --------------------------------------------------------------------------
PAGE = r"""<!doctype html><meta charset=utf-8>
<title>Talk to the avatar</title>
<style>
 body{background:#111;color:#ddd;font:14px/1.5 system-ui;margin:0;padding:18px}
 h1{font-size:16px;font-weight:600;margin:0 0 12px}
 video{width:100%;max-width:840px;background:#000;border-radius:8px}
 button{font:15px system-ui;padding:10px 18px;border:0;border-radius:6px;cursor:pointer}
 #rec{background:#c33;color:#fff} #rec.on{background:#3a3}
 #mic{background:#357;color:#fff} button:disabled{opacity:.45}
 select{font:14px system-ui;padding:8px;border-radius:6px;background:#1a1a1a;
        color:#ddd;border:1px solid #444;max-width:60%}
 pre{background:#000;padding:10px;border-radius:6px;max-width:840px;overflow:auto;
     white-space:pre-wrap;font:12px/1.45 ui-monospace}
 .row{display:flex;gap:10px;align-items:center;margin:12px 0;max-width:840px}
 input{flex:1;font:14px system-ui;padding:9px;border-radius:6px;border:1px solid #444;
       background:#1a1a1a;color:#ddd}
</style>
<h1>Talk to the avatar — live streaming avatar</h1>
<video id=v autoplay playsinline controls></video>
<div class=row>
  <button id=mic>enable mic</button>
  <select id=dev style="display:none"></select>
</div>
<div class=row>
  <button id=rec disabled>talk</button>
  <span id=meter style="flex:1;height:10px;background:#222;border-radius:5px;overflow:hidden">
    <i id=bar style="display:block;height:100%;width:0;background:#3a3"></i></span>
</div>
<div class=row>
  <input id=typed placeholder="…or type and press enter">
</div>
<pre id=log>ready</pre>
<script>
// A JS syntax or runtime error used to bind NO handlers at all -- the button simply
// did nothing and the page looked fine. Make it visible.
window.onerror=(m,src,l,c)=>{const el=document.getElementById('log');
  if(el) el.textContent='JS ERROR line '+l+': '+m; return false;};
const v=document.getElementById('v'), rec=document.getElementById('rec'),
      log=document.getElementById('log'), typed=document.getElementById('typed');
let mr, chunks=[], poll, stream=null, ac=null, an=null, raf=null;
const mic=document.getElementById('mic'), dev=document.getElementById('dev'),
      bar=document.getElementById('bar');

// Hold-to-talk was fragile: on Android touchstart gets cancelled by scroll/long-press,
// and asking for the mic INSIDE the gesture handler races the permission prompt. So:
// permission first, explicit device choice, then tap-to-toggle. The level meter exists
// because "no audio" and "no permission" and "wrong device" look identical otherwise.
const MIMES=['video/mp4; codecs="avc1.42C028, mp4a.40.2"',
             'video/mp4; codecs="avc1.42E028, mp4a.40.2"',
             'video/mp4; codecs="avc1.4D4028, mp4a.40.2"',
             'video/mp4'];
let MIME=null, diag=[];
function dg(m){diag.push(m); log.textContent=diag.join('\n');}

async function send(form){
  log.textContent='sending…';
  const r=await fetch('/turn',{method:'POST',body:form});
  const j=await r.json();
  if(j.error){log.textContent='ERROR: '+j.error;return;}
  log.textContent='heard: '+j.heard; start(j.id);
}

// iPad Safari will not play a chunked fMP4 from <video src> -- its media loader wants
// byte-range requests and abandons a chunked response. MSE sidesteps that and works on
// both iPadOS Safari and desktop Chrome with the same fragments the server emits.
async function start(id){
  diag=[];
  const hasMSE='MediaSource' in window;
  MIME = hasMSE ? (MIMES.find(m=>MediaSource.isTypeSupported(m)) || null) : null;
  dg('MSE: '+hasMSE+'  using: '+(MIME||'none supported'));
  clearInterval(poll);
  poll=setInterval(async()=>{
    try{
      const r=await fetch('/log/'+id); const j=await r.json();
      log.textContent=diag.join('\n')+'\n\n'+j.log.join('\n')+(j.reply?'\n\nreply: '+j.reply:'');
      if(j.done) clearInterval(poll);
    }catch(e){}
  },400);
  if(!MIME){
    dg('MSE unusable — falling back to direct src');
    v.src='/stream/'+id; v.load();
    v.play().catch(()=>dg('autoplay blocked - press play'));
    return;
  }
  const ms=new MediaSource();
  v.src=URL.createObjectURL(ms);
  ms.addEventListener('sourceopen', async()=>{
    let sb;
    try{ sb=ms.addSourceBuffer(MIME); }
    catch(e){ dg('addSourceBuffer failed: '+e.message); return; }
    sb.mode='sequence';
    const q=[]; let ended=false;
    const pump=()=>{
      if(sb.updating||!q.length){
        if(ended&&!sb.updating&&!q.length&&ms.readyState==='open'){ try{ms.endOfStream();}catch(e){} }
        return;
      }
      try{ sb.appendBuffer(q.shift()); }catch(e){ dg('appendBuffer failed: '+e.message); }
    };
    sb.addEventListener('updateend', pump);
    sb.addEventListener('error', ()=>dg('SourceBuffer error'));
    v.addEventListener('error', ()=>dg('video error: '+
      (v.error?('code '+v.error.code+' '+(v.error.message||'')):'unknown')));
    let bytes=0, started=false;
    try{
      const res=await fetch('/stream/'+id);
      dg('stream HTTP '+res.status);
      const rd=res.body.getReader();
      for(;;){
        const {done,value}=await rd.read();
        if(done) break;
        bytes+=value.byteLength; q.push(value); pump();
        if(!started){started=true; dg('first bytes received');
          v.play().catch(()=>dg('autoplay blocked - press play'));}
      }
      ended=true; pump(); dg('stream ended, '+bytes+' bytes');
    }catch(e){ dg('fetch failed: '+e.message); }
  });
}

async function openMic(deviceId){
  if(stream) stream.getTracks().forEach(t=>t.stop());
  const c = deviceId ? {audio:{deviceId:{exact:deviceId}}} : {audio:true};
  stream = await navigator.mediaDevices.getUserMedia(c);
  if(!ac){ ac = new (window.AudioContext||window.webkitAudioContext)(); }
  if(ac.state==='suspended') await ac.resume();
  an = ac.createAnalyser(); an.fftSize=512;
  ac.createMediaStreamSource(stream).connect(an);
  const buf=new Uint8Array(an.fftSize);
  cancelAnimationFrame(raf);
  (function tick(){
    an.getByteTimeDomainData(buf);
    let peak=0; for(const b of buf) peak=Math.max(peak, Math.abs(b-128)/128);
    bar.style.width=Math.min(100, peak*260).toFixed(0)+'%';
    bar.style.background = peak>0.02 ? '#3a3' : '#833';
    raf=requestAnimationFrame(tick);
  })();
  rec.disabled=false;
}

mic.onclick=async()=>{
  try{
    await openMic(null);
    const list=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='audioinput');
    dev.innerHTML=list.map((d,i)=>`<option value="${d.deviceId}">${d.label||('input '+(i+1))}</option>`).join('');
    dev.style.display = list.length>1 ? '' : 'none';
    mic.textContent='mic on ('+list.length+' input'+(list.length===1?'':'s')+')';
    log.textContent='mic ready. green bar = audio arriving. tap talk, speak, tap again.';
  }catch(e){ log.textContent='MIC DENIED/UNAVAILABLE: '+e.name+' '+e.message; }
};
dev.onchange=()=>openMic(dev.value).catch(e=>log.textContent='switch failed: '+e.message);

rec.onclick=()=>{
  if(mr && mr.state!=='inactive'){ mr.stop(); return; }
  if(!stream){ log.textContent='press enable mic first'; return; }
  chunks=[];
  mr=new MediaRecorder(stream);
  mr.ondataavailable=e=>{ if(e.data && e.data.size) chunks.push(e.data); };
  mr.onstop=()=>{
    rec.textContent='talk'; rec.classList.remove('on');
    const blob=new Blob(chunks);
    if(blob.size<1200){ log.textContent='recording too short/empty ('+blob.size+' bytes)'; return; }
    const f=new FormData(); f.append('audio', blob, 'in.webm'); send(f);
  };
  mr.start();
  rec.textContent='listening — tap to send'; rec.classList.add('on');
};

typed.onkeydown=e=>{
  if(e.key!=='Enter'||!typed.value.trim())return;
  const f=new FormData(); f.append('text',typed.value.trim());
  typed.value=''; send(f);
};
</script>
"""


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _trace(self):
        """Log method, path, UA and Range. Safari's media loader behaves differently
        from Chrome's and the difference is invisible without this."""
        ua = self.headers.get("User-Agent", "?")
        rng = self.headers.get("Range", "-")
        print(f"[req] {self.command} {self.path}  Range={rng}  UA={ua[:70]}", flush=True)

    def _send(self, code, ctype, body, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._trace()
        if self.path == "/":
            return self._send(200, "text/html; charset=utf-8", PAGE.encode())
        if self.path.startswith("/log/"):
            t = TURNS.get(self.path[5:])
            if not t:
                return self._send(404, "application/json", b'{"error":"no turn"}')
            return self._send(200, "application/json", json.dumps(
                {"log": t.log, "reply": t.reply, "done": t.done}).encode())
        if self.path.startswith("/stream/"):
            t = TURNS.get(self.path[8:])
            if not t:
                return self._send(404, "text/plain", b"no turn")
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                while True:
                    b = t.out.get()
                    if b is None:
                        self.wfile.write(b"0\r\n\r\n")
                        break
                    self.wfile.write(f"{len(b):X}\r\n".encode() + b + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        return self._send(404, "text/plain", b"not found")

    def do_POST(self):
        self._trace()
        if self.path not in ("/turn", "/say"):
            return self._send(404, "text/plain", b"not found")
        verbatim = self.path == "/say"
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n)
        ctype = self.headers.get("Content-Type", "")
        try:
            text = None
            if "boundary=" in ctype:
                b = ("--" + ctype.split("boundary=")[1]).encode()
                for part in raw.split(b):
                    if b'name="text"' in part:
                        text = part.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
                        text = text.decode("utf-8", "ignore").strip()
                    elif b'name="audio"' in part:
                        blob = part.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
                        src = os.path.join(OUTDIR, "mic.webm")
                        wav = os.path.join(OUTDIR, "mic.wav")
                        open(src, "wb").write(blob)
                        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src,
                                        "-ar", "16000", "-ac", "1", wav], check=True)
                        text = stt(wav)
            if not text:
                return self._send(200, "application/json",
                                  b'{"error":"nothing heard or typed"}')
            turn = Turn(text)
            TURNS[turn.id] = turn
            turn.note(f"heard: {text}")
            threading.Thread(target=run_turn, args=(turn, text),
                             kwargs={"verbatim": verbatim}, daemon=True).start()
            return self._send(200, "application/json",
                              json.dumps({"id": turn.id, "heard": text}).encode())
        except Exception as e:
            return self._send(200, "application/json",
                              json.dumps({"error": f"{type(e).__name__}: {e}"}).encode())


def check_page_js():
    """Parse the page's <script> with node before serving it.

    Earned: a stray real newline inside a JS string literal (PAGE is a non-raw Python
    string, so a written \\n becomes an ACTUAL newline) killed the whole script. Every
    handler silently failed to bind -- the button did nothing, the textbox would not
    send, and the page otherwise looked perfect. A syntax error must not be something a
    human discovers by clicking.
    """
    try:
        js = PAGE.split("<script>", 1)[1].rsplit("</script>", 1)[0]
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(js)
            path = f.name
        r = subprocess.run(["node", "--check", path], capture_output=True, text=True)
        os.unlink(path)
        if r.returncode != 0:
            print("⛔ PAGE JAVASCRIPT DOES NOT PARSE — refusing to serve:", flush=True)
            print(r.stderr.strip()[:600], flush=True)
            sys.exit(1)
        print("[check] page JS parses", flush=True)
    except FileNotFoundError:
        print("[check] node not found, page JS NOT verified", flush=True)


if __name__ == "__main__":
    check_page_js()
    print(f"[serve] http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
