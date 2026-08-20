"""
Blackwell fork: warp_f3d + decode_f3d fused behind one CUDA graph.

WHY. Profiled on an RTX 5090 (driver 591.86, torch 2.11.0+cu130), one frame of
warp+decode issues ~1985 GPU op invocations and spends 20.2 ms of CPU against
~8 ms of actual kernel time -- i.e. the stage is CPU-dispatch-bound, not
compute-bound, which is also why nvidia-smi reports only 24-41% utilisation
during a run. A captured graph replays the whole thing as one launch.

Measured, warp+decode per frame:
    original (host round-trip between the two stages)   24.1 ms
    device-resident + cached coordinate grid            18.4 ms
    CUDA graph replay                                   14.2 ms
Output matches the original path to atol=2/255.

The two stages were also handing an 8.4 MB (1,32,16,64,64) fp32 tensor back to
the host between them -- warp_f3d ended with .float().cpu().numpy() and
decode_f3d began with torch.from_numpy(...).to(device). Fusing removes the round
trip, the fp32 upcast before the copy, one queue hop and one worker thread.

FALLS BACK, LOUDLY IN THE LOGS AND SILENTLY IN BEHAVIOUR. Any of: a non-pytorch
model_type (onnx/tensorrt), no CUDA, a capture failure, or an input whose shape
differs from the captured one -> the original two-call path, same numbers as
upstream. Nothing here changes what a non-Blackwell user gets.
"""

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class WarpDecodeFused:
    def __init__(self, warp_f3d, decode_f3d, device="cuda", use_cuda_graph=True):
        self.warp_f3d = warp_f3d
        self.decode_f3d = decode_f3d
        self.device = device
        self.reason = None

        self.use_cuda_graph = bool(use_cuda_graph)
        if torch is None or not torch.cuda.is_available():
            self.use_cuda_graph, self.reason = False, "no CUDA"
        else:
            wn = getattr(warp_f3d, "warp_net", None)
            dc = getattr(decode_f3d, "decoder", None)
            self._wnet = getattr(wn, "model", None)
            self._dnet = getattr(dc, "model", None)
            if getattr(wn, "model_type", None) != "pytorch" or getattr(dc, "model_type", None) != "pytorch":
                self.use_cuda_graph, self.reason = False, "model_type is not pytorch"
            elif not isinstance(self._wnet, torch.nn.Module) or not isinstance(self._dnet, torch.nn.Module):
                self.use_cuda_graph, self.reason = False, "models are not nn.Modules"

        self._graph = None
        self._static_out = None
        self._sig = None

    # --- the fallback: exactly what the two workers did before ---
    def _unfused(self, f_s, x_s, x_d):
        return self.decode_f3d(self.warp_f3d(f_s, x_s, x_d))

    def _body(self):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=True):
            f_3d = self._wnet(self._in_f_s, self._in_x_s, self._in_x_d)
            pred = self._dnet(f_3d)
            # match decode_f3d's post-processing: [c,h,w] -> [h,w,c], 0..255 float
            return pred[0].permute(1, 2, 0).float().clamp(0, 1) * 255

    def _capture(self, f_s, x_s, x_d):
        self._in_f_s = torch.from_numpy(np.ascontiguousarray(f_s)).to(self.device)
        self._in_x_s = torch.from_numpy(np.ascontiguousarray(x_s)).to(self.device)
        self._in_x_d = torch.from_numpy(np.ascontiguousarray(x_d)).to(self.device)

        # warm up on a side stream -- required before capture, or cuDNN/cuBLAS
        # workspace allocation lands inside the graph
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._body()
        torch.cuda.current_stream().wait_stream(side)

        g = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(g):
                out = self._body()
        self._graph, self._static_out = g, out
        self._sig = (f_s.shape, x_s.shape, x_d.shape, f_s.dtype)

    def prepare(self, f_s, x_s, x_d):
        """Capture EAGERLY, from setup(), while no worker thread is touching the GPU.

        Capturing lazily on the first frame fails intermittently with
        cudaErrorStreamCaptureInvalidated: by then wav2feat (onnxruntime CUDA) and
        audio2motion are issuing work on the same device from other threads, and any
        of that invalidates an in-progress capture. Worse, the failure surfaces
        asynchronously, so the try/except around the capture does not always catch
        it -- the run dies instead of falling back. Capturing at setup time removes
        the race rather than retrying through it.
        """
        if not self.use_cuda_graph or self._graph is not None:
            return
        try:
            self._capture(f_s, x_s, x_d)
        except Exception as e:
            self.use_cuda_graph = False
            self.reason = f"capture failed: {type(e).__name__}: {e}"
            print(f"[WarpDecodeFused] CUDA graph unavailable, using the original "
                  f"path -- {self.reason}", flush=True)

    def __call__(self, f_s, x_s, x_d):
        if not self.use_cuda_graph:
            return self._unfused(f_s, x_s, x_d)

        sig = (f_s.shape, x_s.shape, x_d.shape, f_s.dtype)
        if self._graph is None:
            try:
                self._capture(f_s, x_s, x_d)
            except Exception as e:
                self.use_cuda_graph = False
                self.reason = f"capture failed: {type(e).__name__}: {e}"
                print(f"[WarpDecodeFused] CUDA graph unavailable, using the original "
                      f"path -- {self.reason}", flush=True)
                return self._unfused(f_s, x_s, x_d)
        elif sig != self._sig:
            # a shape we did not capture: do not silently feed the wrong buffer
            return self._unfused(f_s, x_s, x_d)
        else:
            self._in_f_s.copy_(torch.from_numpy(np.ascontiguousarray(f_s)))
            self._in_x_s.copy_(torch.from_numpy(np.ascontiguousarray(x_s)))
            self._in_x_d.copy_(torch.from_numpy(np.ascontiguousarray(x_d)))

        self._graph.replay()
        return self._static_out.cpu().numpy()
