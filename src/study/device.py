"""
Accelerator helpers so the study runs on CUDA GPUs, TPUs (torch_xla) or CPU with one code path.

On a TPU: bf16 autocast, no 4-bit weights, no GradScaler, fixed batch shapes (see PromptBuilder.pad_to),
and `step_done()` after each optimizer step to cut the lazy XLA graph.
"""
import contextlib

import torch


def get_device(prefer: str = "auto") -> str:
    if prefer in ("auto", "xla"):
        try:
            import torch_xla.core.xla_model as xm  # noqa: F401
            if prefer == "xla" or not torch.cuda.is_available():
                return "xla"
        except ImportError:
            if prefer == "xla":
                raise
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def torch_device(device: str):
    if device == "xla":
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    return torch.device(device)


def is_xla(device) -> bool:
    return str(device).startswith("xla")


def autocast(device, dtype):
    if str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=dtype)
    if is_xla(device):
        return torch.autocast("xla", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def step_done(device):
    if is_xla(device):
        import torch_xla.core.xla_model as xm
        xm.mark_step()
