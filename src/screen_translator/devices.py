"""Which processors llama.cpp can run the model on."""

from __future__ import annotations

import ctypes
import re
import sys


def ggml_lib():
    from llama_cpp import _ggml, llama_cpp

    if sys.platform == "win32":
        return _ggml.libggml
    # Wheels hold libggml.dylib and libggml.1.dylib as two copies with separate backend registries;
    # looking symbols up through libllama finds the copy llama.cpp actually uses.
    return llama_cpp._lib


def _ggml_base():
    """Device names live in ggml-base; Windows' GetProcAddress, unlike dlsym, does not search dependent DLLs."""
    if sys.platform != "win32":
        return ggml_lib()
    from llama_cpp import _ggml as bindings

    return ctypes.CDLL(str(bindings.libggml_base_path / "ggml-base.dll"))


def llama_devices() -> list[tuple[str, str, int]]:
    """(name, description, type) per device; type 0 is the CPU, 1 a graphics card, 2 integrated graphics."""
    ggml, base = ggml_lib(), _ggml_base()
    ggml.ggml_backend_dev_count.restype = ctypes.c_size_t
    ggml.ggml_backend_dev_get.argtypes = [ctypes.c_size_t]
    ggml.ggml_backend_dev_get.restype = ctypes.c_void_p
    for fn in (base.ggml_backend_dev_name, base.ggml_backend_dev_description):
        fn.argtypes, fn.restype = [ctypes.c_void_p], ctypes.c_char_p
    base.ggml_backend_dev_type.argtypes, base.ggml_backend_dev_type.restype = [ctypes.c_void_p], ctypes.c_int
    devices = []
    for i in range(ggml.ggml_backend_dev_count()):
        dev = ggml.ggml_backend_dev_get(i)
        name, desc = base.ggml_backend_dev_name(dev), base.ggml_backend_dev_description(dev)
        devices.append((name.decode(), desc.decode(errors="replace"), base.ggml_backend_dev_type(dev)))
    return devices


def model_device(gpu_layers: int = -1) -> str:
    """Where a model loaded with `gpu_layers` runs, e.g. "NVIDIA GeForce RTX 2060 (Vulkan)" or "CPU"."""
    if gpu_layers == 0:
        return "CPU"
    try:
        devices = llama_devices()
    except (OSError, AttributeError, ImportError):
        return ""
    gpus = [(name, desc) for name, desc, kind in devices if kind in (1, 2)]
    if not gpus:
        return "CPU"
    name, desc = gpus[0]
    backend = re.sub(r"\d+$", "", name) or name
    backend = {"MTL": "Metal"}.get(backend, backend)
    return f"{desc.strip()} ({backend})"
