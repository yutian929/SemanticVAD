"""设备相关的跨平台工具。

原始代码把设备硬编码为 "cuda"，无法在 macOS (Apple Silicon / MPS) 或纯 CPU
机器上运行。本模块提供统一的设备解析与缓存清理入口，使同一份代码可以在
CUDA / MPS / CPU 三种后端上运行。
"""

import torch


def resolve_device(requested: str = "auto") -> str:
    """把配置里的设备名解析为当前机器实际可用的设备。

    Args:
        requested: 配置中写的设备名。可以是 "auto" / "cuda" / "mps" / "cpu"，
            也可以带序号（如 "cuda:0"）。当请求的后端不可用时会自动降级，
            这样同一份 config 可以在不同机器间直接复用。

    Returns:
        实际可用的设备字符串。
    """
    requested = (requested or "auto").strip().lower()
    backend = requested.split(":", 1)[0]

    cuda_ok = torch.cuda.is_available()
    mps_ok = torch.backends.mps.is_available()

    if backend == "auto":
        if cuda_ok:
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"

    if backend == "cuda":
        if cuda_ok:
            return requested
        fallback = "mps" if mps_ok else "cpu"
        print(f"[device] 请求 {requested} 但 CUDA 不可用，回退到 {fallback}")
        return fallback

    if backend == "mps":
        if mps_ok:
            return "mps"
        print("[device] 请求 mps 但 MPS 不可用，回退到 cpu")
        return "cpu"

    return "cpu"


def empty_cache(device: str = None) -> None:
    """按当前后端清理显存缓存；CPU 上是空操作。"""
    backend = (device or "").split(":", 1)[0].lower()

    if backend == "cuda" or (not backend and torch.cuda.is_available()):
        torch.cuda.empty_cache()
    elif backend == "mps" or (not backend and torch.backends.mps.is_available()):
        # MPS 的 empty_cache 在部分 torch 版本中才有
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def asr_device(device: str = None) -> str:
    """返回适合 FunASR / ModelScope 使用的设备名。

    这两个框架对 MPS 的支持并不完整（部分算子会直接报错或静默出错），
    因此在非 CUDA 环境下统一让 ASR 走 CPU，保证结果正确。
    """
    backend = (device or "").split(":", 1)[0].lower()
    if backend == "cuda":
        return device
    return "cpu"
