import torch


def select_device() -> torch.device:
    """Choose the best available accelerator, with CPU as the universal fallback."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    # MPS is only available in some PyTorch builds and on supported Apple
    # hardware; avoid assuming that the backend exists on other platforms.
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")
