"""Array backend selection: numpy (CPU) or cupy (GPU).

Every numerical routine in ``sso`` takes ``usegpu: bool`` and obtains its array
module through :func:`xp`.  CuPy is optional; when it is missing or no device is
visible, ``usegpu=True`` raises a clear error instead of silently falling back.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator as _np_LinearOperator
from scipy.sparse.linalg import eigsh as _np_eigsh

try:  # pragma: no cover - depends on hardware
    import cupy as cp
    from cupyx.scipy.sparse.linalg import LinearOperator as _cp_LinearOperator
    from cupyx.scipy.sparse.linalg import eigsh as _cp_eigsh
    HAS_GPU = bool(cp.is_available())
except Exception:  # ImportError, or CUDA runtime errors on GPU-less nodes
    cp = None
    _cp_LinearOperator = None
    _cp_eigsh = None
    HAS_GPU = False


def _check(usegpu):
    if usegpu and not HAS_GPU:
        raise RuntimeError("usegpu=True but CuPy/GPU is not available")


def xp(usegpu=False):
    """Return the array module (``numpy`` or ``cupy``)."""
    _check(usegpu)
    return cp if usegpu else np


def LinearOperator(usegpu=False):
    """Return the LinearOperator class for the chosen backend."""
    _check(usegpu)
    return _cp_LinearOperator if usegpu else _np_LinearOperator


def eigsh(usegpu=False):
    """Return the Lanczos ``eigsh`` for the chosen backend."""
    _check(usegpu)
    return _cp_eigsh if usegpu else _np_eigsh


def to_cpu(x):
    """Move a cupy array to host memory (no-op for numpy arrays and scalars)."""
    return x.get() if hasattr(x, "get") else x


def free_gpu_memory():
    """Release cupy's cached device and pinned memory pools (no-op on CPU)."""
    if cp is not None and HAS_GPU:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
