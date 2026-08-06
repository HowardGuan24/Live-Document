"""Generative video backend (LTX / Wan on ROCm) — availability probe + runner.

On machines without the ROCm PyTorch stack, the job manager falls back to the
procedural renderer (explicit `success_fallback`), keeping the demo honest.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


class GenerativeUnavailableError(RuntimeError):
    """Raised when the generative model stack is not available."""


@lru_cache(maxsize=1)
def probe() -> dict[str, Any]:
    try:
        import torch  # noqa: F401

        hip_ok = getattr(torch.version, "hip", None) is not None
        cuda_like = torch.cuda.is_available()
        return {
            "available": bool(hip_ok or cuda_like),
            "detail": (
                f"torch {torch.__version__} (HIP={hip_ok}, CUDA-visible={cuda_like})"
            ),
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "detail": f"torch import failed: {exc}"}


def run_generative(job_id: str, spec: dict[str, Any], style: dict[str, Any]) -> dict[str, Any]:
    """Run the LTX/Wan pipeline.

    The full LTX integration lives in modules/video_model. The web wrapper
    calls it when the ROCm stack is present; otherwise it raises
    GenerativeUnavailableError so the job manager can fall back.
    """
    info = probe()
    if not info["available"]:
        raise GenerativeUnavailableError(
            f"Generative engine unavailable: {info['detail']}"
        )
    # TODO(M3): call modules.video_model LTX/Wan backend with spec-derived prompts.
    raise GenerativeUnavailableError(
        "Generative engine detected but web LTX integration is not wired yet (M3)"
    )
