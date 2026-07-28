"""Reusable LTX primary, Wan baseline, and procedural fallback backends."""

from __future__ import annotations

import gc
import importlib.metadata
import importlib.util
import math
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .prompting import InputSpec


@dataclass(frozen=True)
class GenerationSettings:
    duration: float
    fps: int
    width: int
    height: int
    seed: int
    inference_steps: int
    guidance_scale: float
    model_id: str | None
    cpu_offload: bool = False
    num_frames: int | None = None
    model_revision: str = "main"
    ltx_checkpoint: str = "ltxv-2b-0.9.8-distilled.safetensors"
    ltx_text_encoder: str = "PixArt-alpha/PixArt-XL-2-1024-MS"
    ltx_upscaler: str = "ltxv-spatial-upscaler-0.9.8.safetensors"
    use_multiscale: bool = False
    vae_tiling: bool = False
    vae_slicing: bool = False
    first_frame: str | None = None


LTX_REPO_ID = "Lightricks/LTX-Video"
LTX_CHECKPOINT = "ltxv-2b-0.9.8-distilled.safetensors"
LTX_DISTILLED_TIMESTEPS = [1.0, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250]
LTX_SECOND_PASS = [0.9094, 0.7250, 0.4219]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch_environment() -> dict[str, Any]:
    details: dict[str, Any] = {
        "torch_installed": False,
        "torch_version": None,
        "accelerator_available": False,
        "rocm_version": None,
        "gpu_name": None,
        "gpu_arch": None,
        "gpu_pci_bus_id": None,
        "gpu_memory_gib": None,
    }
    try:
        import torch
    except ImportError:
        return details
    details["torch_installed"] = True
    details["torch_version"] = torch.__version__
    details["accelerator_available"] = bool(torch.cuda.is_available())
    details["rocm_version"] = getattr(torch.version, "hip", None)
    if details["accelerator_available"]:
        properties = torch.cuda.get_device_properties(0)
        details["gpu_name"] = properties.name
        details["gpu_arch"] = getattr(properties, "gcnArchName", None)
        details["gpu_pci_bus_id"] = getattr(properties, "pci_bus_id", None)
        details["gpu_memory_gib"] = round(properties.total_memory / 1024**3, 2)
    return details


@contextmanager
def _rocm_safe_optional_imports() -> Any:
    """Hide NVIDIA-only optional packages that may be installed but unusable on ROCm.

    Some shared environments contain FlashAttention metadata even though its CUDA
    extension cannot load. Transformers/Diffusers then import it opportunistically.
    LTX does not require these kernels, so report them unavailable during imports.
    """

    import torch

    if not getattr(torch.version, "hip", None):
        yield
        return
    original_find_spec = importlib.util.find_spec
    blocked = {"flash_attn", "flash_attn_3", "xformers"}
    if os.environ.get("HF_HUB_DISABLE_XET", "").lower() in {"1", "true", "yes"}:
        # huggingface-hub 0.30 does not yet implement HF_HUB_DISABLE_XET.
        # Honor the modern variable locally so a broken optional hf_xet install
        # cannot make checkpoint downloads hang.
        blocked.add("hf_xet")

    def safe_find_spec(name: str, package: str | None = None) -> Any:
        if name.split(".", 1)[0] in blocked:
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = safe_find_spec
    try:
        yield
    finally:
        importlib.util.find_spec = original_find_spec


def _honor_hf_hub_disable_xet() -> None:
    """Backport HF_HUB_DISABLE_XET behavior for huggingface-hub 0.30."""

    if os.environ.get("HF_HUB_DISABLE_XET", "").lower() not in {"1", "true", "yes"}:
        return
    from huggingface_hub.utils import _runtime as hub_runtime

    package_versions = getattr(hub_runtime, "_package_versions", None)
    if isinstance(package_versions, dict):
        package_versions["hf_xet"] = "N/A"


def ltx_environment() -> dict[str, Any]:
    """Detect the official LTX package and a real ROCm accelerator."""

    details = _torch_environment()
    details.update(
        {
            "available": False,
            "ltx_video_installed": importlib.util.find_spec("ltx_video") is not None,
            "ltx_video_version": _package_version("ltx-video"),
            "diffusers_version": _package_version("diffusers"),
            "reason": None,
        }
    )
    if not details["torch_installed"]:
        details["reason"] = "PyTorch is not installed"
    elif not details["accelerator_available"]:
        details["reason"] = "PyTorch cannot see an accelerator"
    elif not details["rocm_version"]:
        details["reason"] = "ROCm PyTorch is required for the V2 LTX backend (torch.version.hip is empty)"
    elif not details["ltx_video_installed"]:
        details["reason"] = "the official ltx-video package is not installed"
    else:
        details["available"] = True
    return details


def wan_environment() -> dict[str, Any]:
    """Return availability details without importing model code at module import time."""

    details: dict[str, Any] = {
        "available": False,
        "torch_installed": False,
        "diffusers_installed": False,
        "accelerator_available": False,
        "rocm_version": None,
        "reason": None,
    }
    try:
        import torch

        details["torch_installed"] = True
        details["accelerator_available"] = bool(torch.cuda.is_available())
        details["rocm_version"] = getattr(torch.version, "hip", None)
    except ImportError:
        details["reason"] = "PyTorch is not installed"
        return details

    try:
        with _rocm_safe_optional_imports():
            from diffusers import WanPipeline  # noqa: F401

        details["diffusers_installed"] = True
    except (ImportError, RuntimeError) as exc:
        details["reason"] = f"Diffusers WanPipeline is unavailable: {exc}"
        return details

    if not details["accelerator_available"]:
        details["reason"] = "PyTorch cannot see a CUDA/ROCm accelerator"
        return details

    details["available"] = True
    return details


def _resolved_revision(path: str | Path) -> str | None:
    parts = Path(path).parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def _select_ltx_timesteps(steps: int) -> list[float]:
    if not 1 <= steps <= len(LTX_DISTILLED_TIMESTEPS):
        raise ValueError(
            f"LTX 0.9.8 2B distilled supports 1-{len(LTX_DISTILLED_TIMESTEPS)} "
            "first-pass steps in this runner"
        )
    if steps == 1:
        return [LTX_DISTILLED_TIMESTEPS[0]]
    last = len(LTX_DISTILLED_TIMESTEPS) - 1
    indices = [round(index * last / (steps - 1)) for index in range(steps)]
    # Rounding can duplicate an index for unusual lengths; preserve order.
    return [LTX_DISTILLED_TIMESTEPS[index] for index in dict.fromkeys(indices)]


def _write_tensor_video(images: Any, output_path: Path, fps: int) -> float:
    """Encode official LTX tensor output (B,C,F,H,W) as a raw MP4."""

    import imageio
    import numpy as np

    started = time.perf_counter()
    video = images[0].permute(1, 2, 3, 0).detach().cpu().float().numpy()
    video = np.clip(video * 255.0, 0, 255).astype(np.uint8)
    with imageio.get_writer(output_path, fps=fps, codec="libx264", pixelformat="yuv420p") as writer:
        for frame in video:
            writer.append_data(frame)
    return time.perf_counter() - started


class LTXRunner:
    """Reusable official LTX-Video 0.9.8 2B distilled runner."""

    backend = "ltx"

    def __init__(self) -> None:
        self.pipeline: Any = None
        self.multiscale_pipeline: Any = None
        self._load_info: dict[str, Any] = {}
        self._model_key: tuple[Any, ...] | None = None

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    def _configure_vae(self, settings: GenerationSettings) -> None:
        vae = self.pipeline.vae
        if settings.vae_tiling:
            if hasattr(vae, "enable_hw_tiling"):
                vae.enable_hw_tiling()
            elif hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
            else:
                raise RuntimeError("installed official LTX VAE does not expose spatial tiling")
        elif hasattr(vae, "disable_hw_tiling"):
            vae.disable_hw_tiling()
        elif hasattr(vae, "disable_tiling"):
            vae.disable_tiling()

        if settings.vae_slicing:
            if hasattr(vae, "enable_z_tiling"):
                vae.enable_z_tiling(z_sample_size=8)
            elif hasattr(vae, "enable_slicing"):
                vae.enable_slicing()
            else:
                raise RuntimeError("installed official LTX VAE does not expose temporal slicing")
        elif hasattr(vae, "disable_z_tiling"):
            vae.disable_z_tiling()
        elif hasattr(vae, "disable_slicing"):
            vae.disable_slicing()

    def load(self, settings: GenerationSettings) -> tuple[float, dict[str, Any]]:
        if not settings.model_id:
            raise ValueError("LTX runner requires a model_id")
        model_key = (
            settings.model_id,
            settings.model_revision,
            settings.ltx_checkpoint,
            settings.ltx_text_encoder,
        )
        if self.loaded:
            if model_key != self._model_key:
                raise ValueError("a reusable LTX runner cannot change model or revision")
            self._configure_vae(settings)
            return 0.0, {
                **self._load_info,
                "model_reused": True,
                "vae_spatial_tiling": settings.vae_tiling,
                "vae_temporal_slicing": settings.vae_slicing,
            }
        environment = ltx_environment()
        if not environment["available"]:
            raise RuntimeError(f"LTX backend unavailable: {environment['reason']}")

        import torch
        with _rocm_safe_optional_imports():
            from huggingface_hub import hf_hub_download
            _honor_hf_hub_disable_xet()
            from ltx_video.inference import create_ltx_video_pipeline

        started = time.perf_counter()
        checkpoint_path = hf_hub_download(
            repo_id=settings.model_id,
            filename=settings.ltx_checkpoint,
            revision=settings.model_revision,
        )
        self.pipeline = create_ltx_video_pipeline(
            ckpt_path=checkpoint_path,
            precision="bfloat16",
            text_encoder_model_name_or_path=settings.ltx_text_encoder,
            sampler="from_checkpoint",
            device="cuda",
            enhance_prompt=False,
        )
        self._model_key = model_key
        self._configure_vae(settings)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        self._load_info = {
            "model_id": settings.model_id,
            "checkpoint": settings.ltx_checkpoint,
            "requested_revision": settings.model_revision,
            "resolved_revision": _resolved_revision(checkpoint_path),
            "precision": "bfloat16",
            "torch_version": torch.__version__,
            "rocm_version": torch.version.hip,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None),
            "diffusers_version": _package_version("diffusers"),
            "ltx_video_version": _package_version("ltx-video"),
            "text_encoder_model_id": settings.ltx_text_encoder,
            "text_encoder_revision": getattr(self.pipeline.text_encoder.config, "_commit_hash", None),
            "vae_spatial_tiling": settings.vae_tiling,
            "vae_temporal_slicing": settings.vae_slicing,
            "model_reused": False,
        }
        return elapsed, dict(self._load_info)

    def _ensure_multiscale(self, settings: GenerationSettings) -> float:
        if self.multiscale_pipeline is not None:
            return 0.0
        from huggingface_hub import hf_hub_download
        _honor_hf_hub_disable_xet()
        from ltx_video.inference import create_latent_upsampler
        from ltx_video.pipelines.pipeline_ltx_video import LTXMultiScalePipeline

        started = time.perf_counter()
        upscaler_path = hf_hub_download(
            repo_id=settings.model_id,
            filename=settings.ltx_upscaler,
            revision=settings.model_revision,
        )
        upscaler = create_latent_upsampler(upscaler_path, "cuda")
        self.multiscale_pipeline = LTXMultiScalePipeline(self.pipeline, latent_upsampler=upscaler)
        import torch

        torch.cuda.synchronize()
        return time.perf_counter() - started

    def generate(
        self,
        positive_prompt: str,
        negative_prompt: str,
        output_path: Path,
        settings: GenerationSettings,
    ) -> dict[str, Any]:
        import torch
        from ltx_video.inference import calculate_padding, prepare_conditioning
        from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

        if not self.loaded:
            raise RuntimeError("LTX runner must be loaded before generation")
        if settings.guidance_scale != 1.0:
            raise ValueError("LTX 0.9.8 distilled requires guidance_scale=1.0")
        num_frames = settings.num_frames or (math.ceil(settings.duration * settings.fps / 8) * 8 + 1)
        if (num_frames - 1) % 8:
            raise ValueError("LTX num_frames must follow the 8*k+1 constraint")
        if settings.width % 32 or settings.height % 32:
            raise ValueError("LTX width and height must be divisible by 32")

        extra_load = self._ensure_multiscale(settings) if settings.use_multiscale else 0.0
        torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator(device="cuda").manual_seed(settings.seed)
        conditioning_items = None
        if settings.first_frame:
            padding = calculate_padding(settings.height, settings.width, settings.height, settings.width)
            conditioning_items = prepare_conditioning(
                conditioning_media_paths=[settings.first_frame],
                conditioning_strengths=[1.0],
                conditioning_start_frames=[0],
                height=settings.height,
                width=settings.width,
                num_frames=num_frames,
                padding=padding,
                pipeline=self.pipeline,
            )

        common: dict[str, Any] = {
            "prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "height": settings.height,
            "width": settings.width,
            "num_frames": num_frames,
            "frame_rate": settings.fps,
            "skip_layer_strategy": SkipLayerStrategy.AttentionValues,
            "generator": generator,
            "output_type": "pt",
            "conditioning_items": conditioning_items,
            "is_video": True,
            "vae_per_channel_normalize": True,
            "image_cond_noise_scale": 0.025,
            "mixed_precision": False,
            "offload_to_cpu": settings.cpu_offload,
            "device": "cuda",
            "enhance_prompt": False,
            "stochastic_sampling": False,
            "decode_timestep": 0.05,
            "decode_noise_scale": 0.025,
        }
        torch.cuda.synchronize()
        started = time.perf_counter()
        if settings.use_multiscale:
            images = self.multiscale_pipeline(
                downscale_factor=2 / 3,
                first_pass={
                    "timesteps": LTX_DISTILLED_TIMESTEPS,
                    "guidance_scale": 1.0,
                    "stg_scale": 0.0,
                    "rescaling_scale": 1.0,
                    "skip_block_list": [42],
                },
                second_pass={
                    "timesteps": LTX_SECOND_PASS,
                    "guidance_scale": 1.0,
                    "stg_scale": 0.0,
                    "rescaling_scale": 1.0,
                    "skip_block_list": [42],
                },
                tone_map_compression_ratio=0.6,
                **common,
            ).images
            effective_steps = len(LTX_DISTILLED_TIMESTEPS) + len(LTX_SECOND_PASS)
        else:
            timesteps = _select_ltx_timesteps(settings.inference_steps)
            images = self.pipeline(
                timesteps=timesteps,
                guidance_scale=1.0,
                stg_scale=0.0,
                rescaling_scale=1.0,
                skip_block_list=[42],
                tone_map_compression_ratio=0.6,
                **common,
            ).images
            effective_steps = len(timesteps)
        torch.cuda.synchronize()
        inference = time.perf_counter() - started
        raw_encoding = _write_tensor_video(images, output_path, settings.fps)
        peak_memory = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        return {
            "backend": "ltx",
            "workflow": "LTX-Video 0.9.8 2B distilled official Python pipeline",
            "model_id": settings.model_id,
            "checkpoint": settings.ltx_checkpoint,
            "num_frames": num_frames,
            "effective_inference_steps": effective_steps,
            "generation_mode": "i2v" if settings.first_frame else "t2v",
            "first_frame": str(Path(settings.first_frame).resolve()) if settings.first_frame else None,
            "device": "cuda (ROCm PyTorch API)",
            "rocm_version": torch.version.hip,
            "gpu_name": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": peak_memory,
            "timing_seconds": {
                "additional_model_load": round(extra_load, 3),
                "inference": round(inference, 3),
                "vae_decode": None,
                "encoding_mp4": round(raw_encoding, 3),
            },
            "timing_limitations": [
                "The official LTX pipeline performs denoising and VAE decode in one call; inference includes VAE decode."
            ],
        }


class WanRunner:
    """Reusable Wan baseline runner for warm benchmark comparisons."""

    backend = "wan"

    def __init__(self) -> None:
        self.pipeline: Any = None
        self._load_info: dict[str, Any] = {}
        self._model_key: tuple[Any, ...] | None = None

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    def load(self, settings: GenerationSettings) -> tuple[float, dict[str, Any]]:
        if not settings.model_id:
            raise ValueError("Wan runner requires a model_id")
        model_key = (settings.model_id, settings.model_revision, settings.cpu_offload)
        if self.loaded:
            if model_key != self._model_key:
                raise ValueError("a reusable Wan runner cannot change model or revision")
            return 0.0, {**self._load_info, "model_reused": True}
        environment = wan_environment()
        if not environment["available"]:
            raise RuntimeError(f"Wan backend unavailable: {environment['reason']}")
        import torch
        with _rocm_safe_optional_imports():
            from diffusers import AutoencoderKLWan, WanPipeline
            from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        _honor_hf_hub_disable_xet()

        started = time.perf_counter()
        vae = AutoencoderKLWan.from_pretrained(
            settings.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            revision=settings.model_revision,
        )
        self.pipeline = WanPipeline.from_pretrained(
            settings.model_id,
            vae=vae,
            torch_dtype=torch.bfloat16,
            revision=settings.model_revision,
        )
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self.pipeline.scheduler.config,
            flow_shift=3.0,
        )
        if settings.cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to("cuda")
        self._model_key = model_key
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        self._load_info = {
            "model_id": settings.model_id,
            "requested_revision": settings.model_revision,
            "resolved_revision": getattr(self.pipeline.config, "_commit_hash", None),
            "precision": "bfloat16 (VAE float32)",
            "torch_version": torch.__version__,
            "rocm_version": getattr(torch.version, "hip", None),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None),
            "diffusers_version": _package_version("diffusers"),
            "ltx_video_version": None,
            "model_reused": False,
        }
        return elapsed, dict(self._load_info)

    def generate(
        self,
        positive_prompt: str,
        negative_prompt: str,
        output_path: Path,
        settings: GenerationSettings,
    ) -> dict[str, Any]:
        import torch
        from diffusers.utils import export_to_video

        if not self.loaded:
            raise RuntimeError("Wan runner must be loaded before generation")
        requested_frames = settings.num_frames or max(9, round(settings.duration * settings.fps))
        num_frames = max(9, math.ceil((requested_frames - 1) / 4) * 4 + 1)
        torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator(device="cuda").manual_seed(settings.seed)
        torch.cuda.synchronize()
        started = time.perf_counter()
        frames = self.pipeline(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            height=settings.height,
            width=settings.width,
            num_frames=num_frames,
            num_inference_steps=settings.inference_steps,
            guidance_scale=settings.guidance_scale,
            generator=generator,
        ).frames[0]
        torch.cuda.synchronize()
        inference = time.perf_counter() - started
        encode_started = time.perf_counter()
        export_to_video(frames, str(output_path), fps=settings.fps)
        raw_encoding = time.perf_counter() - encode_started
        return {
            "backend": "wan",
            "workflow": "text-to-video with Wan2.1 via Diffusers",
            "model_id": settings.model_id,
            "num_frames": num_frames,
            "generation_mode": "t2v",
            "device": "cuda (ROCm when torch.version.hip is set)",
            "rocm_version": getattr(torch.version, "hip", None),
            "gpu_name": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "timing_seconds": {
                "additional_model_load": 0.0,
                "inference": round(inference, 3),
                "vae_decode": None,
                "encoding_mp4": round(raw_encoding, 3),
            },
            "timing_limitations": ["Diffusers WanPipeline returns decoded frames; inference includes VAE decode."],
        }


class ProceduralRunner:
    backend = "procedural"

    @property
    def loaded(self) -> bool:
        return True

    def load(self, settings: GenerationSettings) -> tuple[float, dict[str, Any]]:
        return 0.0, {
            "model_id": None,
            "requested_revision": None,
            "resolved_revision": None,
            "precision": None,
            "torch_version": _package_version("torch"),
            "rocm_version": None,
            "gpu_name": None,
            "diffusers_version": _package_version("diffusers"),
            "ltx_video_version": _package_version("ltx-video"),
            "model_reused": True,
        }

    def generate(
        self,
        spec: InputSpec,
        output_path: Path,
        settings: GenerationSettings,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        details = generate_procedural(spec, output_path, settings)
        elapsed = time.perf_counter() - started
        details.update(
            {
                "generation_mode": "deterministic",
                "peak_gpu_memory_gib": None,
                "timing_seconds": {
                    "additional_model_load": 0.0,
                    "inference": None,
                    "vae_decode": None,
                    "encoding_mp4": round(elapsed, 3),
                },
                "timing_limitations": [
                    "Procedural frame rendering and raw MP4 encoding are measured together as encoding_mp4."
                ],
            }
        )
        return details


def make_runner(backend: str) -> LTXRunner | WanRunner | ProceduralRunner:
    if backend == "ltx":
        return LTXRunner()
    if backend == "wan":
        return WanRunner()
    if backend == "procedural":
        return ProceduralRunner()
    raise ValueError(f"unsupported concrete backend: {backend}")


def release_runner(runner: Any) -> None:
    """Release a failed model runner before trying the next auto backend."""

    for attribute in ("multiscale_pipeline", "pipeline"):
        if hasattr(runner, attribute):
            setattr(runner, attribute, None)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def generate_wan(
    positive_prompt: str,
    negative_prompt: str,
    output_path: Path,
    settings: GenerationSettings,
) -> dict[str, Any]:
    """Backward-compatible one-shot wrapper around the reusable Wan runner."""

    runner = WanRunner()
    load_seconds, load_info = runner.load(settings)
    result = runner.generate(positive_prompt, negative_prompt, output_path, settings)
    result["one_shot_model_load_seconds"] = round(load_seconds, 3)
    result["load_info"] = load_info
    return result


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _base_frame(width: int, height: int, content_type: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "#f7fafc")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (width * 0.045, height * 0.07, width * 0.955, height * 0.93),
        radius=max(8, width // 60),
        fill="#ffffff",
        outline="#d8e2ec",
        width=max(2, width // 320),
    )
    draw.text(
        (width * 0.075, height * 0.105),
        content_type.replace("_", " ").upper(),
        fill="#24364b",
        font=_font(max(14, height // 24)),
    )
    return image, draw


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], color: str, width: int) -> None:
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head = max(9, width * 3)
    points = [end]
    for offset in (2.55, -2.55):
        points.append((end[0] + head * math.cos(angle + offset), end[1] + head * math.sin(angle + offset)))
    draw.polygon(points, fill=color)


def _process_frame(draw: ImageDraw.ImageDraw, width: int, height: int, phase: float, gradient: bool) -> None:
    line_width = max(3, width // 180)
    if gradient:
        left, right = width * 0.13, width * 0.88
        top, bottom = height * 0.28, height * 0.79
        points = []
        for index in range(121):
            ratio = index / 120
            x = left + (right - left) * ratio
            y = top + (bottom - top) * (1 - math.exp(-4.2 * ratio))
            points.append((x, y))
        draw.line(points, fill="#4f6f8f", width=line_width)
        # Travel downhill only. Fade at both ends so the loop reset does not
        # visually teach the incorrect reverse action (climbing uphill).
        eased = phase * phase * (3 - 2 * phase)
        index = min(120, round(eased * 120))
        x, y = points[index]
        radius = max(8, width // 55)
        visibility = max(0.0, min(1.0, phase / 0.12, (1.0 - phase) / 0.18))
        fill = tuple(round(255 + (target - 255) * visibility) for target in (237, 106, 90))
        outline = tuple(round(255 + (target - 255) * visibility) for target in (158, 57, 48))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)
        draw.text((left, bottom + height * 0.05), "START", fill="#6b7f93", font=_font(max(12, height // 30)))
        draw.text((right - width * 0.09, bottom + height * 0.05), "MINIMUM", fill="#2a8f68", font=_font(max(12, height // 30)))
        return

    y = height * 0.56
    xs = [width * 0.2, width * 0.5, width * 0.8]
    labels = ["INPUT", "CHANGE", "RESULT"]
    for index in range(2):
        _arrow(draw, (xs[index] + width * 0.075, y), (xs[index + 1] - width * 0.075, y), "#93a9bf", line_width)
    for x, label in zip(xs, labels):
        radius = width * 0.065
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#e8f2fb", outline="#3978a8", width=line_width)
        box = draw.textbbox((0, 0), label, font=_font(max(10, height // 35)))
        draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2), label, fill="#244b68", font=_font(max(10, height // 35)))
    progress = (phase * 1.15) % 1.0
    x = xs[0] + (xs[-1] - xs[0]) * progress
    alpha = math.sin(math.pi * min(1.0, progress))
    radius = max(7, width // 70)
    color = (237, int(106 + 50 * alpha), 90)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def _state_change_frame(draw: ImageDraw.ImageDraw, width: int, height: int, phase: float) -> None:
    y = height * 0.57
    x1, x2 = width * 0.3, width * 0.7
    progress = 0.5 - 0.5 * math.cos(2 * math.pi * phase)
    line_width = max(3, width // 180)
    draw.text((x1 - width * 0.065, height * 0.31), "BEFORE", fill="#6b7f93", font=_font(max(12, height // 30)))
    draw.text((x2 - width * 0.05, height * 0.31), "AFTER", fill="#6b7f93", font=_font(max(12, height // 30)))
    radius1 = width * 0.06
    draw.ellipse((x1 - radius1, y - radius1, x1 + radius1, y + radius1), fill="#8dc5e8", outline="#3978a8", width=line_width)
    _arrow(draw, (x1 + width * 0.1, y), (x2 - width * 0.1, y), "#758ba0", line_width)
    radius2 = width * (0.06 + 0.025 * progress)
    red = int(141 + (79 * progress))
    green = int(197 - (45 * progress))
    blue = int(232 - (119 * progress))
    draw.rounded_rectangle((x2 - radius2, y - radius2, x2 + radius2, y + radius2), radius=int(radius2 * (1 - 0.7 * progress)), fill=(red, green, blue), outline="#9e6a26", width=line_width)


def _data_flow_frame(draw: ImageDraw.ImageDraw, width: int, height: int, phase: float) -> None:
    y = height * 0.56
    centers = [width * 0.2, width * 0.5, width * 0.8]
    labels = ["SOURCE", "TRANSFORM", "OUTPUT"]
    line_width = max(3, width // 180)
    for index in range(2):
        _arrow(draw, (centers[index] + width * 0.1, y), (centers[index + 1] - width * 0.1, y), "#9eb2c5", line_width)
    for index, (x, label) in enumerate(zip(centers, labels)):
        fill = ["#e8f2fb", "#fff0d9", "#e5f5ec"][index]
        outline = ["#3978a8", "#bf7b22", "#2a8f68"][index]
        draw.rounded_rectangle((x - width * 0.09, y - height * 0.105, x + width * 0.09, y + height * 0.105), radius=12, fill=fill, outline=outline, width=line_width)
        font = _font(max(10, height // 38))
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2), label, fill="#33485d", font=font)
    for offset in (0.0, 0.28, 0.56):
        progress = (phase + offset) % 1.0
        x = centers[0] + (centers[-1] - centers[0]) * progress
        radius = max(5, width // 100)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#ed6a5a")


def _scene_frame(draw: ImageDraw.ImageDraw, width: int, height: int, phase: float) -> None:
    center = (width * 0.5, height * 0.57)
    orbit_x, orbit_y = width * 0.25, height * 0.2
    line_width = max(3, width // 180)
    draw.ellipse((center[0] - orbit_x, center[1] - orbit_y, center[0] + orbit_x, center[1] + orbit_y), outline="#b8c8d8", width=line_width)
    core = max(15, width // 32)
    draw.ellipse((center[0] - core, center[1] - core, center[0] + core, center[1] + core), fill="#ffd166", outline="#b87d14", width=line_width)
    angle = 2 * math.pi * phase
    x = center[0] + orbit_x * math.cos(angle)
    y = center[1] + orbit_y * math.sin(angle)
    radius = max(9, width // 55)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#4e9dcc", outline="#29678e", width=2)
    draw.text((width * 0.39, height * 0.84), "ONE RELATIONSHIP", fill="#6b7f93", font=_font(max(12, height // 30)))


def generate_procedural(
    spec: InputSpec,
    output_path: Path,
    settings: GenerationSettings,
) -> dict[str, Any]:
    """Render a deterministic explainer used for CI, smoke tests, and fallback."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    frame_count = max(2, round(settings.duration * settings.fps))
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{settings.width}x{settings.height}",
        "-r",
        str(settings.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        lower_text = f"{spec.source_text} {spec.visual_goal} {spec.video_prompt}".lower()
        for index in range(frame_count):
            # Denominator excludes the duplicate endpoint for a clean periodic loop.
            phase = index / frame_count
            image, draw = _base_frame(settings.width, settings.height, spec.content_type)
            if spec.content_type == "process":
                gradient = any(term in lower_text for term in ("gradient", "loss curve", "梯度", "损失", "下坡"))
                _process_frame(draw, settings.width, settings.height, phase, gradient)
            elif spec.content_type == "state_change":
                _state_change_frame(draw, settings.width, settings.height, phase)
            elif spec.content_type == "data_flow":
                _data_flow_frame(draw, settings.width, settings.height, phase)
            else:
                _scene_frame(draw, settings.width, settings.height, phase)
            process.stdin.write(image.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if process.stderr:
            process.stderr.close()
        return_code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        if process.stderr:
            process.stderr.close()
        raise
    if return_code != 0:
        raise RuntimeError(f"procedural ffmpeg render failed: {stderr.strip()}")

    return {
        "backend": "procedural",
        "workflow": "deterministic explanation animation (smoke-test/fallback)",
        "model_id": None,
        "num_frames": frame_count,
        "device": "CPU",
        "rocm_version": None,
    }
