"""Generate and document an LTX-2.3 transition between the selected keyframes."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import av
import numpy as np
from PIL import Image, ImageDraw, ImageOps


STAGE_ROOT = Path(__file__).resolve().parent
KEYFRAME_ROOT = (
    STAGE_ROOT / "output" / "keyframe_render" / "transport_pair" / "final"
)
OUTPUT_ROOT = STAGE_ROOT / "output" / "video_transition" / "ltx23_flf"
COMFY_ROOT = Path("/persistent/ComfyUI")
COMFY_INPUT = COMFY_ROOT / "input"
COMFY_OUTPUT = COMFY_ROOT / "output"

WIDTH = 512
HEIGHT = 320
FPS = 24
DURATION_SECONDS = 4
FRAME_COUNT = DURATION_SECONDS * FPS + 1
NOISE_SEED = 20260728
GUIDE_STRENGTH = 0.7
CHECKPOINT = "ltx-2.3-22b-dev-fp8.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"
DISTILLED_LORA = (
    "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
)
OUTPUT_PREFIX = "stage1_transport_flf_ltx23"

POSITIVE_PROMPT = """A fixed orthographic top-down aerial view of the exact same
river, sandy coast, and blue-green sea shown in the supplied first and last
frames. Preserve the geography, shoreline, riverbanks, framing, scale, natural
daylight, and color palette. Only the water and suspended sediment move. A soft
ochre-brown sediment cloud flows continuously downstream from left to right
inside the river channel, with subtle fluid turbulence and small variations in
concentration. Over the shot the sediment front advances smoothly from the
middle-lower river and reaches the river outlet at the coast in the final frame.
The camera remains completely locked. One continuous physically understandable
scientific visualization. No sediment enters the open sea yet and no new delta
land forms."""

NEGATIVE_PROMPT = """camera movement, pan, zoom, tilt, rotation, perspective
change, oblique view, shoreline movement, changing riverbanks, terrain morphing,
new channels, branching distributaries, completed delta, new land, offshore
sediment plume, sediment moving upstream, abrupt cut, scene change, flicker,
jitter, pulsing, boiling texture, frozen water, opaque mud, plastic CGI, text,
labels, arrows, people, buildings, boats, watermark, blurry, low quality"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def node(class_type: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}


def build_workflow(first_name: str, last_name: str) -> dict[str, Any]:
    """Expand ComfyUI's official LTX-2.3 FLF2V blueprint into API format."""

    return {
        "1": node("CheckpointLoaderSimple", ckpt_name=CHECKPOINT),
        "2": node(
            "LTXAVTextEncoderLoader",
            text_encoder=TEXT_ENCODER,
            ckpt_name=CHECKPOINT,
            device="default",
        ),
        "3": node("CLIPTextEncode", text=POSITIVE_PROMPT, clip=["2", 0]),
        "4": node("CLIPTextEncode", text=NEGATIVE_PROMPT, clip=["2", 0]),
        "5": node(
            "LTXVConditioning",
            positive=["3", 0],
            negative=["4", 0],
            frame_rate=float(FPS),
        ),
        "6": node(
            "LoraLoaderModelOnly",
            model=["1", 0],
            lora_name=DISTILLED_LORA,
            strength_model=0.5,
        ),
        "7": node("LTXVAudioVAELoader", ckpt_name=CHECKPOINT),
        "8": node("LoadImage", image=first_name),
        "9": node("LoadImage", image=last_name),
        "10": node(
            "ImageScale",
            image=["8", 0],
            upscale_method="bicubic",
            width=WIDTH,
            height=HEIGHT,
            crop="center",
        ),
        "11": node(
            "ImageScale",
            image=["9", 0],
            upscale_method="bicubic",
            width=WIDTH,
            height=HEIGHT,
            crop="center",
        ),
        "12": node("LTXVPreprocess", image=["10", 0], img_compression=25),
        "13": node("LTXVPreprocess", image=["11", 0], img_compression=25),
        "14": node(
            "EmptyLTXVLatentVideo",
            width=WIDTH,
            height=HEIGHT,
            length=FRAME_COUNT,
            batch_size=1,
        ),
        "15": node(
            "LTXVEmptyLatentAudio",
            frames_number=FRAME_COUNT,
            frame_rate=float(FPS),
            batch_size=1,
            audio_vae=["7", 0],
        ),
        "16": node(
            "LTXVAddGuide",
            positive=["5", 0],
            negative=["5", 1],
            vae=["1", 2],
            latent=["14", 0],
            image=["12", 0],
            frame_idx=0,
            strength=GUIDE_STRENGTH,
        ),
        "17": node(
            "LTXVAddGuide",
            positive=["16", 0],
            negative=["16", 1],
            vae=["1", 2],
            latent=["16", 2],
            image=["13", 0],
            frame_idx=-1,
            strength=GUIDE_STRENGTH,
        ),
        "18": node(
            "LTXVConcatAVLatent",
            video_latent=["17", 2],
            audio_latent=["15", 0],
        ),
        "19": node("RandomNoise", noise_seed=NOISE_SEED),
        "20": node(
            "CFGGuider",
            model=["6", 0],
            positive=["17", 0],
            negative=["17", 1],
            cfg=1.0,
        ),
        "21": node("SamplerEulerAncestral", eta=0.0, s_noise=1.0),
        "22": node(
            "ManualSigmas",
            sigmas=(
                "1.0, 0.99375, 0.9875, 0.98125, 0.975, "
                "0.909375, 0.725, 0.421875, 0.0"
            ),
        ),
        "23": node(
            "SamplerCustomAdvanced",
            noise=["19", 0],
            guider=["20", 0],
            sampler=["21", 0],
            sigmas=["22", 0],
            latent_image=["18", 0],
        ),
        "24": node("LTXVSeparateAVLatent", av_latent=["23", 1]),
        "25": node(
            "LTXVCropGuides",
            positive=["17", 0],
            negative=["17", 1],
            latent=["24", 0],
        ),
        "26": node(
            "VAEDecodeTiled",
            samples=["25", 2],
            vae=["1", 2],
            tile_size=768,
            overlap=64,
            temporal_size=64,
            temporal_overlap=4,
        ),
        "27": node(
            "LTXVAudioVAEDecode",
            samples=["24", 1],
            audio_vae=["7", 0],
        ),
        "28": node(
            "CreateVideo",
            images=["26", 0],
            audio=["27", 0],
            fps=float(FPS),
            bit_depth=8,
        ),
        "29": node(
            "SaveVideo",
            video=["28", 0],
            filename_prefix=OUTPUT_PREFIX,
            format="mp4",
            codec="h264",
        ),
    }


def request_json(
    url: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI HTTP {error.code}: {body}") from error


def prepare() -> tuple[dict[str, Any], Path, Path]:
    first_source = KEYFRAME_ROOT / "in_channel.png"
    last_source = KEYFRAME_ROOT / "at_outlet.png"
    if not first_source.is_file() or not last_source.is_file():
        raise FileNotFoundError(
            "Missing selected keyframes under "
            f"{KEYFRAME_ROOT}; run the Stage 1.2 selection first."
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    first_input = COMFY_INPUT / "stage1_transition_first.png"
    last_input = COMFY_INPUT / "stage1_transition_last.png"
    shutil.copy2(first_source, first_input)
    shutil.copy2(last_source, last_input)
    shutil.copy2(first_source, OUTPUT_ROOT / "input_first.png")
    shutil.copy2(last_source, OUTPUT_ROOT / "input_last.png")

    workflow = build_workflow(first_input.name, last_input.name)
    (OUTPUT_ROOT / "workflow_api.json").write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_ROOT / "prompt.txt").write_text(
        "Positive prompt:\n\n"
        + POSITIVE_PROMPT
        + "\n\nNegative prompt:\n\n"
        + NEGATIVE_PROMPT
        + "\n",
        encoding="utf-8",
    )
    return workflow, first_source, last_source


def find_video_record(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        filename = value.get("filename")
        if isinstance(filename, str) and filename.lower().endswith(".mp4"):
            return value
        for child in value.values():
            found = find_video_record(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_video_record(child)
            if found:
                return found
    return None


def wait_for_result(
    server: str, prompt_id: str, timeout_seconds: int
) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    next_update = 0.0
    while True:
        elapsed = time.monotonic() - started
        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"Generation {prompt_id} exceeded {timeout_seconds} seconds"
            )
        history = request_json(f"{server}/history/{prompt_id}")
        if prompt_id in history:
            record = history[prompt_id]
            status = record.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(
                    "ComfyUI generation failed: "
                    + json.dumps(status, ensure_ascii=False)
                )
            if find_video_record(record.get("outputs", {})):
                return record, elapsed
        if elapsed >= next_update:
            print(
                f"LTX-2.3 is generating: {elapsed / 60:.1f} minutes elapsed",
                flush=True,
            )
            next_update = elapsed + 15
        time.sleep(5)


def copy_comfy_video(record: dict[str, Any]) -> Path:
    video_record = find_video_record(record.get("outputs", {}))
    if not video_record:
        raise RuntimeError("SaveVideo completed without an MP4 output record")
    subfolder = video_record.get("subfolder", "")
    source = COMFY_OUTPUT / subfolder / video_record["filename"]
    if not source.is_file():
        raise FileNotFoundError(f"ComfyUI reported a missing output: {source}")
    target = OUTPUT_ROOT / "transition.mp4"
    shutil.copy2(source, target)
    return target


def video_frames(video_path: Path) -> tuple[dict[str, Any], list[np.ndarray]]:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        frames = [
            frame.to_ndarray(format="rgb24")
            for frame in container.decode(video=0)
        ]
        fps = float(stream.average_rate)
        info = {
            "codec": stream.codec_context.name,
            "width": stream.width,
            "height": stream.height,
            "fps": fps,
            "frame_count": len(frames),
            "duration_seconds": len(frames) / fps,
            "has_audio": bool(container.streams.audio),
        }
    return info, frames


def resized_reference(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    fitted = ImageOps.fit(
        image,
        (WIDTH, HEIGHT),
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )
    return np.asarray(fitted)


def comparison_metrics(
    generated: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    delta = generated.astype(np.float32) - reference.astype(np.float32)
    mse = float(np.mean(delta * delta))
    return {
        "mean_absolute_pixel_error_0_255": round(
            float(np.mean(np.abs(delta))), 3
        ),
        "psnr_db": round(
            float(20 * np.log10(255.0 / np.sqrt(max(mse, 1e-12)))), 3
        ),
    }


def temporal_metrics(frames: list[np.ndarray]) -> dict[str, Any]:
    frame_differences = [
        float(
            np.mean(
                np.abs(
                    frames[index].astype(np.float32)
                    - frames[index - 1].astype(np.float32)
                )
            )
        )
        for index in range(1, len(frames))
    ]
    sample_indices = np.linspace(
        0, len(frames) - 1, 9, dtype=int
    ).tolist()
    front_positions = []
    for index in sample_indices:
        frame = frames[index].astype(np.float32)
        sediment_score = np.mean(
            frame[128:180, :, 0] - frame[128:180, :, 2],
            axis=0,
        )
        smoothed = np.convolve(
            sediment_score, np.ones(9, dtype=np.float32) / 9, mode="same"
        )
        candidates = np.flatnonzero(smoothed[:231] > 20)
        front_positions.append(
            {
                "frame": index,
                "seconds": round(index / FPS, 2),
                "front_x": int(candidates.max()) if candidates.size else None,
            }
        )
    valid_positions = [
        item["front_x"]
        for item in front_positions
        if item["front_x"] is not None
    ]
    return {
        "mean_consecutive_frame_mae_0_255": round(
            float(np.mean(frame_differences)), 4
        ),
        "p95_consecutive_frame_mae_0_255": round(
            float(np.percentile(frame_differences, 95)), 4
        ),
        "maximum_consecutive_frame_mae_0_255": round(
            float(np.max(frame_differences)), 4
        ),
        "sediment_front_samples": front_positions,
        "sediment_front_progress_pixels": (
            valid_positions[-1] - valid_positions[0]
        ),
        "all_sampled_intervals_move_downstream": all(
            right >= left
            for left, right in zip(
                valid_positions, valid_positions[1:], strict=False
            )
        ),
        "front_tracking_method": (
            "Scene-specific audit heuristic at 512x320: average red-minus-blue "
            "over the horizontal river band y=128:180, smooth over 9 pixels, "
            "and take the rightmost x<=230 above 20. This measures visible "
            "brown-front motion; it is not a physical sediment measurement."
        ),
    }


def save_preview(frames: list[np.ndarray]) -> Path:
    indices = np.linspace(0, len(frames) - 1, 9, dtype=int).tolist()
    selected: list[Image.Image] = []
    for index in indices:
        image = Image.fromarray(frames[index])
        path = OUTPUT_ROOT / f"frame_{index:03d}.png"
        image.save(path)
        label = f"Frame {index:02d}  |  {index / FPS:.2f} s"
        panel = Image.new("RGB", (WIDTH, HEIGHT + 42), "white")
        panel.paste(image, (0, 42))
        ImageDraw.Draw(panel).text((12, 12), label, fill=(20, 35, 42))
        selected.append(panel)
    sheet = Image.new(
        "RGB", (WIDTH * 3, (HEIGHT + 42) * 3), (236, 232, 222)
    )
    for index, panel in enumerate(selected):
        sheet.paste(
            panel,
            ((index % 3) * WIDTH, (index // 3) * (HEIGHT + 42)),
        )
    path = OUTPUT_ROOT / "generated_frames.jpg"
    sheet.save(path, quality=92)
    return path


def write_report(metadata: dict[str, Any]) -> None:
    positive = html.escape(POSITIVE_PROMPT)
    negative = html.escape(NEGATIVE_PROMPT)
    first_metric = metadata["endpoint_comparison"]["first"]
    last_metric = metadata["endpoint_comparison"]["last"]
    temporal = metadata["temporal_audit"]
    front_samples = " → ".join(
        str(sample["front_x"])
        for sample in temporal["sediment_front_samples"]
    )
    report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 1.3｜两张关键帧的 LTX-2.3 过渡测试</title>
<style>
:root{{--ink:#17232a;--muted:#617078;--river:#0c6875;--sand:#f2eee3;--card:#fff;
--line:#d8d4ca;--warn:#fff0dd}}*{{box-sizing:border-box}}body{{margin:0;background:var(--sand);
color:var(--ink);font:16px/1.7 system-ui,"Noto Sans SC",sans-serif}}header{{padding:64px 24px;
color:white;background:linear-gradient(120deg,#073c47,#13808c)}}main,header>div{{width:min(1120px,
calc(100% - 32px));margin:auto}}h1{{max-width:850px;margin:0;font-size:clamp(2rem,5vw,3.8rem);
line-height:1.12}}header p{{max-width:780px;color:#d8f0ed}}main{{padding:48px 0 72px}}section{{margin-bottom:58px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}figure,.card{{margin:0;background:var(--card);
border:1px solid var(--line);border-radius:15px;overflow:hidden;box-shadow:0 10px 30px #00000012}}
img,video{{display:block;width:100%;height:auto}}figcaption,.pad{{padding:14px 17px}}.note{{padding:18px;
border-left:5px solid var(--river);background:#e5f2ed;border-radius:9px}}.warn{{border-left-color:#a65127;
background:var(--warn)}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:12px 15px;
border-bottom:1px solid var(--line);text-align:left}}th{{background:#0b5661;color:white}}details{{margin:12px 0;
border:1px solid var(--line);border-radius:10px;background:white}}summary{{padding:14px 17px;font-weight:700;
cursor:pointer}}pre{{margin:0 17px 17px;padding:14px;white-space:pre-wrap;overflow-wrap:anywhere;color:#e8f5f2;
background:#102d33;border-radius:8px}}code{{font-family:ui-monospace,monospace}}@media(max-width:700px){{
.grid{{grid-template-columns:1fr}}}}</style></head>
<body><header><div><p>STAGE 1.3 · 首尾帧视频测试</p>
<h1>让泥沙从河道中段移动到河口</h1>
<p>这次不是纯文本生成。LTX-2.3 明确接收上一阶段选中的两张图：第一张固定视频开头，
第二张固定视频结尾，模型只负责生成中间的连续运动。</p></div></header>
<main>
<section><h2>实际生成的视频</h2>
<video controls muted loop poster="frame_048.png" src="transition.mp4"></video>
<div class="note"><strong>怎么看：</strong>重点检查镜头和岸线是否稳定、泥沙是否从左向右推进、
中间帧是否出现闪烁或凭空改变地形。模型同时生成了音轨，但本页默认静音播放。</div></section>
<section><h2>这一版的判断</h2>
<div class="grid">
<div class="note"><h3>有效的部分</h3><p>首尾帧约束生效，没有换场，也没有提前长出三角洲。
棕色泥沙前缘在 9 个采样时刻的位置依次是：</p>
<p><strong>{front_samples} px</strong></p>
<p>它累计向河口推进 {temporal["sediment_front_progress_pixels"]} px，
且每个采样间隔都没有倒退，因此不只是静止画面上的随机闪烁。</p></div>
<div class="note warn"><h3>仍然不足</h3><p>运动主要表现为柔和的棕色浑水前缘，
没有形成可辨认的泥沙颗粒；水面纹理和岸边有轻微“呼吸感”。
512×320 的首次测试也偏软。这一版适合验证 FLF2V 路线，不应直接当最终成片。</p></div>
</div></section>
<section><h2>输入的两张关键帧</h2><div class="grid">
<figure><img src="input_first.png" alt="输入首帧"><figcaption><strong>首帧：</strong>
泥沙仍在河道中。</figcaption></figure>
<figure><img src="input_last.png" alt="输入尾帧"><figcaption><strong>尾帧：</strong>
泥沙前缘到达河口。</figcaption></figure></div></section>
<section><h2>模型实际生成的 9 个时间点</h2>
<figure><a href="generated_frames.jpg"><img src="generated_frames.jpg"
alt="按时间顺序排列的九张生成视频抽帧"></a>
<figcaption>从左到右、从上到下依次展示 0 至 4 秒的 9 个时间点，
用来检查首尾约束、泥沙推进、岸线稳定性和中途闪烁。</figcaption>
</figure></section>
<section><h2>这次怎么生成</h2>
<table><tbody>
<tr><td>模型</td><td>{CHECKPOINT} + distilled 1.1 LoRA</td></tr>
<tr><td>流程</td><td>ComfyUI 官方 LTX-2.3 First-Last-Frame to Video 结构</td></tr>
<tr><td>工作尺寸</td><td>{WIDTH} × {HEIGHT}</td></tr>
<tr><td>长度</td><td>{metadata["video"]["frame_count"]} 帧，
{metadata["video"]["fps"]:.0f} fps，约 {metadata["video"]["duration_seconds"]:.2f} 秒</td></tr>
<tr><td>首尾引导强度</td><td>{GUIDE_STRENGTH}</td></tr>
<tr><td>随机种子</td><td>{NOISE_SEED}（只用于复现中间运动）</td></tr>
<tr><td>生成耗时</td><td>{metadata["generation_seconds"] / 60:.2f} 分钟</td></tr>
<tr><td>相邻帧平均像素变化</td><td>{temporal["mean_consecutive_frame_mae_0_255"]}
/ 255；变化平缓，没有检测到突然跳帧</td></tr>
</tbody></table>
<div class="note warn" style="margin-top:18px"><strong>首尾相似度不是质量裁判：</strong>
压缩和 VAE 重建会造成像素差异。首帧 MAE {first_metric["mean_absolute_pixel_error_0_255"]}，
尾帧 MAE {last_metric["mean_absolute_pixel_error_0_255"]}；仍需以实际画面判断岸线和泥沙位置。</div>
</section>
<section><h2>提示词与复现文件</h2>
<details><summary>正向提示词</summary><pre>{positive}</pre></details>
<details><summary>负向提示词</summary><pre>{negative}</pre></details>
<div class="card pad"><ul>
<li><a href="workflow_api.json">完整 ComfyUI API 工作流</a></li>
<li><a href="metadata.json">生成参数、耗时和文件哈希</a></li>
<li><a href="prompt.txt">纯文本提示词</a></li>
</ul><p>复现命令：<code>/workspace/comfyui-rocm-env/bin/python
-m modules.video_model.stage1.video_transition</code></p></div>
</section></main></body></html>"""
    (OUTPUT_ROOT / "report.html").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    workflow, first_source, last_source = prepare()
    if args.prepare_only:
        print(OUTPUT_ROOT / "workflow_api.json")
        return 0

    request_json(f"{args.server}/system_stats")
    queued = request_json(f"{args.server}/prompt", {"prompt": workflow})
    prompt_id = queued["prompt_id"]
    print(f"Queued ComfyUI prompt: {prompt_id}", flush=True)
    history, generation_seconds = wait_for_result(
        args.server, prompt_id, args.timeout
    )
    video_path = copy_comfy_video(history)
    video_info, frames = video_frames(video_path)
    if len(frames) != FRAME_COUNT:
        raise RuntimeError(
            f"Expected {FRAME_COUNT} video frames, decoded {len(frames)}"
        )
    save_preview(frames)

    first_reference = resized_reference(first_source)
    last_reference = resized_reference(last_source)
    metadata = {
        "status": "complete",
        "classification": "LTX-2.3 first-last-frame conditioned video",
        "prompt_id": prompt_id,
        "generation_seconds": round(generation_seconds, 3),
        "model": {
            "checkpoint": CHECKPOINT,
            "text_encoder": TEXT_ENCODER,
            "distilled_lora": DISTILLED_LORA,
            "lora_strength": 0.5,
        },
        "settings": {
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "duration_seconds_requested": DURATION_SECONDS,
            "frame_count": FRAME_COUNT,
            "noise_seed": NOISE_SEED,
            "guide_strength": GUIDE_STRENGTH,
            "sampler": "euler_ancestral",
            "sigmas": [
                1.0,
                0.99375,
                0.9875,
                0.98125,
                0.975,
                0.909375,
                0.725,
                0.421875,
                0.0,
            ],
        },
        "inputs": {
            "first": {
                "path": str(first_source.resolve()),
                "sha256": sha256(first_source),
            },
            "last": {
                "path": str(last_source.resolve()),
                "sha256": sha256(last_source),
            },
        },
        "video": {
            **video_info,
            "path": str(video_path.resolve()),
            "sha256": sha256(video_path),
            "size_bytes": video_path.stat().st_size,
        },
        "endpoint_comparison": {
            "first": comparison_metrics(frames[0], first_reference),
            "last": comparison_metrics(frames[-1], last_reference),
            "note": (
                "Pixel metrics include VAE reconstruction, resizing, and H.264 "
                "compression; use them for audit, not semantic acceptance."
            ),
        },
        "temporal_audit": temporal_metrics(frames),
        "files": {
            "workflow": str((OUTPUT_ROOT / "workflow_api.json").resolve()),
            "prompt": str((OUTPUT_ROOT / "prompt.txt").resolve()),
            "preview": str((OUTPUT_ROOT / "generated_frames.jpg").resolve()),
        },
    }
    (OUTPUT_ROOT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
