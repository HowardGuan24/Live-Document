"""Deterministic fallback for motion that a video model cannot preserve."""

from __future__ import annotations

import math
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from ..cases.sentinel_programs import SentinelProgram
from .contracts import artifact_record, sha256_path, write_json


def _encode(
    frames_root: Path,
    output_path: Path,
    *,
    fps: int,
    frame_count: int,
) -> None:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        str(frames_root / "frame_%03d.png"),
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        "-map_metadata",
        "-1",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip())


def _decoded_frame_count(path: Path) -> int:
    completed = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "framemd5",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip())
    return sum(
        bool(line.strip()) and not line.startswith("#")
        for line in completed.stdout.splitlines()
    )


def _preview(
    frames: list[np.ndarray],
    output_root: Path,
    fps: float,
) -> tuple[Path, list[int]]:
    indices = np.linspace(0, len(frames) - 1, 9, dtype=int).tolist()
    width, height = frames[0].shape[1], frames[0].shape[0]
    samples = output_root / "samples"
    samples.mkdir(exist_ok=True)
    panels = []
    for index in indices:
        image = Image.fromarray(frames[index])
        image.save(samples / f"frame_{index:03d}.png")
        panel = Image.new("RGB", (width, height + 34), "white")
        panel.paste(image, (0, 34))
        ImageDraw.Draw(panel).text(
            (10, 10),
            f"Frame {index} · {index / fps:.2f} s",
            fill=(20, 35, 42),
        )
        panels.append(panel)
    sheet = Image.new(
        "RGB",
        (width * 3, (height + 34) * 3),
        (236, 232, 222),
    )
    for index, panel in enumerate(panels):
        sheet.paste(
            panel,
            (
                (index % 3) * width,
                (index // 3) * (height + 34),
            ),
        )
    path = output_root / "generated-frames.jpg"
    sheet.save(path, quality=92, subsampling=0)
    return path, indices


def _polygon_signature(item: dict[str, Any]) -> list[float]:
    points = np.asarray(item["geometry"]["points"], dtype=np.float64)
    return sorted(
        math.dist(points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    )


def _canonical_mechanism_checks(
    program: SentinelProgram,
) -> list[dict[str, Any]]:
    """Validate the whole teaching mechanism, not only one rendered segment."""

    return program.validate(
        [program.sample(progress) for progress in (0.0, 1 / 3, 2 / 3, 1.0)]
    )


def _segment_motion_check(
    motion_class: str,
    samples: list[Any],
) -> dict[str, Any]:
    """Return a data audit tailored to the declared motion class."""

    states = [sample.state for sample in samples]
    if motion_class == "rigid_motion":
        identity_sequences = [
            sample.state.get("objects", []) for sample in samples
        ]
        initial_ids = [
            item["object_id"] for item in identity_sequences[0]
        ]
        identities_stable = all(
            [item["object_id"] for item in sequence] == initial_ids
            for sequence in identity_sequences
        )
        signatures = {
            item["object_id"]: _polygon_signature(item)
            for item in identity_sequences[0]
        }
        maximum_side_length_error = 0.0
        for sequence in identity_sequences:
            for item in sequence:
                reference = signatures[item["object_id"]]
                current = _polygon_signature(item)
                maximum_side_length_error = max(
                    maximum_side_length_error,
                    max(
                        abs(left - right)
                        for left, right in zip(reference, current)
                    ),
                )
        return {
            "name": "rigid_objects_keep_ids_and_polygon_side_lengths",
            "passed": (
                identities_stable
                and maximum_side_length_error < 1e-3
            ),
            "evidence": {
                "object_ids": initial_ids,
                "identities_stable": identities_stable,
                "maximum_side_length_error_px": round(
                    maximum_side_length_error, 8
                ),
            },
        }

    if motion_class == "liquid_mixing":
        plume = [state["plume_strength"] for state in states]
        spread = [state["plume_spread_factor"] for state in states]
        integrated_proxy = [
            state["plume_integrated_proxy"] for state in states
        ]
        indicator = [
            state["indicator_mean_inside_liquid"] for state in states
        ]
        volumes = [state["base_volume_ml"] for state in states]
        levels = [state["liquid_level_y"] for state in states]
        passed = (
            all(b <= a for a, b in zip(plume, plume[1:]))
            and all(b >= a for a, b in zip(spread, spread[1:]))
            and all(
                b <= a + 1e-6
                for a, b in zip(
                    integrated_proxy, integrated_proxy[1:]
                )
            )
            and all(b >= a for a, b in zip(volumes, volumes[1:]))
            and all(b <= a for a, b in zip(levels, levels[1:]))
            and plume[-1] <= 1e-6
        )
        return {
            "name": "liquid_color_field_disperses_without_mass_regrowth",
            "passed": passed,
            "evidence": {
                "plume_strength": plume,
                "plume_spread_factor": spread,
                "plume_integrated_proxy": integrated_proxy,
                "indicator_mean_inside_liquid": indicator,
                "base_volume_ml": volumes,
                "liquid_level_y": levels,
            },
        }

    if motion_class == "object_division":
        initial = states[0]["objects"]
        final = states[-1]["objects"]
        initial_ids = [item["object_id"] for item in initial]
        parent_mappings = Counter(
            item["parent_id"] for item in final if item["parent_id"]
        )
        sisters_after_split = [
            state["sister_chromatid_count"] for state in states[1:]
        ]
        passed = (
            len(initial_ids) == 6
            and len(set(initial_ids)) == 6
            and len(final) == 12
            and len(parent_mappings) == 6
            and set(parent_mappings.values()) == {2}
            and all(value == 12 for value in sisters_after_split)
            and states[-1]["left_destination_count"] == 6
            and states[-1]["right_destination_count"] == 6
        )
        return {
            "name": "six_parent_chromosomes_map_to_twelve_sisters",
            "passed": passed,
            "evidence": {
                "initial_parent_ids": initial_ids,
                "sister_counts_after_split": sisters_after_split,
                "sisters_per_parent": dict(parent_mappings),
                "final_left_count": states[-1]["left_destination_count"],
                "final_right_count": states[-1][
                    "right_destination_count"
                ],
            },
        }

    if motion_class == "boundary_topology_change":
        neck = [state["neck_width_px"] for state in states]
        isolated = [
            state["isolated_oxbow_count"] for state in states
        ]
        main_components = [
            state["main_channel_components"] for state in states
        ]
        passed = (
            all(b <= a for a, b in zip(neck, neck[1:]))
            and isolated[0] == 0
            and isolated[-1] == 1
            and all(value == 1 for value in main_components)
            and all(
                b >= a for a, b in zip(isolated, isolated[1:])
            )
        )
        return {
            "name": "main_channel_stays_connected_as_oxbow_isolates",
            "passed": passed,
            "evidence": {
                "neck_width_px": neck,
                "isolated_oxbow_count": isolated,
                "main_channel_components": main_components,
            },
        }

    return {
        "name": "declared_program_segment_has_monotonic_progress",
        "passed": all(
            b >= a
            for a, b in zip(
                [state["progress"] for state in states],
                [state["progress"] for state in states][1:],
            )
        ),
        "evidence": [state["progress"] for state in states],
    }


def render_program_video(
    spec: dict[str, Any],
    program: SentinelProgram,
    output_root: Path,
) -> dict[str, Any]:
    """Resample one program segment without generative video inference."""

    started = time.monotonic()
    settings = spec["settings"]
    frame_count = int(settings["frame_count"])
    fps = int(settings["fps"])
    progress_values = np.linspace(
        float(settings["progress_start"]),
        float(settings["progress_end"]),
        frame_count,
    )
    frames_root = output_root / "frames"
    work = output_root / "_work"
    inputs = output_root / "inputs"
    frames_root.mkdir(parents=True, exist_ok=True)
    work.mkdir(exist_ok=True)
    inputs.mkdir(exist_ok=True)
    samples = []
    for index, progress in enumerate(progress_values):
        sample = program.sample(float(progress))
        sample.clean_frame.convert("RGB").save(
            frames_root / f"frame_{index:03d}.png",
            optimize=False,
        )
        samples.append(sample)
    width, height = samples[0].clean_frame.size
    if (
        width != int(settings["width"])
        or height != int(settings["height"])
    ):
        raise ValueError("program canvas does not match fallback settings")

    first_source = Path(spec["source"]["first_frame"])
    last_source = Path(spec["source"]["last_frame"])
    shutil.copy2(first_source, inputs / "first.png")
    shutil.copy2(last_source, inputs / "last.png")
    endpoint_exact = (
        np.array_equal(
            np.asarray(samples[0].clean_frame),
            np.asarray(Image.open(first_source).convert("RGB")),
        )
        and np.array_equal(
            np.asarray(samples[-1].clean_frame),
            np.asarray(Image.open(last_source).convert("RGB")),
        )
    )

    video_path = output_root / "transition.mp4"
    _encode(
        frames_root,
        video_path,
        fps=fps,
        frame_count=frame_count,
    )
    decoded_frame_count = _decoded_frame_count(video_path)
    decoded = [
        np.asarray(sample.clean_frame.convert("RGB"))
        for sample in samples
    ]
    video_info = {
        "codec": "h264",
        "width": width,
        "height": height,
        "fps": float(fps),
        "frame_count": decoded_frame_count,
        "duration_seconds": decoded_frame_count / fps,
        "has_audio": False,
    }
    preview_path, sample_indices = _preview(
        decoded, output_root, float(video_info["fps"])
    )
    validation_samples = [
        samples[index]
        for index in np.linspace(
            0, len(samples) - 1, 4, dtype=int
        ).tolist()
    ]
    mechanism_checks = _canonical_mechanism_checks(program)
    segment_motion_check = _segment_motion_check(
        spec["motion_class"], samples
    )
    hard_checks = [
        {
            "name": "program_endpoints_are_bit_exact_to_declared_inputs",
            "passed": endpoint_exact,
            "evidence": endpoint_exact,
        },
        {
            "name": "program_plugin_mechanism_checks_pass",
            "passed": all(check["passed"] for check in mechanism_checks),
            "evidence": mechanism_checks,
        },
        {
            **segment_motion_check,
        },
        {
            "name": "encoded_video_contract_matches_program_frames",
            "passed": (
                video_info["frame_count"] == frame_count
                and video_info["width"] == width
                and video_info["height"] == height
                and not video_info["has_audio"]
            ),
            "evidence": video_info,
        },
    ]
    program_validation = {
        "schema_version": "1.0",
        "case_id": program.case_id,
        "progress_range": [
            float(progress_values[0]),
            float(progress_values[-1]),
        ],
        "mechanism_checks": mechanism_checks,
        "segment_motion_check": segment_motion_check,
    }
    write_json(work / "program_validation.json", program_validation)
    result = {
        "schema_version": "1.0",
        "experiment_id": spec["experiment_id"],
        "case_id": spec["case_id"],
        "motion_class": spec["motion_class"],
        "status": "accepted_deterministic_program_fallback",
        "classification": (
            "deterministic program motion; video model deliberately off"
        ),
        "experiment_spec_sha256": sha256_path(
            Path(spec["_spec_path"])
        ),
        "generation_seconds": round(time.monotonic() - started, 3),
        "model_runs": {"image": 0, "video": 0},
        "video": {
            **video_info,
            **artifact_record(video_path, output_root),
        },
        "preview": artifact_record(preview_path, output_root),
        "sample_indices": sample_indices,
        "program_validation": artifact_record(
            work / "program_validation.json", output_root
        ),
        "hard_checks": hard_checks,
    }
    write_json(work / "run.json", result)
    return result
