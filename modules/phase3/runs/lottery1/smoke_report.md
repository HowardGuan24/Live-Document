# Phase 3 Smoke Report — lottery1

## Scope

- Phase 1 motion source: the brief, cutoff event, 20.3–26.3 second interval, overlay, and subtitles from `phase1/runs/lottery1`.
- Phase 2 visual source: the realistic `narrow_neck` and `cutoff_channel` anchors from `phase2/runs/lottery1`.
- Tested transition: `narrow_neck → cutoff_channel`.
- Required motion: floodwater erodes only the narrow neck and progressively establishes a shortcut; the old bend remains water-filled and connected at the endpoint.

## Execution

- Engine: local ComfyUI LTX-2.3 First/Last Frame.
- Model: `ltx-2.3-22b-dev-fp8.safetensors` with the distilled 1.1 LoRA.
- Working specification: 512×288, 24 fps, 97 frames, 4.0417 seconds.
- GPU execution was serial. The lowest reported free VRAM was approximately 15.9 GiB, with no OOM.
- This was a single-stage smoke configuration; no two-stage high-resolution refinement was run.

## Attempts

The first attempt is preserved as `segments/01/attempt-1.mp4`. It used guide strength 0.7 and image compression 25. A moving dark wet patch appeared inside the meander loop, while the neck change remained too diffuse.

The single targeted retry raised guide strength to 0.85, lowered image compression to 10, and explicitly locked the interior grassland. It removed the new wet patch and kept the camera, banks, and vegetation sufficiently stable, so it became `segments/01/video.mp4`.

## English delivery

`smoke_base_video.mp4` remains the text-free generated clip. `smoke_final_video.mp4` now uses the English teaching overlay and English subtitle from the accepted `01` segment. The original Chinese delivery is preserved outside the submission run under `phase3/runs/.language-backups/lottery1-zh`.

Known limitation: the cutoff opening remains visually subtle at 512×288, and the banks show slight generative breathing. A future quality pass should introduce an intermediate anchor showing the first shallow breach instead of relying only on a higher resolution.
