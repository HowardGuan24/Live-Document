# Phase 3 Complete Run Report — lottery1

## Sources and scope

- Phase 1 motion source: the brief, Bridge moments, cutoff event, overlays, and subtitles in `phase1/runs/lottery1`.
- Phase 2 visual anchors: `gentle_meander`, `expanded_meander`, `narrow_neck`, `cutoff_channel`, and `oxbow_lake`.
- Phase 1 and Phase 2 remained unchanged.
- The original one-segment smoke outputs remain available through the language backup.

## Complete timeline

1. `meander_growth`: a gentle bend expands into a mature meander, 97 frames.
2. `neck_narrowing`: the meander neck narrows while the land barrier remains intact, 73 frames.
3. `01`: floodwater breaches the neck and establishes a shortcut channel, 97 frames; this uses the targeted smoke retry.
4. `partial_isolation`: sediment begins accumulating at both entrances without sealing them completely, 73 frames.
5. `oxbow_completion`: both entrances close and the crescent-shaped water body separates from the main river, 73 frames.

Assembly removes one duplicate first frame from each segment after the first, producing 409 frames at 24 fps, or 17.0417 seconds.

## Derived intermediate anchor

`partial_isolation` prevents LTX from having to create sediment bars and close both channel entrances in one transition.

- Structural source: the Phase 1 clean frame at 30.5 seconds.
- Visual source: the Phase 2 `world_reference.png`.
- Generation prompt: `derived_anchors/partial_isolation/prompt.txt`.
- Meaning: the shortcut carries the main flow, while the old bend remains water-filled through two small openings.

The derived anchor exists only in Phase 3 and was not written back to the Phase 2 manifest.

## LTX generation

- Local ComfyUI LTX-2.3 First/Last Frame workflow.
- `ltx-2.3-22b-dev-fp8.safetensors` with the distilled 1.1 LoRA.
- All segments are 512×288 at 24 fps, with guide strength 0.85 and image compression 10.
- GPU tasks ran serially; the lowest reported free VRAM was approximately 15.9 GiB, with no OOM.
- The four non-smoke segments were accepted on their first generation attempt.

## English delivery optimization

The LTX videos were not regenerated. Every teaching overlay, stage label, subtitle, segment annotation, preview, and final composition was rebuilt in English from the existing text-free base videos. The revised layout uses a compact topic tag, a stage badge, one targeted arrow callout, and a single-line subtitle so the river remains visible. The smoke final was updated from the English `01` segment.

Known limitations: the geometric changes in `neck_narrowing` and `01` remain subtle at 512×288, and some banks and grass show slight generative breathing. This run validates the complete Phase 3 chain but is not a two-stage high-resolution release.
