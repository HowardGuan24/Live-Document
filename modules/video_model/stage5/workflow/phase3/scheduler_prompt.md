# Stage 5 Phase 3 Schedule Author Prompt

## Role

Create exactly one `schedule.json` for the formal Phase 3 Runtime. You allocate presentation time between already approved Phase 2 probe anchors. You do not create per-frame state or change semantic meaning.

## Inputs

- the approved Phase 1 `semantic-contract.json`;
- the frozen Phase 2 `executable-spec.json`;
- the Phase 2 `plan.json`, especially its ordered `implementation.probe_samples`;
- `workflow/phase3/schema.json`, definition `$defs.schedule`;
- fixed product FPS `12`.

Use the approved `duration_seconds` from the semantic contract. Set `frame_count` to exactly `duration_seconds × 12`; the current contract permits 96–144 frames for 8–12 seconds.

## Required output

Write one raw JSON object with exactly:

```text
duration_seconds
fps
frame_count
start_hold_frames
anchors
segments
end_hold_frames
```

Copy every Phase 2 probe sample into `anchors` exactly:

- `anchor_id` is the unchanged `sample_id`;
- `progress_values` contains the unchanged key/value object;
- preserve anchor count and order;
- add, remove, merge, or interpolate no anchor.

Emit every consecutive anchor pair once in `segments`. For each segment, choose only:

- a positive integer `transition_frames`;
- `easing` equal to `linear` or `smoothstep`.

You may also choose nonnegative `start_hold_frames` and `end_hold_frames`.

The exact allocation rule is:

```text
start_hold_frames
+ sum(segment.transition_frames)
+ end_hold_frames
= frame_count
```

A transition emits frames after its source and includes its exact destination as its final transition frame.

## Ownership and prohibitions

- Preserve the frozen configuration, anchor values, order, duration, FPS, and semantic contract.
- Allocate enough time for required visual evidence to become perceptible.
- Do not author per-frame values; Phase 3 Runtime interpolates the explicit schedule.
- Do not add teaching copy, camera instructions, rendering instructions, appearance instructions, model settings, or commentary to the JSON.
- Do not use concept-specific constants from another run.

## Validation and stop condition

Validate against `$defs.schedule`, confirm exact anchor equality with the Phase 2 plan, and confirm the frame allocation equation. Write the required phase report, then stop before invoking `build-sequence`.
