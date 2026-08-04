# Phase 1 — Semantic Animation Contract

You receive one approved Phase 0 artifact:

```text
{{SCOPE_JSON}}
```

The input is the complete contents of `scope.json`.

Run this phase only when `scope_status` is `as_is` or `narrowed`.
If `scope_status` is `unsuitable`, stop the workflow instead of producing a semantic contract.

## Fixed product context

- Target audience: middle-school students and general adults encountering the concept for the first time.
- Output language: English.
- The final product is a short, program-driven explanatory animation.
- Phase 0 has already fixed the concept boundary, learning goal, duration, causal chain, excluded content, and misconceptions.
- Do not broaden the approved scope or reintroduce excluded content.

## Your task

Translate the approved Phase 0 scope into an implementation-independent semantic animation contract.

The contract must state what every later implementation must preserve, regardless of whether it uses splines, graphs, fields, particles, masks, meshes, or another program representation.

Produce:

1. the important conceptual entities;
2. the ordered semantic stages;
3. abstract progress variables;
4. temporal constraints;
5. spatial constraints;
6. visual evidence that must appear in the animation itself;
7. forbidden visual interpretations;
8. authority boundaries among the program, the appearance model, and text.

## Entities

Define only entities needed for the approved explanation.

For each entity:

- `name` identifies the conceptual entity;
- `role` explains why it exists in the mechanism;
- `persistence` explains whether it exists throughout, transforms, advances, or gradually emerges.

Keep entity names conceptual and implementation-independent.

Good:

```text
existing fracture
water
underground cavity
```

Forbidden:

```text
fracture_01
water_mask
480x270 rock array
```

Do not create decorative entities that are not required by the causal chain.

## Stages

Convert the approved causal chain into 2–6 ordered semantic stages.

For each stage:

- `name` is a short semantic label;
- `cause` states why the stage occurs;
- `visible_change` states what the viewer must visibly observe;
- `result` states the semantic condition produced for the next stage.

`visible_change` must describe an observable change, not a camera shot, caption, rendering method, or program algorithm.

Good:

```text
The fracture boundary gradually retreats around the wetted path.
```

Forbidden:

```text
Zoom into the fracture and apply a Gaussian blur.
```

## Progress variables

Define only abstract semantic progress needed to describe gradual change.

Each progress variable must include:

- `name`: lowercase snake_case;
- `meaning`: what conceptual change it tracks;
- `behavior`: qualitative progression and persistence.

Good:

```text
dissolution_progress
increases gradually after water reaches the fracture, then remains complete
```

Forbidden:

```text
dissolution_rate = 0.0075
cavity_threshold = 0.18
```

Do not specify numeric rates, thresholds, units, data types, array shapes, or update equations.

## Constraints

### Temporal constraints

State ordering, continuity, gradualness, and state inheritance.

Examples:

- one condition must precede another;
- a result must emerge over multiple visible moments;
- an entity must not disappear and reappear at stage boundaries.

Do not specify exact frames or timestamps.

### Spatial constraints

State where entities may exist relative to one another and how transformations remain connected.

Examples:

- a path remains inside an allowed region;
- an emerging region remains connected to its source;
- one result appears above, beside, or within another entity.

Do not define coordinates, pixels, masks, control points, or geometry parameters.

## Visual evidence

`required_visual_evidence` must list facts that a viewer can verify from the animation without relying only on captions.

Each item should be suitable for later checking against program probes and final keyframes.

## Forbidden interpretations

Translate Phase 0 misconceptions into concrete visual failures later phases must avoid.

Do not merely repeat vague warnings. State what an incorrect animation would visibly imply.

## Authority boundaries

Use the following fixed division of responsibility:

### Program

The program owns all semantic truth, including:

- entity identity;
- spatial relationships;
- causal order;
- state progression;
- event timing.

### Appearance model

The appearance model may affect only non-semantic appearance, including:

- material texture;
- surface detail;
- lighting;
- non-semantic color variation.

It must not determine geometry, masks, entity identity, causal order, or scientific truth.

### Text

Text may:

- name the process;
- clarify compressed timescales;
- explain events already visible in the animation.

Text must not replace missing visual evidence or repair incorrect geometry.

Express these responsibilities in `authority_boundaries` using concise strings. Do not grant broader authority.

## Boundaries

Do not decide or include:

- program object IDs;
- arrays, masks, semantic layers, or schemas for per-pixel data;
- coordinates, geometry, paths, curves, meshes, or control points;
- numerical parameters, rates, thresholds, units, or data types;
- update equations or current-frame/previous-frame dependencies;
- frame rate, exact frame ranges, or easing functions;
- camera design, scene composition, or visual style;
- image-model or video-model prompts;
- model names, checkpoints, or generation settings;
- rendering algorithms or post-processing methods.

Do not modify the approved `scoped_concept` or `duration_seconds`.
Do not add fields outside the required schema.

## Output

Return only one JSON object that validates against `schema.json`.

Do not wrap the JSON in Markdown.
Do not include commentary before or after the JSON.
