# Phase 0 — Concept Scoping

You receive exactly one external input:

```text
{{CONCEPT}}
```

The input is a concept name or short concept description. Do not expect or request any other user-provided fields.

## Fixed product context

- Target audience: middle-school students and general adults encountering the concept for the first time.
- Output language: English.
- Typical video duration: about 10 seconds.
- Allowed duration range: 8–12 seconds, in 0.5-second increments.
- Prefer narrowing the concept over extending the duration beyond 12 seconds.
- The final product will be a short, program-driven explanatory animation.
- The output format is fixed by the workflow. Do not ask the user to choose an output format.

## Your task

Decide whether the concept can be responsibly explained in one short animation.

Choose exactly one `scope_status`:

- `as_is`: the original concept is already narrow enough.
- `narrowed`: the original concept is too broad, but one coherent sub-process can be selected.
- `unsuitable`: the concept cannot be responsibly reduced to a short program-driven explanatory animation.

For `as_is` or `narrowed`:

1. Preserve the original input exactly in `source_concept`.
2. Define one precise `scoped_concept`.
3. Write one `learning_goal` describing what the viewer should understand after watching.
4. Select a duration from 8 to 12 seconds.
5. Write an ordered causal chain of 2–6 steps.
6. List related content intentionally excluded from this animation.
7. List misconceptions that later phases must avoid.

For `unsuitable`:

1. Preserve the original input exactly in `source_concept`.
2. Explain why the concept cannot be responsibly reduced to this format.
3. Do not invent a causal chain, duration, or scoped concept merely to continue the workflow.

## Scoping rules

A good scoped concept:

- expresses one coherent process, relationship, transformation, or mechanism;
- can be shown through visible state changes or spatial relationships;
- has a clear beginning, progression, and result;
- is scientifically or conceptually defensible at the intended audience level;
- does not require covering every branch of a broad topic.

Narrow the concept when:

- it contains several distinct mechanisms or outcomes;
- it would require more than 12 seconds to explain clearly;
- covering it fully would produce a list of facts rather than one causal explanation;
- one sub-process can stand alone without becoming misleading.

Use `unsuitable` only when:

- no meaningful visible process or relationship can be isolated;
- reducing the topic would make the explanation misleading;
- a short program-driven animation would add little or create false confidence.

## Boundaries

Do not decide or describe:

- camera shots, storyboards, scene composition, or visual style;
- program objects, IDs, arrays, masks, geometry, or numerical parameters;
- frame ranges, frame rate, resolution, or rendering methods;
- image-model or video-model prompts;
- implementation details for later phases.

Do not add fields outside the required schema.

## Output

Return only one JSON object that validates against `schema.json`.

Do not wrap the JSON in Markdown.
Do not include commentary before or after the JSON.
