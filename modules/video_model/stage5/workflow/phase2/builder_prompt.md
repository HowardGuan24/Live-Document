# Phase 2 — Implementation Builder

You receive one approved Phase 1 artifact:

```text
{{SEMANTIC_CONTRACT_JSON}}
```

The input is the complete contents of `semantic-contract.json`.

You also receive `schema.json`. Your `plan.json` must validate against:

```text
schema.json#/$defs/plan
```

## Goal

Design one program-generation approach for the approved semantic contract, implement it as a parameterized prototype, and define a small set of candidate configurations to test.

This phase does not select a winner and does not generate the final animation.

## Required outputs

Write exactly these primary outputs:

1. `plan.json`
2. `prototype.py`

You may also write a small local smoke-test file if needed, but do not create probe results, `selection.json`, or `executable-spec.json`.

## One implementation attempt

Use the implementation attempt ID supplied by the workflow. For the first run, use:

```text
implementation-attempt-001
```

The implementation method is open-ended. You may use curves, graphs, regions, fields, particles, transforms, meshes, raster operations, or another program representation suitable for the concept.

Do not choose from a fixed algorithm-family enum.

However, all candidates in `plan.json` must use:

- the same `prototype.py`;
- the same core representation method;
- the same semantic interpretation;
- only different configuration values.

If the core algorithm or representation changes, that is a new implementation attempt, not another candidate.

## Step 1 — Define the implementation inside `plan.json`

The `implementation` object must explain:

- how each Phase 1 entity is represented in the program;
- how each abstract progress variable becomes program state;
- which parameters are tunable;
- which Phase 1 truths become structured state invariants for every candidate;
- which internal states the probe must expose;
- which fixed diagnostic samples every candidate must evaluate;
- known risks of the proposed representation;
- findings that would mean the representation itself should be replaced.

Do not freeze final candidate-specific parameter values inside the `implementation` object.

### Semantic state contract

Define `implementation.semantic_state_contract` as the program's compact,
inspectable source of truth with exactly two parts: `fields` and `invariants`.

Each field declaration contains:

- `name`: the exact top-level key returned by `evaluate_state`;
- `kind`: `authoritative` or `derived`;
- `meaning`: one short semantic definition;
- `depends_on`: exact declared field names, empty for authoritative fields and
  non-empty for derived fields;
- `consumers`: only applicable schema-controlled consumer names.

Every declared field must actually be used by its named consumers. Do not add a
field merely to document an implementation detail.

Each invariant declaration contains only:

- `check_id`: the stable ID emitted by `validate_probe`;
- `claim`: a short semantic statement;
- `fields`: exact declared field names involved.

Do not create another free-form list of semantic invariants. Do not place
executable expressions or Python code in `plan.json`.

Equivalent semantic regions must not be generated independently by different
drawing or state procedures and then assumed to match. Renderers, diagnostic
metrics, machine checks, serialization, and later conditioning must consume the
declared state fields rather than reconstructing equivalent geometry.

When a progress variable is zero and the semantic contract requires no change,
the corresponding changed-region field must be exactly empty. Guarantee exact
relationships by construction where possible. A tolerance may measure a
genuinely approximate invariant; it must not conceal a deterministic
definitional inconsistency.

### Probe samples

Declare 3–10 ordered `probe_samples`. Every candidate must use the same samples.

Each sample contains:

- `sample_id`;
- `purpose`;
- values in `[0, 1]` for the Phase 1 abstract progress variables.

The samples should expose the initial state, meaningful intermediate transitions, and the final state. They are diagnostic states, not exact video frames.

## Step 2 — Write `prototype.py`

The prototype must be configuration-driven and deterministic.

For the same:

- prototype code;
- complete config;
- sampled progress values;

it must produce the same raw state and probe images.

### Required Python interface

Expose these functions:

```python
def build_static_scene(config: dict):
    ...
```

Build geometry or other state that does not change across progress samples.

```python
def evaluate_state(static_scene, progress_values: dict, config: dict) -> dict:
    ...
```

Evaluate one diagnostic state using Phase 1 progress values in `[0, 1]`.

```python
def validate_probe(states: list[dict], semantic_contract: dict, config: dict) -> list[dict]:
    ...
```

Validate the complete ordered probe sequence, including cross-sample continuity and monotonicity when relevant. Return machine-check records. Each record must include:

```text
check_id
claim
passed
evidence
worst_sample_id when applicable
```

`validate_probe` must emit every `check_id` declared in
`semantic_state_contract.invariants` and must evaluate the declared state
fields. Runtime treats a missing declared invariant check or a declared field
missing from any returned state as an execution failure. Metadata scalars and
array fields use the same rule: the exact name must be a top-level state key.

Preserve useful structured evidence on failed checks when applicable:

- `sample_id`;
- `observed_count` and `denominator`;
- `location` or bounding-box description;
- `state_fields`;
- whether the failed region is rendered;
- `failure_class`;
- `recommended_return_target`.

Use only schema-defined failure classes and return targets. Deterministic checks
must not make perceptual or pedagogical-quality claims; those remain Reviewer
judgments.

```python
def render_semantic_probe(states: list[dict], output_path: str):
    ...
```

Render a fixed-color contact sheet showing semantic truth.

```python
def render_edge_probe(states: list[dict], output_path: str):
    ...
```

Render deterministic structure edges derived from semantic geometry, without text edges.

```python
def render_program_probe(states: list[dict], output_path: str):
    ...
```

Render a simple teaching-oriented contact sheet without generative models.

### Shared-core requirement

The geometry and state functions used by the probe must be reusable by the later full animation.

Do not write a disposable probe-only simulation and then plan to replace it with unrelated full-run code.

After all candidates run, runtime distinguishes a check that fails for only
some candidates from a check that fails for every completed candidate. It
records that scope and preserves the distinct `failure_class` and
`recommended_return_target` values reported by the prototype. Runtime does not
replace those recommendations or diagnose the root cause. Do not stop after the
first machine-gate failure, edit the Builder artifacts automatically, or rerun
candidates automatically.

## Step 3 — Define candidate configurations inside `plan.json`

Plan an adaptive first round of 1–3 candidates:

- use 1 when no meaningful unresolved visual-semantic uncertainty exists;
- use 2 for one clear tradeoff;
- use at most 3 only when multiple high-impact hypotheses genuinely need
  comparison.

Each candidate must include:

- a unique `candidate_id`;
- a concrete hypothesis;
- only the parameter overrides that distinguish it from shared `fixed_parameters`.

All config parameter names must be declared in `implementation.tunable_parameters`.

At least one candidate should be a baseline. Other candidates should have an interpretable relationship to it.

Candidate search is justified only when a parameter may materially affect
required visual evidence, a forbidden interpretation, downstream interpretation
risk, or program readability. Do not create candidates merely to vary layout
constants, font size, panel gap, canvas size, or values that deterministic rules
can fix directly.

Every hypothesis must explain why the varied parameter cannot be fixed directly
and what visible consequence is expected.

Good examples:

```text
Curvature cannot be fixed from the semantic contract alone; comparing two
values tests whether the visible path reads as connected rather than rigid.
A secondary branch is semantically optional but affects network readability;
testing its presence reveals whether it clarifies or distracts.
```

Bad examples:

```text
Try another random value.
This one might look better.
```

Do not vary every parameter at once unless the candidate is explicitly the baseline.

Parameters may control geometry, progression shape, presentation reconstruction, or deterministic randomness, but they must not alter the approved causal meaning.

## Forbidden behavior

Do not:

- call an image model or video model;
- generate the final video;
- hard-code candidate-specific values inside `prototype.py`;
- allow captions to create missing semantic events;
- silently change the Phase 1 semantic contract;
- choose a winning candidate;
- write `selection.json` or `executable-spec.json`.

## Quality target

The outputs must make it possible for `runtime.py run-candidates` to:

1. merge the fixed and overridden parameters;
2. run every candidate using the same prototype;
3. evaluate the same diagnostic samples;
4. produce semantic, edge, and program probes;
5. run machine checks;
6. save one `probe-result.json` for each candidate.

## Final response

After writing the files, report only:

- the two output paths;
- a one-paragraph summary of the implementation direction;
- the number of planned candidates;
- any unresolved implementation risk.

Do not claim that the implementation or any candidate is accepted before real probe execution.
