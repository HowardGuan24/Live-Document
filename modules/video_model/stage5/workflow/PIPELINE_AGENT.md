# Stage 5 Pipeline Controller Agent

Use this prompt to run or resume the formal Stage 5 concept-to-delivery pipeline.

```text
Concept: {{CONCEPT}}
Run ID: {{RUN_ID}}
```

`{{RUN_ID}}` may be left empty for automatic allocation. The concept must be one nonempty line.

## Controller role

You coordinate the existing Phase 0–6 workflow through `modules/video_model/stage5/run_pipeline.py`. You do not replace phase prompts, schemas, deterministic Runtimes, or human approval.

The pipeline preserves immutable attempts, validates and hashes formal outputs, runs existing Runtime CLIs as subprocesses, pauses at the Phase 4 and Phase 5 human gates, and resumes from `run-state.json`.

## Start or resume

1. Read:

   - `modules/video_model/stage5/README.md`;
   - `modules/video_model/stage5/workflow/WORKFLOW.md`.

2. If the requested run does not exist, initialize it:

   ```bash
   python modules/video_model/stage5/run_pipeline.py init \
     --concept "{{CONCEPT}}" \
     --run-id "{{RUN_ID}}"
   ```

   Omit `--run-id` when the placeholder is empty.

3. If the run exists, inspect it without changing state:

   ```bash
   python modules/video_model/stage5/run_pipeline.py status \
     --run modules/video_model/stage5/runs/{{RUN_ID}}
   ```

4. Advance deterministic work until the next boundary:

   ```bash
   python modules/video_model/stage5/run_pipeline.py next \
     --run modules/video_model/stage5/runs/{{RUN_ID}} \
     --until-blocked
   ```

## When the pipeline waits for an Agent

1. Read the exact task path printed by the CLI.
2. Follow only that task's authoritative phase prompt, allowed reads, allowed writes, schema contract, validation, report path, and stop condition.
3. Do not start a later phase or invoke the next Runtime yourself.
4. After writing the required outputs, call `next --until-blocked` again.

The emitted task is the complete scope for that Agent step. Do not broaden it from memory or from a historical run.

## When the pipeline waits for a human

Stop. Do not choose on the user's behalf and do not treat your own recommendation as approval.

Return to the user:

- one concise explanation of the gate;
- the review-packet path and its contact sheets, review MP4, metrics, and recommendation evidence;
- the exact allowed selection IDs printed by the CLI;
- the exact approval command:

```bash
python modules/video_model/stage5/run_pipeline.py approve \
  --run modules/video_model/stage5/runs/{{RUN_ID}} \
  --phase <4-or-5> \
  --selection <allowed-id> \
  --notes "<human review notes>"
```

Wait for the human to supply the selection and notes. After the decision is recorded, resume the same run with `next --until-blocked`.

If the human rejects the evidence, use the owning phase and their reason:

```bash
python modules/video_model/stage5/run_pipeline.py retry \
  --run modules/video_model/stage5/runs/{{RUN_ID}} \
  --phase <owning-phase-number> \
  --reason "<human reason>"
```

## Failure handling

If the CLI reports `failed`:

1. stop advancing;
2. inspect `last_error`, Runtime stdout/stderr, and retained failure artifacts;
3. identify the owning phase using `WORKFLOW.md`;
4. ask for human direction when the route or product choice is not already authorized;
5. use `retry` only after diagnosis.

Never overwrite an attempt or delete downstream files. Retry marks prior registrations superseded and creates new canonical attempt roots.

## Completion

When status is `completed`, run:

```bash
python modules/video_model/stage5/run_pipeline.py verify \
  --run modules/video_model/stage5/runs/{{RUN_ID}}
```

Return these paths:

- final MP4;
- final GIF;
- delivery manifest;
- final evaluation;
- `runs/{{RUN_ID}}/run-report.md`.

Do not paste giant manifests or command transcripts when paths are sufficient.

## Prohibitions

- Do not edit workflow files during a run.
- Do not silently alter registered phase outputs.
- Do not bypass Phase 5 or either human gate.
- Do not overwrite or reuse attempt/output roots.
- Do not infer human approval from metrics or an Agent recommendation.
- Do not continue after an unresolved failure.
- Do not recursively launch another Codex process.
