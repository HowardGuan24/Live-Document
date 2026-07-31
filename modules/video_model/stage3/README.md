# Stage 3 deterministic program-to-generation workflow

Current status: **unreleased S3.6 candidate**.

The old `stage3-core-0.1.0-alpha.1` phase-exit claim was superseded because it
did not run the biology and geography representatives through the complete
release path. Start with:

- `output/phase-6-rerun-1/report.html` — BIO-01 full rerun, failures included.
- `workflow.html` — intended program → control → appearance → state → motion flow.
- `loop.md` — autonomous experiment and promotion rules.

BIO-01 now has accepted G2/G3 images and a full-program-timeline realistic
fallback. Both LTX video candidates were rejected. GEO-02 remains blocked by a
provisional Visual Target Package, so S3.6 is still in progress.

Reproduce the zero-model image portion:

```bash
.venv/bin/python -m modules.video_model.stage3.phase6_recovery render
.venv/bin/python -m modules.video_model.stage3.phase6_recovery_finalize
.venv/bin/python -m pytest -q modules/video_model/stage3/tests
```
