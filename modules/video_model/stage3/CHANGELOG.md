# Stage 3 changelog

## Unreleased — S3.6 rerun 1

- Superseded the invalid `0.1.0-alpha.1` phase-exit claim; the old output is
  retained as historical evidence.
- Added case-agnostic `region_material` and `identity_stroke` operators.
- Completed BIO-01 image G2/G3 with CHEM-01, MATH-02 and delta regressions.
- Rejected BIO-01 LTX L1 and L2 videos for topology/identity failures.
- Added a 49-frame full-program-timeline State Renderer fallback that carries
  the accepted material through deterministic motion.
- Kept S3.6 in progress because GEO-02 is still provisional and incomplete.


## 0.1.0-alpha.1 — 2026-07-31

- Froze eleven versioned input contracts: ten scale cases and one historical
  delta regression.
- Implemented three geometry policies: `preserve_exact`, `canonicalize`, and
  `layout_only`.
- Added fixed SDXL + Canny ControlNet candidate matrices and deterministic
  hard-gate selection.
- Added a data-driven prompt compiler and primitive-declared landmark gates.
- Added deterministic State Renderer B operators for region, scalar field,
  object-local material, and height/normal rendering.
- Evaluated LTX-2.3 motion levels with five candidates and seven model calls.
  Continuous-field propagation defaults to L1; liquid mixing and exact rigid
  identity fall back to deterministic program motion when G4 fails.
- Preserved rejected image/video candidates and their measurements.
- Added a zero-model release audit and an explicit cross-discipline maturity
  matrix.

Known release limits:

- BIO-01 and GEO-02 have not completed the Stage 3 G2–G4 back half.
- GEO-02's Visual Target Package is provisional.
- Five scale cases still have missing Visual Target Packages.
- The deployed LTX first/last-frame workflow cannot directly consume program
  video, object tracks, masks, trajectories, or motion fields.
- Deterministic motion fallbacks preserve mechanism but do not yet carry every
  accepted realistic material through all program frames.
