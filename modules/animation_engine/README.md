# Animation Engine

This module turns a validated JSON animation DSL into Manim MP4 and a
palette-optimized GIF. It never executes Python supplied by the JSON.

## Install

```powershell
python -m pip install -r requirements.txt
```

Manim needs a working LaTeX installation when the DSL uses `formula` objects or
enables numeric coordinate labels on `axes`. Other included object types do not
require LaTeX. FFmpeg is resolved from `PATH` first and falls back to the binary
bundled by `imageio-ffmpeg`.

## Run

Validate without rendering:

```powershell
python -m modules.animation_engine modules/animation_engine/examples/data_flow.json --validate-only
```

Render MP4, GIF, normalized JSON and result metadata:

```powershell
python -m modules.animation_engine modules/animation_engine/examples/data_flow.json -o outputs
```

Render the gradient descent example:

```powershell
python -m modules.animation_engine modules/animation_engine/examples/gradient_descent.json -o outputs
```

Artifacts are written to `outputs/<id>/`:

```text
animation.mp4
animation.gif
normalized_spec.json
result.json
```

## DSL surface

Objects: `text`, `formula`, `circle`, `rectangle`, `point`, `axes`, `graph`,
`arrow`, `line`, `group`, `image`.

Actions: `add`, `create`, `write`, `fade_in`, `fade_out`, `move`, `move_by`,
`highlight`, `change_color`, `scale`, `rotate`, `transform`, `follow_path`,
`grow_arrow`, `wait`.

The parser also accepts the repository's legacy `ExplanationSpec` containing
string `objects` and `steps`. It normalizes that input to a generic storyboard.
For a precise animation, provide explicit DSL `objects` and `timeline` arrays.
