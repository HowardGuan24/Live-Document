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

Render the step-by-step formula example without requiring LaTeX:

```powershell
python -m modules.animation_engine modules/animation_engine/examples/formula_derivation.json -o outputs
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
`formula_transform`, `highlight_parts`, `grow_arrow`, `wait`.

Formula objects use `render_mode: "latex"` by default and accept an `isolate`
array for semantic token matching. Use `render_mode: "text"` with Unicode
mathematical symbols when a LaTeX installation is unavailable. The
`formula_transform` action uses `TransformMatchingTex` in LaTeX mode and
`TransformMatchingShapes` in text mode; after each step, later actions continue
to address the current formula through the original target ID.

```json
{
  "objects": [
    {
      "id": "equation",
      "type": "formula",
      "content": "x^2-1=0",
      "isolate": ["x", "1"]
    },
    {
      "id": "factored",
      "type": "formula",
      "content": "(x-1)(x+1)=0",
      "isolate": ["x", "1"]
    }
  ],
  "timeline": [
    {
      "action": "highlight_parts",
      "target": "equation",
      "parts": ["x", "1"]
    },
    {
      "action": "formula_transform",
      "target": "equation",
      "replacement": "factored",
      "duration": 0.8
    }
  ]
}
```

In text mode, matching shapes still animate between expressions, but
`highlight_parts` highlights the complete expression because Pango text does
not retain semantic LaTeX token groups.

The parser also accepts the repository's legacy `ExplanationSpec` containing
string `objects` and `steps`. It normalizes that input to a generic storyboard.
For a precise animation, provide explicit DSL `objects` and `timeline` arrays.
