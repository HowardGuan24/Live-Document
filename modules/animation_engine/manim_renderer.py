"""Manim implementation of the animation DSL interpreter."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Callable

from .schema import ActionSpec, AnimationSpec, ObjectSpec, wrap_text


class RendererDependencyError(RuntimeError):
    """Raised when an external rendering dependency is unavailable."""


def _manim() -> Any:
    try:
        import manim
    except ImportError as exc:
        raise RendererDependencyError(
            "Manim is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc
    return manim


def _vector(value: list[float]) -> list[float]:
    return [*value, 0.0] if len(value) == 2 else value


def _named_function(name: str, parameters: dict[str, Any]) -> Callable[[float], float]:
    if name == "quadratic":
        center = float(parameters.get("center", 0.0))
        scale = float(parameters.get("scale", 1.0))
        offset = float(parameters.get("offset", 0.0))
        return lambda x: scale * (x - center) ** 2 + offset
    if name == "linear":
        slope = float(parameters.get("slope", 1.0))
        intercept = float(parameters.get("intercept", 0.0))
        return lambda x: slope * x + intercept
    if name == "sin":
        amplitude = float(parameters.get("amplitude", 1.0))
        frequency = float(parameters.get("frequency", 1.0))
        phase = float(parameters.get("phase", 0.0))
        return lambda x: amplitude * math.sin(frequency * x + phase)
    if name == "cos":
        amplitude = float(parameters.get("amplitude", 1.0))
        frequency = float(parameters.get("frequency", 1.0))
        phase = float(parameters.get("phase", 0.0))
        return lambda x: amplitude * math.cos(frequency * x + phase)
    if name == "sigmoid":
        steepness = float(parameters.get("steepness", 1.0))
        center = float(parameters.get("center", 0.0))
        return lambda x: 1.0 / (1.0 + math.exp(-steepness * (x - center)))
    if name == "gaussian":
        mean = float(parameters.get("mean", 0.0))
        sigma = max(float(parameters.get("sigma", 1.0)), 1e-6)
        amplitude = float(parameters.get("amplitude", 1.0))
        return lambda x: amplitude * math.exp(-((x - mean) ** 2) / (2 * sigma**2))
    raise ValueError(f"Unsupported graph function: {name}")


class ManimDSLScene:
    """Facade that creates a concrete Manim Scene class for one specification."""

    def __init__(self, spec: AnimationSpec, asset_root: Path):
        self.spec = spec
        self.asset_root = asset_root
        self.m = _manim()

    def create(self) -> Any:
        spec = self.spec
        asset_root = self.asset_root
        interpreter = self
        manim = self.m

        class GeneratedDSLScene(manim.Scene):
            def construct(scene_self) -> None:
                scene_self.dsl_objects = interpreter.build_registry(spec.objects, asset_root)
                for action in spec.timeline:
                    interpreter.execute(scene_self, action, scene_self.dsl_objects)

        GeneratedDSLScene.__name__ = f"DSLScene_{spec.id}"
        return GeneratedDSLScene()

    def build_registry(self, specs: list[ObjectSpec], asset_root: Path) -> dict[str, Any]:
        registry: dict[str, Any] = {}
        for item in specs:
            if item.type not in {"arrow", "graph", "group", "line"}:
                registry[item.id] = self._build_independent(item, asset_root)
        self._apply_relative_layout(specs, registry)

        # Groups may arrange their members, so links are deliberately built last.
        # The retry loop also supports groups that reference another group.
        priorities = {"group": 0, "graph": 1, "arrow": 2, "line": 2}
        pending = sorted(
            (item for item in specs if item.type in priorities),
            key=lambda item: priorities[item.type],
        )
        while pending:
            progressed = False
            for item in list(pending):
                if self._dependencies(item).issubset(registry):
                    registry[item.id] = self._build_dependent(item, registry)
                    pending.remove(item)
                    progressed = True
            if not progressed:
                unresolved = ", ".join(item.id for item in pending)
                raise ValueError(f"Cyclic or unresolved object dependencies: {unresolved}")
        return registry

    @staticmethod
    def _dependencies(item: ObjectSpec) -> set[str]:
        if item.type == "group":
            return set(item.properties.get("members", []))
        if item.type == "graph":
            return {item.properties["axes"]}
        if item.type in {"arrow", "line"}:
            return {
                value
                for value in (item.properties.get("from"), item.properties.get("to"))
                if isinstance(value, str)
            }
        return set()

    def _finish_object(self, obj: Any, properties: dict[str, Any]) -> Any:
        if "position" in properties:
            obj.move_to(_vector(properties["position"]))
        if "scale" in properties:
            obj.scale(float(properties["scale"]))
        if "opacity" in properties:
            obj.set_opacity(float(properties["opacity"]))
        if "z_index" in properties:
            obj.set_z_index(int(properties["z_index"]))
        return obj

    def _build_independent(self, item: ObjectSpec, asset_root: Path) -> Any:
        m = self.m
        p = item.properties
        color = p.get("color", "#F4F7FB")

        if item.type == "text":
            content = wrap_text(p["content"], int(p.get("wrap_width", 48)))
            obj = m.Text(content, font_size=float(p.get("font_size", 32)), color=color)
            if p.get("max_width") and obj.width > float(p["max_width"]):
                obj.scale_to_fit_width(float(p["max_width"]))
        elif item.type == "formula":
            obj = m.MathTex(p["content"], font_size=float(p.get("font_size", 42)), color=color)
            if p.get("max_width") and obj.width > float(p["max_width"]):
                obj.scale_to_fit_width(float(p["max_width"]))
        elif item.type == "circle":
            obj = m.Circle(
                radius=float(p.get("radius", 0.5)),
                color=color,
                fill_color=p.get("fill_color", color),
                fill_opacity=float(p.get("fill_opacity", 0.0)),
                stroke_width=float(p.get("stroke_width", 4.0)),
            )
        elif item.type == "rectangle":
            obj = m.RoundedRectangle(
                width=float(p.get("width", 2.0)),
                height=float(p.get("height", 1.0)),
                corner_radius=float(p.get("corner_radius", 0.08)),
                color=color,
                fill_color=p.get("fill_color", color),
                fill_opacity=float(p.get("fill_opacity", 0.0)),
                stroke_width=float(p.get("stroke_width", 3.0)),
            )
        elif item.type == "point":
            obj = m.Dot(
                radius=float(p.get("radius", 0.09)),
                color=color,
            )
        elif item.type == "axes":
            obj = m.Axes(
                x_range=p.get("x_range", [-5, 5, 1]),
                y_range=p.get("y_range", [-3, 5, 1]),
                x_length=float(p.get("x_length", 8.0)),
                y_length=float(p.get("y_length", 5.0)),
                tips=bool(p.get("tips", False)),
                axis_config={"color": color, "stroke_width": float(p.get("stroke_width", 2.0))},
            )
            # Manim's numeric coordinate labels use LaTeX internally. Keep them
            # opt-in so ordinary graph animations work without a TeX install.
            if p.get("labels", False):
                obj.add_coordinates(font_size=float(p.get("label_font_size", 20)))
        elif item.type == "image":
            image_path = Path(str(p.get("path", "")))
            if not image_path.is_absolute():
                image_path = asset_root / image_path
            if not image_path.is_file():
                raise ValueError(f"Image object {item.id!r} does not exist: {image_path}")
            obj = m.ImageMobject(str(image_path))
            if p.get("width"):
                obj.scale_to_fit_width(float(p["width"]))
            elif p.get("height"):
                obj.scale_to_fit_height(float(p["height"]))
        else:
            raise ValueError(f"Unsupported independent object: {item.type}")
        return self._finish_object(obj, p)

    def _endpoint(self, value: Any, registry: dict[str, Any], anchor: str) -> Any:
        if isinstance(value, list):
            return _vector(value)
        obj = registry[value]
        getter = getattr(obj, f"get_{anchor}", obj.get_center)
        return getter()

    def _build_dependent(self, item: ObjectSpec, registry: dict[str, Any]) -> Any:
        m = self.m
        p = item.properties
        color = p.get("color", "#F4F7FB")
        if item.type in {"arrow", "line"}:
            start = self._endpoint(p["from"], registry, p.get("from_anchor", "right"))
            end = self._endpoint(p["to"], registry, p.get("to_anchor", "left"))
            klass = m.Arrow if item.type == "arrow" else m.Line
            kwargs = {
                "color": color,
                "stroke_width": float(p.get("stroke_width", 4.0)),
            }
            if item.type == "arrow":
                kwargs["buff"] = float(p.get("buff", 0.1))
            obj = klass(start, end, **kwargs)
        elif item.type == "graph":
            axes = registry[p["axes"]]
            function = _named_function(p["function"], p.get("parameters", {}))
            obj = axes.plot(
                function,
                x_range=p.get("x_range"),
                color=color,
                stroke_width=float(p.get("stroke_width", 4.0)),
            )
        elif item.type == "group":
            obj = m.VGroup(*(registry[member] for member in p["members"]))
            arrangement = p.get("arrange")
            if arrangement:
                directions = {
                    "down": m.DOWN,
                    "left": m.LEFT,
                    "right": m.RIGHT,
                    "up": m.UP,
                }
                obj.arrange(
                    directions[arrangement],
                    buff=float(p.get("buff", 0.35)),
                    aligned_edge=m.LEFT if p.get("aligned_edge") == "left" else m.ORIGIN,
                )
        else:
            raise ValueError(f"Unsupported dependent object: {item.type}")
        return self._finish_object(obj, p)

    def _apply_relative_layout(self, specs: list[ObjectSpec], registry: dict[str, Any]) -> None:
        m = self.m
        directions = {"above": m.UP, "below": m.DOWN, "left_of": m.LEFT, "right_of": m.RIGHT}
        for item in specs:
            relative_to = item.properties.get("relative_to")
            placement = item.properties.get("placement")
            if relative_to and placement in directions:
                registry[item.id].next_to(
                    registry[relative_to],
                    directions[placement],
                    buff=float(item.properties.get("buff", 0.25)),
                )

    def _destination(self, value: Any, registry: dict[str, Any]) -> Any:
        if isinstance(value, str):
            return registry[value].get_center()
        return _vector(value)

    def _compile_animation(self, action: ActionSpec, registry: dict[str, Any]) -> Any:
        m = self.m
        target = registry[action.target]
        p = action.properties
        if action.action == "add":
            return m.FadeIn(target)
        if action.action == "create":
            if isinstance(target, m.ImageMobject):
                return m.FadeIn(target)
            return m.Create(target)
        if action.action == "write":
            return m.Write(target)
        if action.action == "fade_in":
            return m.FadeIn(target)
        if action.action == "fade_out":
            return m.FadeOut(target)
        if action.action == "highlight":
            return m.Indicate(target, color=p.get("color", "#FFD166"), scale_factor=float(p.get("scale_factor", 1.15)))
        if action.action == "grow_arrow":
            return m.GrowArrow(target)
        if action.action == "move":
            return target.animate.move_to(self._destination(p["to"], registry))
        if action.action == "move_by":
            return target.animate.shift(_vector(p["vector"]))
        if action.action == "change_color":
            return target.animate.set_color(p["color"])
        if action.action == "scale":
            return target.animate.scale(float(p["factor"]))
        if action.action == "rotate":
            return m.Rotate(target, angle=float(p.get("angle", math.pi / 2)))
        if action.action == "transform":
            return m.Transform(target, registry[p["replacement"]].copy())
        if action.action == "follow_path":
            return m.MoveAlongPath(target, registry[p["path"]])
        raise ValueError(f"Action {action.action!r} cannot be compiled as an animation")

    def execute(self, scene: Any, action: ActionSpec, registry: dict[str, Any]) -> None:
        if action.parallel:
            animations = [self._compile_animation(child, registry) for child in action.parallel]
            scene.play(*animations, run_time=action.duration)
            return
        if action.action == "wait":
            scene.wait(action.duration)
            return
        scene.play(self._compile_animation(action, registry), run_time=action.duration)


def render_manim(spec: AnimationSpec, job_dir: Path, asset_root: Path) -> Path:
    """Render one validated specification and return the produced MP4 path."""

    manim = _manim()
    media_dir = job_dir / "manim_media"
    output_path = job_dir / "animation.mp4"
    config = {
        "background_color": spec.background_color,
        "disable_caching": True,
        "format": "mp4",
        "frame_rate": spec.output.fps,
        "media_dir": str(media_dir),
        "output_file": "animation",
        "pixel_height": spec.output.height,
        "pixel_width": spec.output.width,
        "preview": False,
        "progress_bar": "none",
        "write_to_movie": True,
    }
    with manim.tempconfig(config):
        scene = ManimDSLScene(spec, asset_root).create()
        scene.render()
        rendered_path = Path(scene.renderer.file_writer.movie_file_path)

    if not rendered_path.is_file() or rendered_path.stat().st_size == 0:
        raise RuntimeError(f"Manim did not produce a valid MP4: {rendered_path}")
    if rendered_path.resolve() != output_path.resolve():
        shutil.copy2(rendered_path, output_path)
    return output_path
