"""Select an image-enhancement responsibility split from case capabilities."""

from __future__ import annotations

from typing import Any


EXACT_OR_TOPOLOGICAL_TAGS = {
    "exact_geometry",
    "many_object_identity",
    "paired_identity",
    "water_topology",
    "nucleation",
    "instrument_readout",
}


def select_image_route(case: dict[str, Any]) -> dict[str, Any]:
    """Return a data-type route without branching on a case ID."""

    tags = set(case["capability_tags"])
    layers = set(case["primary_layer_types"])
    hard_gates = list(case["case_hard_gates"])

    if {"fixed_terrain", "coupled_fields"} <= tags:
        route_id = "layered_static_scene_and_dynamic_field"
        model_role = (
            "图像模型只生成静态地形材质；程序根据 scalar/vector field "
            "在生成后绘回云雨等动态机制。"
        )
        required_inputs = [
            "static height_or_normal or hard boundary",
            "dynamic scalar/vector field",
            "post-generation overlay plugin",
        ]
        phase3_evidence = (
            "static-terrain plus dynamic-field control scan rejected "
            "one-pass redraw"
        )
    elif tags & EXACT_OR_TOPOLOGICAL_TAGS:
        route_id = "program_geometry_with_region_limited_material"
        model_role = (
            "程序像素保留数量、身份、读数和拓扑；模型只提供允许区域内的"
            "限幅稳健材质残差。"
        )
        required_inputs = [
            "object_identity",
            "allowed material region",
            "protected hard boundary",
        ]
        phase3_evidence = (
            "two disciplines rejected exact-count redraw; constrained "
            "region projection passed"
        )
    elif "continuous_field" in tags or "height_or_normal" in layers:
        route_id = "program_low_frequency_with_robust_material_residual"
        model_role = (
            "程序保存连续场和大尺度形状；模型供体只贡献多种子中位数"
            "中高频材质。"
        )
        required_inputs = [
            "scalar or height field",
            "protected identities",
            "material donor prompt",
        ]
        phase3_evidence = (
            "continuous-field full-redraw boundary and robust residual "
            "experiment"
        )
    elif "transparent_mixing" in tags:
        route_id = "semantic_boundary_t2i_with_program_state_overlay"
        model_role = (
            "语义充分线稿生成固定玻璃器材；液面、颜色场、液滴和教学标记"
            "由程序在后续状态层控制。"
        )
        required_inputs = [
            "semantically sufficient apparatus line art",
            "state region/scalar field",
            "post-generation state overlay",
        ]
        phase3_evidence = "semantic apparatus line art passed 4/4"
    else:
        route_id = "program_first_no_full_frame_redraw"
        model_role = "保持程序结构，先补齐允许区域后再做小规模模型实验。"
        required_inputs = ["hard boundary", "allowed region"]
        phase3_evidence = "conservative fallback"

    return {
        "route_id": route_id,
        "model_role_zh": model_role,
        "required_inputs": required_inputs,
        "hard_gates": hard_gates,
        "phase3_evidence": phase3_evidence,
    }
