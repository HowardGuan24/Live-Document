from modules.video_model.stage2.framework.contracts import load_json
from modules.video_model.stage2.phase9_ab_lineage import (
    MANIFEST_PATH,
    REPORT_PATH,
    build,
)


def test_phase9_lineage_report_passes() -> None:
    manifest = build(check_only=True)
    assert manifest["status"] == "passed"
    assert manifest["model_runs"] == {"image": 0, "video": 0}
    assert all(item["passed"] for item in manifest["checks"])


def test_phase9_records_only_one_direct_phase7_a_to_b_case() -> None:
    manifest = load_json(MANIFEST_PATH)
    assert "CHEM-01" in manifest["actual_relation_zh"]
    assert all(
        not item["direct_phase7_A_to_B"]
        for item in manifest["other_actual_lineages"]
    )


def test_phase9_report_is_beginner_readable_and_reproducible() -> None:
    text = REPORT_PATH.read_text(encoding="utf-8")
    required = (
        "一张 A 底图，怎样变成",
        "程序先决定“发生什么”",
        "mask（遮罩）实际是什么",
        "pH scalar field（标量场）实际是什么",
        "A 只生成“真实烧杯长什么样”",
        "为什么只冻结一张",
        "B 怎样把程序状态放进真实底图",
        "其他案例当时到底有没有 A→B",
        "放回通用项目后的真实运行规则",
        "怎样复现这条实际链",
        "Canny 会先计算图像亮度梯度",
        "多层控制残差，不是 RGB 图片",
        "ControlNet 不替代 SDXL",
        "SDXL Base UNet",
        "SDXL VAE",
        "FP16",
        "作为 img2img 初始图片送进 SDXL",
        "strength: 0.5",
        "先直接回答这三个问题",
        "没有自动文件派生关系",
        "固定像素坐标",
        "标准链路",
        "本次接受链路",
        "down_block_additional_residuals",
        "参数字典：每个数到底控制什么",
        "guidance_scale",
        "controlnet_conditioning_scale",
        "control_guidance_start/end",
        "把本次选中结果还原成一次模型调用",
        "误导性元数据",
    )
    assert all(item in text for item in required)
    assert REPORT_PATH.stat().st_size > 12_000
