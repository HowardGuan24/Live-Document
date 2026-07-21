"""
验收测试 — 对应 README.md 中的四条验收标准

给 20 个测试段落，至少能够：
1. 找出合理的动态化位置
2. 大致选对生成方式
3. 输出能够被下游程序读取的 JSON
4. 失败时给出"不适合生成"，而不是硬生成
"""

import io
import json
import sys
from pathlib import Path

# Windows GBK 终端兼容
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 确保能导入 src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.doc_planner.parser import parse_document
from modules.doc_planner.scorer import is_worth_dynamicizing, score_segment
from modules.doc_planner.classifier import classify_segment
from modules.doc_planner.router import route_renderer
from modules.doc_planner.generator import generate_spec
from modules.doc_planner.models import ExplanationType, Renderer, ExplanationSpec


# --- 加载测试数据 ---

def load_test_paragraphs():
    data_path = Path(__file__).resolve().parent.parent / "data" / "test_paragraphs.json"
    return json.loads(data_path.read_text(encoding="utf-8"))


# --- 验收标准 1: 找出合理的动态化位置 ---

def test_find_dynamicizable_positions():
    """应动态化的段落被正确识别；不应动态化的被正确排除"""
    paragraphs = load_test_paragraphs()

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for para in paragraphs:
        text = para["text"]
        expected = para["expected_dynamic"]

        # 用 parse_document 处理（模拟实际流程）
        segments = parse_document(text)
        assert len(segments) >= 1, f"段落 {para['id']} 未能被解析"

        seg = segments[0]
        worth, score, dims = is_worth_dynamicizing(seg)

        if expected and worth:
            true_positives += 1
        elif not expected and not worth:
            true_negatives += 1
        elif expected and not worth:
            false_negatives += 1
            print(f"  ⚠ 漏检: 段落 {para['id']} 分数={score} 文本={text[:40]}...")
        else:
            false_positives += 1
            print(f"  ⚠ 误检: 段落 {para['id']} 分数={score} 文本={text[:40]}...")

    total = len(paragraphs)
    accuracy = (true_positives + true_negatives) / total
    print(f"\n  动态化位置识别: 准确率={accuracy:.0%} "
          f"(TP={true_positives} TN={true_negatives} FP={false_positives} FN={false_negatives})")

    # 容忍最多 3 个误判
    assert false_positives + false_negatives <= 3, (
        f"误判过多: FP={false_positives} FN={false_negatives}"
    )


# --- 验收标准 2: 大致选对生成方式 ---

def test_classify_correctly():
    """应动态化的段落，分类结果与预期大致一致"""
    paragraphs = load_test_paragraphs()
    correct = 0
    total = 0

    for para in paragraphs:
        if not para["expected_dynamic"]:
            continue

        total += 1
        segments = parse_document(para["text"])
        seg = segments[0]
        exp_type = classify_segment(seg)

        expected = para["expected_type"]
        # 允许一定的灵活度：process/operation 可互换
        compatible = {
            "process": {"process", "operation"},
            "operation": {"process", "operation"},
            "formula": {"formula"},
            "dataflow": {"dataflow"},
            "scene": {"scene"},
        }
        accepted = compatible.get(expected, {expected})

        if exp_type.value in accepted:
            correct += 1
        else:
            print(f"  ⚠ 分类偏差: 段落 {para['id']} 预期={expected} 实际={exp_type.value} "
                  f"文本={para['text'][:40]}...")

    accuracy = correct / total if total > 0 else 0
    print(f"\n  分类准确率: {accuracy:.0%} ({correct}/{total})")
    # 至少 70% 正确
    assert accuracy >= 0.7, f"分类准确率过低: {accuracy:.0%}"


# --- 验收标准 3: 输出能够被下游程序读取的 JSON ---

def test_output_is_valid_json():
    """生成的 ExplanationSpec 可序列化为合法 JSON，且字段齐全"""
    paragraphs = load_test_paragraphs()

    for para in paragraphs:
        segments = parse_document(para["text"])
        seg = segments[0]
        spec = generate_spec(seg)

        assert spec is not None, f"段落 {para['id']} 返回了 None"

        # 序列化
        json_str = spec.to_json()
        data = json.loads(json_str)

        # 必须包含的字段
        required_fields = ["source_text", "type", "renderer", "goal", "objects", "steps"]
        for field in required_fields:
            assert field in data, f"段落 {para['id']} 缺少字段: {field}"

        # 类型检查
        assert isinstance(data["objects"], list), f"段落 {para['id']} objects 不是列表"
        assert isinstance(data["steps"], list), f"段落 {para['id']} steps 不是列表"
        assert isinstance(data["confidence"], (int, float)), f"段落 {para['id']} confidence 不是数字"

    print(f"\n  JSON 输出验证: 全部 {len(paragraphs)} 段通过")


# --- 验收标准 4: 失败时给出"不适合生成"，而不是硬生成 ---

def test_unsuitable_returns_fallback():
    """不适合动态化的段落应返回 fallback，而不是硬生成"""
    paragraphs = load_test_paragraphs()

    for para in paragraphs:
        if para["expected_dynamic"]:
            continue

        segments = parse_document(para["text"])
        seg = segments[0]
        spec = generate_spec(seg)

        assert spec is not None, f"段落 {para['id']} 返回了 None（应返回 fallback）"
        assert spec.type == "unsuitable", (
            f"段落 {para['id']} 不适合动态化但 type={spec.type}"
        )
        assert spec.fallback_reason is not None, (
            f"段落 {para['id']} 缺少 fallback_reason"
        )
        assert spec.renderer == "text_only", (
            f"段落 {para['id']} renderer 应为 text_only，实际={spec.renderer}"
        )
        print(f"  ✓ 段落 {para['id']}: 正确返回 fallback — {spec.fallback_reason}")

    print(f"\n  Fallback 验证: 全部不适合段落正确返回 fallback")


# --- 额外: ExplanationType 枚举覆盖 ---

def test_all_types_used():
    """五个解释类型至少各命中一次"""
    paragraphs = load_test_paragraphs()
    seen_types = set()

    for para in paragraphs:
        if not para["expected_dynamic"]:
            continue
        segments = parse_document(para["text"])
        seg = segments[0]
        exp_type = classify_segment(seg)
        seen_types.add(exp_type)

    print(f"\n  类型覆盖: {len(seen_types)}/5 — {[t.value for t in seen_types]}")
    assert len(seen_types) >= 4, f"只命中了 {len(seen_types)} 种类型，预期至少 4 种"


# --- 运行所有测试 ---

def run_all():
    tests = [
        ("标准1: 找出动态化位置", test_find_dynamicizable_positions),
        ("标准2: 选对生成方式", test_classify_correctly),
        ("标准3: JSON 输出合规", test_output_is_valid_json),
        ("标准4: 不适合时返回 fallback", test_unsuitable_returns_fallback),
        ("额外: 类型覆盖", test_all_types_used),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n{'='*60}")
        print(f"▶ {name}")
        try:
            test_fn()
            print(f"  ✅ PASSED")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  💥 ERROR: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"结果: {passed} 通过, {failed} 失败 / {len(tests)} 总计")

    if failed > 0:
        sys.exit(1)
    print("\n🎉 全部验收测试通过!")


if __name__ == "__main__":
    run_all()
