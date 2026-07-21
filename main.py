"""
live-document 入口 — 文档 → ExplanationSpec 全流程

用法：
    python main.py data/test_paragraphs.json
    python main.py --file document.md
    python main.py --text "梯度下降沿负梯度方向更新参数"
"""

import argparse
import json
import sys
from pathlib import Path

from modules.doc_planner.parser import parse_document
from modules.doc_planner.generator import generate_spec


def process_text(text: str) -> list:
    """处理纯文本，返回 ExplanationSpec 列表"""
    segments = parse_document(text)
    specs = []
    for seg in segments:
        spec = generate_spec(seg)
        if spec is not None:
            specs.append(spec)
    return specs


def process_file(filepath: str) -> list:
    """处理文件，返回 ExplanationSpec 列表"""
    path = Path(filepath)
    if not path.exists():
        print(f"错误：文件不存在 — {filepath}", file=sys.stderr)
        sys.exit(1)

    text = path.read_text(encoding="utf-8")

    # JSON 文件：每个元素是一个段落
    if path.suffix == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            all_text = ""
            for item in data:
                if isinstance(item, dict):
                    section = item.get("section", "")
                    t = item.get("text", "")
                    if section:
                        all_text += f"\n## {section}\n{t}\n"
                    else:
                        all_text += f"\n{t}\n"
                else:
                    all_text += f"\n{item}\n"
            return process_text(all_text)

    return process_text(text)


def main():
    parser = argparse.ArgumentParser(
        description="live-document: 文档 → ExplanationSpec"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("filepath", nargs="?", help="输入文件路径")
    group.add_argument("--file", "-f", help="输入文件路径")
    group.add_argument("--text", "-t", help="直接输入文本")

    parser.add_argument("--pretty", "-p", action="store_true", help="格式化输出 JSON")
    parser.add_argument("--output", "-o", help="输出文件路径（默认 stdout）")

    args = parser.parse_args()

    if args.text:
        specs = process_text(args.text)
    else:
        filepath = args.filepath or args.file
        specs = process_file(filepath)

    result = [spec.__dict__ for spec in specs]
    indent = 2 if args.pretty else None
    output = json.dumps(result, ensure_ascii=False, indent=indent)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"已写入 {len(specs)} 条 ExplanationSpec → {args.output}")
    else:
        print(output)

    suitable = sum(1 for s in specs if s.type != "unsuitable")
    print(f"\n共 {len(specs)} 段，其中 {suitable} 段适合动态化", file=sys.stderr)


if __name__ == "__main__":
    main()
