"""Bit-for-bit replay audit for the three Stage 3.11 repair sentinels.

The image model is deliberately outside this replay: accepted SDXL outputs are
frozen inputs with recorded hashes.  This audit reruns the complete downstream
timeline compiler, semantic export, State Renderer, caption compositor and MP4
encoder twice, then compares every generated artifact in each case directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from modules.video_model.stage3.framework.contracts import (
    file_record,
    sha256_path,
    write_json,
)
from modules.video_model.stage3.phase11_pedagogy import OUTPUT, REPO_ROOT, run


CASES = ("CHEM-01", "CHEM-02", "GEO-01")
AUDIT_PATH = OUTPUT / "reproducibility-audit.json"


def _tree_record(root: Path) -> dict[str, Any]:
    records = []
    digest = hashlib.sha256()
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        sha = sha256_path(path)
        size = path.stat().st_size
        records.append({"path": relative, "sha256": sha, "size_bytes": size})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha.encode("ascii"))
        digest.update(b"\n")
    return {
        "root": root.relative_to(REPO_ROOT).as_posix(),
        "file_count": len(records),
        "total_size_bytes": sum(item["size_bytes"] for item in records),
        "tree_sha256": digest.hexdigest(),
        "files": records,
    }


def _snapshot() -> dict[str, dict[str, Any]]:
    return {
        case_id: _tree_record(OUTPUT / f"cases/{case_id}")
        for case_id in CASES
    }


def run_audit() -> dict[str, Any]:
    run(set(CASES))
    first = _snapshot()
    run(set(CASES))
    second = _snapshot()
    checks = []
    for case_id in CASES:
        checks.append(
            {
                "case_id": case_id,
                "passed": first[case_id]["tree_sha256"]
                == second[case_id]["tree_sha256"],
                "run_1_tree_sha256": first[case_id]["tree_sha256"],
                "run_2_tree_sha256": second[case_id]["tree_sha256"],
                "file_count": second[case_id]["file_count"],
                "total_size_bytes": second[case_id]["total_size_bytes"],
            }
        )
    result = {
        "schema_version": "1.0",
        "phase": "S3.11",
        "scope": list(CASES),
        "claim_zh": (
            "冻结的 SDXL 外观图作为输入；从教学时间轴、程序状态、语义层、"
            "确定性材质渲染、字幕合成到 MP4 编码连续运行两次。"
        ),
        "excluded_from_replay_zh": (
            "不重新抽样图片模型候选；候选搜索是有限随机实验，入选图、seed、"
            "prompt、控制图和模型指纹已冻结并由哈希约束。"
        ),
        "run_1": first,
        "run_2": second,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }
    write_json(AUDIT_PATH, result)
    if not result["passed"]:
        raise RuntimeError("Stage 3.11 replay produced different artifact trees")
    return {
        "passed": True,
        "audit": file_record(AUDIT_PATH, REPO_ROOT),
        "checks": checks,
    }


if __name__ == "__main__":
    print(json.dumps(run_audit(), ensure_ascii=False, indent=2))
