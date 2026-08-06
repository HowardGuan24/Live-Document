"""Audit and publish the readable Stage 3.11 pedagogical timeline release."""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    verify_file_record,
    write_json,
)
from modules.video_model.stage3.framework.pedagogy import find_cjk_font


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output/phase-11-pedagogy"
CONFIG = STAGE3 / "pedagogy_contracts_v1.json"
CASES = (
    "MATH-01", "MATH-02", "PHYS-01", "PHYS-02", "CHEM-01",
    "CHEM-02", "BIO-01", "BIO-02", "GEO-01", "GEO-02",
)
OLD_VIDEOS = {
    "CHEM-01": STAGE3 / "output/phase-10-release/sentinel-motion/CHEM-01/deterministic/transition.mp4",
    "CHEM-02": STAGE3 / "output/phase-9-scale-motion/CHEM-02/deterministic/transition.mp4",
    "GEO-01": STAGE3 / "output/phase-9-scale-motion/GEO-01/L1/transition.mp4",
}
DELTA_REFERENCE = (
    REPO_ROOT
    / "modules/video_model/stage1/output/causal_delta/delta_causal.mp4"
)


def _uri(path: Path) -> str:
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".mp4": "video/mp4",
    }[path.suffix.lower()]
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _case_map() -> dict[str, dict[str, Any]]:
    config = load_json(CONFIG)
    return {item["case_id"]: item for item in config["cases"]}


def _contact_sheet() -> Path:
    width, cell_height = 1280, 470
    canvas = Image.new("RGB", (width, cell_height * 5), (13, 28, 29))
    draw = ImageDraw.Draw(canvas)
    font = find_cjk_font(22)
    for index, case_id in enumerate(CASES):
        source = Image.open(
            OUTPUT / f"cases/{case_id}/stage-contact-sheet.jpg"
        ).convert("RGB")
        source.thumbnail((620, 414), Image.Resampling.LANCZOS)
        column, row = index % 2, index // 2
        x, y = column * 640 + 10, row * cell_height + 42
        canvas.paste(source, (x, y))
        draw.text(
            (x, row * cell_height + 8), case_id, font=font,
            fill=(242, 196, 96),
        )
    target = OUTPUT / "report-assets/ten-case-stage-contact-sheet.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=91, subsampling=0)
    return target


def _audit() -> dict[str, Any]:
    manifest = load_json(OUTPUT / "phase11-manifest.json")
    replay = load_json(OUTPUT / "reproducibility-audit.json")
    records = []
    for case_id in CASES:
        result_path = OUTPUT / f"cases/{case_id}/result.json"
        result = load_json(result_path)
        verified = []
        error = None
        try:
            for key in (
                "video", "stage_contact_sheet", "timeline",
                "state_render_plan", "render_manifest",
                "story_audit", "mechanism_audit", "appearance_anchor",
            ):
                verify_file_record(result[key], REPO_ROOT)
                verified.append(key)
        except Exception as exc:  # evidence is persisted below
            error = f"{type(exc).__name__}: {exc}"
        duration = float(result["timeline_compile"]["duration_seconds"])
        records.append(
            {
                "case_id": case_id,
                "passed": bool(result["passed"])
                and error is None
                and 8.5 <= duration <= 12.0,
                "duration_seconds": duration,
                "frame_count": result["timeline_compile"]["frame_count"],
                "fps": result["timeline_compile"]["fps"],
                "verified_records": verified,
                "error": error,
                "result": file_record(result_path, REPO_ROOT),
            }
        )
    result = {
        "schema_version": "1.0",
        "phase": "S3.11",
        "case_count": len(records),
        "cases": records,
        "manifest_passed": manifest["passed"] and manifest["case_count"] == 10,
        "replay_passed": replay["passed"],
        "replay_scope": replay["scope"],
        "passed": manifest["passed"]
        and manifest["case_count"] == 10
        and replay["passed"]
        and all(item["passed"] for item in records),
    }
    write_json(OUTPUT / "release-audit.json", result)
    if not result["passed"]:
        raise RuntimeError("Stage 3.11 release audit failed")
    return result


def _update_baselines() -> None:
    path = STAGE3 / "baselines/accepted.json"
    value = load_json(path)
    superseded = {
        "CORE-STATE-RENDERER-B-V3",
        "SCALE-PROGRAM-PROVIDER-S3.9-V1",
    }
    records = [
        item for item in value["records"]
        if item["baseline_id"] not in superseded
    ]
    additions = (
        (
            "CORE-STATE-RENDERER-B-V4",
            "accepted_core",
            STAGE3 / "framework/state_renderer.py",
        ),
        (
            "PEDAGOGY-COMPILER-S3.11-V1",
            "accepted_core",
            STAGE3 / "framework/pedagogy.py",
        ),
        (
            "PEDAGOGY-CONTRACT-S3.11-V1",
            "accepted_core",
            CONFIG,
        ),
        (
            "SCALE-PROGRAM-PROVIDER-S3.11-V2",
            "accepted_core",
            REPO_ROOT / "modules/video_model/stage2/cases/remaining_programs.py",
        ),
        (
            "SENTINEL-PROGRAM-PROVIDER-S3.11-V1",
            "accepted_core",
            REPO_ROOT / "modules/video_model/stage2/cases/sentinel_programs.py",
        ),
        (
            "PEDAGOGY-REPLAY-S3.11-V1",
            "accepted_core_evidence",
            OUTPUT / "reproducibility-audit.json",
        ),
    )
    by_id = {item["baseline_id"]: item for item in records}
    for baseline_id, kind, source in additions:
        by_id[baseline_id] = {
            "baseline_id": baseline_id,
            "kind": kind,
            **file_record(source, REPO_ROOT),
        }
    value["records"] = list(by_id.values())
    write_json(path, value)


def _beats_table(case: dict[str, Any]) -> str:
    rows = "".join(
        "<tr><td>" + html.escape(beat["title_zh"]) + "</td><td>"
        + html.escape(beat["caption_zh"]) + "</td><td>"
        + f"{beat['dynamic_seconds']}s 动态 + {beat['hold_seconds']}s 停留"
        + "</td></tr>"
        for beat in case["beats"]
    )
    return (
        "<table><thead><tr><th>阶段</th><th>观众应理解什么</th>"
        "<th>时间分配</th></tr></thead><tbody>" + rows + "</tbody></table>"
    )


def _case_cards(cases: dict[str, dict[str, Any]]) -> str:
    cards = []
    for case_id in CASES:
        case = cases[case_id]
        result = load_json(OUTPUT / f"cases/{case_id}/result.json")
        story = load_json(OUTPUT / f"cases/{case_id}/story-audit.json")
        jump = next(
            item for item in story["checks"]
            if item["name"] == "scene_has_no_large_single_frame_pixel_jump"
        )["evidence"]["maximum_scene_mae_0_255"]
        cards.append(
            "<section class='case'><h3>" + case_id + " · "
            + html.escape(case["title_zh"]) + "</h3><p>"
            + html.escape(case["summary_zh"]) + "</p><img src='"
            + _uri(OUTPUT / f"cases/{case_id}/stage-contact-sheet.jpg")
            + "' alt='" + case_id + " 四阶段总览'><video controls preload='metadata' src='"
            + _uri(OUTPUT / f"cases/{case_id}/teaching-video.mp4")
            + "'></video><p class='audit'>"
            + f"{result['timeline_compile']['frame_count']} 帧 · "
            + f"{result['timeline_compile']['duration_seconds']} 秒 · "
            + f"最大相邻场景变化 {jump}/255 · 机制门与解释门通过"
            + "</p>" + _beats_table(case) + "</section>"
        )
    return "".join(cards)


def _report(contact: Path, audit: dict[str, Any]) -> Path:
    cases = _case_map()
    chem_exp = OUTPUT / "appearance/CHEM-02/EXP-S3-20260731-038"
    chem_empty = OUTPUT / "appearance/CHEM-02/EXP-S3-20260731-035"
    positive = (chem_exp / "inputs/positive_prompt.txt").read_text(
        encoding="utf-8"
    )
    negative = (chem_exp / "inputs/negative_prompt.txt").read_text(
        encoding="utf-8"
    )
    replay = load_json(OUTPUT / "reproducibility-audit.json")
    replay_rows = "".join(
        f"<tr><td>{item['case_id']}</td><td>{item['file_count']}</td>"
        f"<td><code>{item['run_1_tree_sha256'][:16]}…</code></td>"
        "<td class='ok'>两轮相同</td></tr>"
        for item in replay["checks"]
    )
    comparison = "".join(
        "<div class='compare'><h3>" + case_id + "</h3>"
        "<figure><video controls preload='metadata' src='" + _uri(old)
        + "'></video><figcaption>旧版：约 2 秒、49 帧、没有为解释而分段</figcaption></figure>"
        "<figure><video controls preload='metadata' src='"
        + _uri(OUTPUT / f"cases/{case_id}/teaching-video.mp4")
        + "'></video><figcaption>新版：命名阶段、逐帧程序状态、阅读停留与字幕</figcaption></figure></div>"
        for case_id, old in OLD_VIDEOS.items()
    )
    source = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Stage 3.11：从可动到讲得清、可复现</title><style>
:root{{--ink:#17302e;--muted:#5c6b67;--paper:#f2eee4;--card:#fffdf8;--line:#d4cbbb;--gold:#b16e19;--ok:#147149;--bad:#a33c34}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.72 system-ui,-apple-system,"Noto Sans SC",sans-serif}}main{{max-width:1240px;margin:auto;padding:34px 24px 90px}}h1{{font-size:clamp(34px,5vw,62px);line-height:1.08;max-width:1000px}}h2{{margin-top:2.4em;padding-top:1em;border-top:1px solid var(--line)}}h3{{margin-bottom:.35em}}.lead{{font-size:20px;max-width:980px}}.hero,.case,figure,.note,.flow-step{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:15px}}img,video{{display:block;width:100%;height:auto;border-radius:8px;background:#102725}}figcaption,.muted,.audit{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px}}.flow{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.flow-step b{{display:block;color:var(--gold);font-size:20px}}.arrow{{font-size:24px;color:var(--gold)}}table{{width:100%;border-collapse:collapse;background:var(--card);margin:.7em 0 1.4em}}th,td{{border:1px solid var(--line);padding:10px;text-align:left;vertical-align:top}}pre,code{{background:#e7e1d5;border-radius:5px}}pre{{padding:14px;white-space:pre-wrap;overflow:auto}}code{{padding:.1em .35em}}.ok{{color:var(--ok);font-weight:700}}.bad{{color:var(--bad);font-weight:700}}.case{{margin:22px 0}}.case video{{margin-top:12px}}.compare{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:22px 0}}.compare h3{{grid-column:1/-1}}.compare figure{{margin:0}}.tag{{display:inline-block;background:#e4dccb;border-radius:99px;padding:3px 10px;margin-right:6px}}@media(max-width:760px){{.flow{{grid-template-columns:1fr}}.compare{{grid-template-columns:1fr}}}}</style></head><body><main>
<p>Stage 3 · Phase 11 · 2026-07-31</p><h1>这次解决的不是“再做十个短视频”，而是让每个过程分阶段、连续、可信且能重跑</h1>
<p class='lead'>十个正式案例现在都是 8.5–12 秒的教学时间线。每个阶段有名字、解释、连续运动和阅读停留；图像模型只提供外观，不决定科学状态；三项重点修复完整重跑两次，目录级哈希一致。</p>
<div class='note'><span class='tag'>10/10 案例</span><span class='tag'>约 9–9.5 秒</span><span class='tag'>12 fps</span><span class='tag'>三案 bit-for-bit replay</span><p><b>直接结论：</b>牛轭湖已改成“河曲迁移 → 洪水切颈 → 两端淤塞 → 独立湖体”；滴定终点颜色跨 31 个展示帧变化，不再一帧变红；析晶改用真实 SDXL 晶面像素，但成核阈值、数量与生长仍由程序锁定。</p></div>

<h2>1. 目标长什么样：先看 Stage 1 参考，再看旧版与新版</h2>
<figure><video controls preload='metadata' src='{_uri(DELTA_REFERENCE)}'></video><figcaption>Stage 1 的三角洲参考：约 8.75 秒，五个因果阶段，每段给观众观察时间。这一节奏成为本轮教学时间线的基准，不复制三角洲的具体内容。</figcaption></figure>
{comparison}

<h2>2. 确定下来的通用流程</h2><div class='flow'>
<div class='flow-step'><b>① 分段说明表</b>代码中叫“教学合同”。它写明要讲哪几个阶段、每段说什么、看什么变化以及播放多久。</div>
<div class='flow-step'><b>② 动画程序直接计算事实</b>代码中叫“程序真值”。动画程序按进度计算这一刻有哪些物体、在哪里、数值是多少；不是从截图猜。</div>
<div class='flow-step'><b>③ 外观与状态分工</b>SDXL/参考材质只回答“玻璃、河滩、晶面看起来怎样”；程序层回答“在哪里、何时出现、怎样变化”。</div>
<div class='flow-step'><b>④ 安排观众看到的快慢</b>代码中叫“展示时钟”。变化快的地方多分配几帧，每阶段末停一会儿让人读字幕。</div>
<div class='flow-step'><b>⑤ 按程序状态逐帧画</b>代码中叫“确定渲染”。把固定的真实材质放进这一帧由程序指定的区域，其他地方不乱变。</div>
<div class='flow-step'><b>⑥ 教学画面</b>加入阶段标题、简明解释、阶段计数和进度条，编码为约 10 秒 MP4。</div>
<div class='flow-step'><b>⑦ 科学正确与是否好懂都检查</b>代码中叫“双门检验”：一组检查科学过程，一组检查时长、停留和画面跳变。</div>
<div class='flow-step'><b>⑧ 原样重跑</b>代码中叫“复现”。固定模型图与参数，完整运行两次，检查所有中间文件是否相同。</div></div>

<h3>把上面的通用流程换成一个实际案例</h3>
<div class='note'><p><b>以 CHEM-02“盐溶液蒸发并析晶”为例：</b></p>
<p>先在分段说明表中写四段：水分蒸发 → 接近饱和 → 开始成核 → 晶体生长。动画程序随后为每个时刻直接计算液体还剩多少、浓度多少、是否达到成核条件、现在有几颗晶体。SDXL 生成的空玻璃皿和晶面只提供外观；程序算出的液体区域与晶体区域决定这些材质能出现在哪里。接近成核阈值时多安排一些画面，阶段末停留 0.7 秒。最后检查阈值前没有晶体、晶体数只能增加到四个、溶质总量不变，并把整条流程运行两次比较全部文件。</p></div>

<h3>通用名词与实际产物逐项对照</h3>
<table><thead><tr><th>报告中的名词</th><th>不使用术语时是什么意思</th><th>在实际案例中具体对应什么</th></tr></thead><tbody>
<tr><td><b>教学合同</b></td><td>一张“这段视频怎样讲”的分段说明表。它不计算化学或物理，只规定阶段、字幕、关注内容和时长。</td><td>CHEM-02 中就是四段：蒸发、接近饱和、成核、生长；每段 1.7 秒运动，再停 0.7 秒。它保存在 <code>pedagogy_contracts_v1.json</code>。</td></tr>
<tr><td><b>程序 provider</b></td><td>真正生成程序动画状态的那段代码。给它一个进度，它返回该时刻发生了什么。这里的 provider 可以直接理解成“动画状态生成器”。</td><td>给结晶程序进度 <code>0.75</code>，它返回液体体积、浓度、晶体生长比例和两颗已出现的晶体；给牛轭湖程序进度 <code>0.95</code>，它返回切颈完成、两条沙坝接近完成和一个独立湖体。</td></tr>
<tr><td><b>科学进度 / progress</b></td><td>程序内部从开始到结束的完成比例，范围是 0 到 1。它不是视频秒数。</td><td>结晶进度小于约 0.55 时不允许出现晶体；牛轭湖进度 0.72 后才开始在旧河道两端堆积泥沙。</td></tr>
<tr><td><b>程序真值</b></td><td>动画程序自己算出的事实，而不是语言模型或图片模型看图猜出的结果。</td><td>结晶程序明确算出总溶质质量始终为 <code>0.48</code>；牛轭湖程序直接计算主河有几个连通部分、旧河湾是否已经独立。</td></tr>
<tr><td><b>state JSON</b></td><td>把某一帧的程序真值写成可检查的文字数字文件。</td><td>结晶某帧会写出 <code>crystal_count</code>、<code>concentration</code>、<code>solvent_volume</code>；牛轭湖某帧会写出 <code>breach_fraction</code>、封堵比例和连通数量。</td></tr>
<tr><td><b>语义层</b></td><td>程序额外导出的“哪块地方代表什么”。最终渲染器因此不用从彩色截图重新猜对象。</td><td>结晶案例分别导出液体区域、晶体区域、浓度分布和四颗晶体的身份；牛轭湖分别导出水体区域、泥沙塞区域和水体身份。</td></tr>
<tr><td><b>region</b></td><td>一张只有区域内外之分的黑白范围图，也就是报告后面说的 mask。</td><td><code>chem02_crystal_region</code> 的白色部分表示这一帧真正已经存在的晶体。真实晶面像素只能进入白色部分，所以终态晶体不会提前出现。</td></tr>
<tr><td><b>scalar</b></td><td>每个位置都有一个连续数值的分布图，而不只是“有/没有”。</td><td><code>chem02_concentration</code> 保存器皿各处的浓度；滴定案例的 pH 场保存液滴附近与主体溶液不同的酸碱度。</td></tr>
<tr><td><b>object identity</b></td><td>给同一个物体一个跨帧不变的编号，用来防止它凭空消失、复制或与别的物体交换。</td><td>四颗盐晶体各有固定 ID。即使它们从很小长到完整大小，程序仍知道哪一颗是哪一颗；牛轭湖形成后也会获得独立水体 ID。</td></tr>
<tr><td><b>外观锚 / appearance anchor</b></td><td>一张固定不动的场景底图，负责相机、光线、器皿和背景外观。</td><td>结晶案例使用 SDXL 候选 <code>seed 113503</code> 的空玻璃皿。它负责玻璃反光和台面，但里面没有提前画好的液体与晶体。</td></tr>
<tr><td><b>材质供体</b></td><td>只借用某种材料长什么样的图片，不允许它决定物体的数量和位置。</td><td>结晶案例从 <code>seed 113801</code> 借用晶面像素；每一帧仍由程序的晶体区域裁切，所以候选图里错误的玻璃横梁不会进入成片。</td></tr>
<tr><td><b>展示时钟</b></td><td>把程序进度换成观众实际看到的播放节奏。科学顺序不变，只调整哪里慢一点、哪里停一下。</td><td>滴定在终点附近变化很快，系统把进度 0.94–1.0 分配成 20 个连续变化帧，而不是让溶液在一帧内变红。</td></tr>
<tr><td><b>State Renderer</b></td><td>逐帧合成画面的通用绘图器。输入底图、这一帧的区域和数值，输出一张没有字幕的最终场景图。</td><td>结晶案例中，它先放空玻璃皿，再按液体区域加入透明溶液，最后只在当前晶体区域内加入晶面；牛轭湖中则把水和棕色沙坝写入各自区域。</td></tr>
<tr><td><b>确定渲染</b></td><td>相同输入一定执行相同绘图步骤，不让图片模型在每一帧重新随机发挥。</td><td>第 70 帧的结晶状态无论重跑多少次，都读取同一底图、同一程序区域和同一材质参数，因此得到相同画面。</td></tr>
<tr><td><b>教学画面</b></td><td>在无字幕场景图下面加入阶段标题、解释、阶段序号和进度条后的画面。</td><td>“第三阶段：达到阈值后成核”及下面那句解释，是渲染完成后统一加上的，不会被 SDXL 画进器皿。</td></tr>
<tr><td><b>机制门</b></td><td>按具体学科检查“讲的事情是否真的发生，而且科学上没有矛盾”。</td><td>结晶检查阈值前为 0 颗、数量只能增加、质量守恒；牛轭湖检查切颈在前、沙坝封堵在后、主河始终连通。</td></tr>
<tr><td><b>故事门</b></td><td>检查第一次看的观众是否有时间看懂，以及画面有没有突然跳变。</td><td>每个案例必须在 8.5–12 秒之间，每个阶段既有运动又有阅读停留，同时检查相邻两帧的画面变化不能突然过大。</td></tr>
<tr><td><b>复现 / Replay</b></td><td>把同样的输入从头完整运行两次，比较所有中间文件和最终视频，而不只是肉眼看着相似。</td><td>滴定两轮各比较 908 个文件，结晶和牛轭湖各比较 1020 个文件；三案两轮目录摘要分别完全相同。</td></tr>
</tbody></table>
<p><b>程序视频的利用方式：</b>不是让程序凭空理解像素。程序本来就是动画的生成者，所以它知道河道区域、液体区域、晶体身份、磁铁位置等结构化真值。新流程使用完整连续状态，不再只取四张关键帧。关键帧仍用于讲解分段和外观锚定，连续帧用于保证运动自然。</p>

<h2>3. 结晶实例：从程序图到真实材质视频的完整血缘</h2>
<div class='grid'><figure><img src='{_uri(chem_exp / 'inputs/clean_keyframe.png')}'><figcaption><b>程序终态图。</b>器皿、液面与四个晶体的几何是机制真值；它不是最终外观。</figcaption></figure>
<figure><img src='{_uri(chem_exp / 'controls/dense_canny.png')}'><figcaption><b>dense Canny。</b>先灰度化程序图，用阈值 5/15 找亮暗突变，再膨胀 1 次。白线是给 ControlNet 的结构条件。</figcaption></figure>
<figure><img src='{_uri(chem_exp / 'candidates-labeled.jpg')}'><figcaption><b>四个模型候选。</b>只改变固定 seed；模型、prompt、控制图和参数完全相同。整图都存在液面误读，所以不直接拿任何一张当视频终帧。</figcaption></figure>
<figure><img src='{_uri(chem_exp / 'raw/final_scene_dense_control/seed_113801.png')}'><figcaption><b>入选材质供体。</b>它的晶体位置最贴合程序控制。最终只读取程序晶体区域内的像素，错误的玻璃横梁不会进入成片。</figcaption></figure>
<figure><img src='{_uri(chem_empty / 'raw/empty_dish_t2i_controlnet/seed_113503.png')}'><figcaption><b>空器皿外观锚。</b>由另一轮模型实验生成。它负责稳定玻璃、光线和台面，程序再逐帧加入液体与晶体。</figcaption></figure>
<figure><img src='{_uri(OUTPUT / 'cases/CHEM-02/stage-contact-sheet.jpg')}'><figcaption><b>最终四阶段。</b>蒸发和浓度上升在前；进度 0.55 之前晶体数必须为 0；随后晶核陆续出现并长大，最终四个身份不交换。</figcaption></figure></div>
<video controls preload='metadata' src='{_uri(OUTPUT / 'cases/CHEM-02/teaching-video.mp4')}'></video>

<h3>Canny、ControlNet、SDXL 的输入输出关系</h3><table><tr><th>组件</th><th>输入</th><th>输出/作用</th></tr>
<tr><td>Canny 算法</td><td>程序 RGB 图</td><td>确定性的黑白边缘图；不知道“这是玻璃还是晶体”，只找亮暗突变。</td></tr>
<tr><td>SDXL Canny ControlNet FP16</td><td>黑白边缘图 + 当前去噪潜变量</td><td>在 SDXL 每一步内部计算结构特征/残差并注入主模型；它不单独返回最终图片，也没有一个供本项目再处理的“残差文件”。</td></tr>
<tr><td>SDXL Base 1.0 FP16</td><td>固定 seed 的初始噪声 + 正负提示词 + ControlNet 结构条件</td><td>输出一个 RGB 候选图。候选必须经数量、布局、伪影和语义门检查，不能自动成为真值。</td></tr></table>
<h3>这次真正使用的提示词</h3><p><b>正向：</b></p><pre>{html.escape(positive)}</pre><p><b>反向：</b></p><pre>{html.escape(negative)}</pre>
<table><tr><th>参数</th><th>本轮值</th><th>新人应怎样理解</th></tr>
<tr><td>seed</td><td>113801–113804</td><td>初始噪声编号；同配置复现同一候选，不是科学参数，也不是质量分。</td></tr>
<tr><td>steps</td><td>30</td><td>去噪迭代次数；更多通常更慢，不保证更正确。</td></tr>
<tr><td>guidance / CFG</td><td>6.0</td><td>文字条件的影响强度；太高可能僵硬和过度锐化。</td></tr>
<tr><td>ControlNet scale</td><td>0.58</td><td>Canny 结构约束强度；太低会丢布局，太高会把程序线条做成刻槽。</td></tr>
<tr><td>strength</td><td>未使用</td><td>本轮是 text-to-image ControlNet，不是 img2img；旧 spec 中的 0.5 是无效字段，新 generate.json 已记为 null。</td></tr></table>

<h2>4. 三项重点问题怎样被硬门约束</h2><table><tr><th>案例</th><th>程序修复</th><th>不允许发生什么</th><th>最终证据</th></tr>
<tr><td>CHEM-01 滴定</td><td>用强酸/强碱电荷平衡与水自解离连续求 pH；终点区的展示时钟使用 257 个分析样本重新分配 20 个动态帧。</td><td>pH 在当量点重置、颜色一帧跳红。</td><td>终点中间色跨 31 帧；最大 indicator 步长约 0.091，低于 0.22 门限。</td></tr>
<tr><td>CHEM-02 析晶</td><td>溶剂连续减少；达到阈值后四个稳定晶核分批激活，每个从小到大；真实晶面像素只在当帧晶体 mask 内出现。</td><td>阈值前析晶、晶体数倒退、凭空丢失溶质、模型添加额外晶体。</td><td>阈值前 0 个；计数单调到 4；总溶质质量极差 &lt; 1e-7。</td></tr>
<tr><td>GEO-01 牛轭湖</td><td>河曲两侧真实迁移并缩窄河颈；捷径连续打开；旧河道两端泥沙塞连续增长；用连通分量判断隔离。</td><td>河颈只改数字不改图、旧河道一帧消失、主河断开、未封口先声称牛轭湖形成。</td><td>四机制顺序固定；隔离时 breach=1 且封堵&gt;0.75；主河始终 1 个连通分量，终态恰有 1 个牛轭湖。</td></tr></table>

<h2>5. 十个案例的分阶段总览</h2><figure><img src='{_uri(contact)}'><figcaption>数学、物理、化学、生物、地理各两个案例。每张小图就是报告下方对应视频的阶段终点，不是另画的宣传图。</figcaption></figure>
{_case_cards(cases)}

<h2>6. 可复现性不是“记住 seed”</h2><p>随机图片候选与确定视频生产链分开管理。SDXL 候选固定模型权重哈希、包版本、prompt、Canny 图、seed 和采样参数；一旦选中就成为只读外观输入。其后所有状态和帧由程序确定生成。</p>
<table><tr><th>复现哨兵</th><th>比较文件数</th><th>目录摘要</th><th>结果</th></tr>{replay_rows}</table>
<p>每个目录摘要覆盖 state JSON、semantic JSON/NPY、无字幕渲染帧、字幕帧、审计文件、联系表和 MP4。三案两轮 bit-for-bit 相同。</p>

<h2>7. 从零复现</h2><pre>cd /persistent/workspace-project/Live-Document
# 只有需要重新抽样外观候选时才运行；已有候选会按输入签名复用
/opt/venv/bin/python -m modules.video_model.stage3.phase11_generate_assets --generate

# 生成十个教学时间线与视频
/workspace/comfyui-rocm-env/bin/python -m modules.video_model.stage3.phase11_pedagogy

# 三项重点案例完整重跑两轮并比较目录哈希
/workspace/comfyui-rocm-env/bin/python -m modules.video_model.stage3.phase11_replay

# 发布审计与本报告
/opt/venv/bin/python -m modules.video_model.stage3.phase11_finalize

# 回归测试
/opt/venv/bin/python -m pytest -q modules/video_model/stage2/tests modules/video_model/stage3/tests</pre>
<p class='muted'>模型路径和权重 SHA-256、运行时版本、GPU、候选耗时与峰值显存记录在每个实验的 <code>_work/generate.json</code>。报告的全部图片和视频均是 data URI，移动目录或用 Live Preview 打开不会丢媒体。</p>

<h2>8. 目前的边界</h2><p>这套系统已确定“解释时间线 + 程序真值 + 冻结外观 + 确定渲染”的主流程，但不声称所有抽象概念都应变成照片。数学关系、波场和磁感应更适合可读的受控模拟；器皿、晶体、地貌和显微材料适合增强真实材质。若视频模型不能保持对象身份、守恒量或精确几何，就不让它猜中间帧，而使用完整程序状态渲染。这是按能力边界选择工具，不是隐藏失败。</p>
<p class='audit'>发布审计：{audit['case_count']}/10 案例 · 全部 8.5–12 秒 · 媒体哈希通过 · 重点三案复现通过。</p>
</main></body></html>"""
    path = OUTPUT / "report.html"
    path.write_text(source, encoding="utf-8")
    return path


def run() -> dict[str, Any]:
    audit = _audit()
    contact = _contact_sheet()
    report = _report(contact, audit)
    media = re.findall(
        r"\b(?:src)=['\"]([^'\"]+)",
        report.read_text(encoding="utf-8"),
    )
    if not media or not all(item.startswith("data:") for item in media):
        raise RuntimeError("Stage 3.11 report has a non-embedded media src")
    _update_baselines()
    manifest = {
        "schema_version": "1.0",
        "release_id": "stage3-pedagogical-timeline-2026-07-31",
        "release_class": "validated_pedagogical_timeline_candidate",
        "case_count": 10,
        "cases": list(CASES),
        "audit": file_record(OUTPUT / "release-audit.json", REPO_ROOT),
        "reproducibility_audit": file_record(
            OUTPUT / "reproducibility-audit.json", REPO_ROOT
        ),
        "report": file_record(report, REPO_ROOT),
        "contact_sheet": file_record(contact, REPO_ROOT),
        "report_embedded_media_count": len(media),
        "all_report_media_embedded": True,
        "passed": True,
    }
    write_json(OUTPUT / "release-manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
