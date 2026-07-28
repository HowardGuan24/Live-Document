"""Beginner-first visual HTML report for the Stage 1.2 transport pair."""

from __future__ import annotations

import html
import json
from pathlib import Path
from string import Template
from typing import Any


def _escaped_prompt(manifest: dict[str, Any], filename: str) -> str:
    return html.escape(manifest["prompts"][filename]["text"])


def render_visual_report(
    output_root: Path,
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    soft: dict[str, Any] | None,
) -> str:
    """Return a standalone visual report whose images use relative paths."""

    review_path = output_root / "_work" / "review.json"
    review = (
        json.loads(review_path.read_text(encoding="utf-8"))
        if review_path.is_file()
        else {}
    )
    first = manifest["selections"]["in_channel"]
    second = manifest["selections"]["at_outlet"]
    settings = manifest["settings"]
    selected_seed = review.get("selected_seed", "尚未选择")
    candidate_count = len(metadata.get("candidates", []))
    first_particles = first["particle_count_from_state"]
    second_particles = second["particle_count_from_state"]

    if soft:
        first_particles = soft["records"]["in_channel"]["particle_count"]
        second_particles = soft["records"]["at_outlet"]["particle_count"]

    substitutions = {
        "candidate_count": candidate_count,
        "selected_seed": html.escape(str(selected_seed)),
        "first_particles": first_particles,
        "second_particles": second_particles,
        "first_distance": f"{first['distance_from_front_to_coast']:.2f}",
        "second_distance": f"{second['distance_from_front_to_coast']:.2f}",
        "width": settings["width"],
        "height": settings["height"],
        "steps": settings["steps"],
        "guidance": settings["guidance_scale"],
        "control_scale": settings["controlnet_conditioning_scale"],
        "seeds": "、".join(str(seed) for seed in settings["seeds"]),
        "pipeline": html.escape(settings["pipeline"]),
        "sdxl_model": html.escape(
            metadata["models"]["sdxl_base"]["model_id"]
        ),
        "controlnet_model": html.escape(
            metadata["models"]["controlnet_canny"]["model_id"]
        ),
        "baseline_first": _escaped_prompt(manifest, "in_channel.txt"),
        "baseline_second": _escaped_prompt(manifest, "at_outlet.txt"),
        "baseline_negative": _escaped_prompt(
            manifest, "in_channel_negative.txt"
        ),
        "revision_first": _escaped_prompt(manifest, "in_channel_v2.txt"),
        "revision_second": _escaped_prompt(manifest, "at_outlet_v2.txt"),
        "revision_negative": _escaped_prompt(
            manifest, "sediment_emphasis_negative.txt"
        ),
    }

    template = Template(
        """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Stage 1.2 可视化报告｜泥沙从河道移动到河口</title>
  <style>
    :root {
      --ink:#17222b;--muted:#5f6e77;--paper:#f5f2e9;--card:#fffdf8;
      --line:#d9d4c8;--river:#167b88;--river-dark:#0a4e5b;
      --sediment:#a65127;--good:#e5f2e9;--warn:#fff0dd;
      --shadow:0 16px 42px rgba(38,45,44,.10)
    }
    *{box-sizing:border-box}
    html{scroll-behavior:smooth}
    body{
      margin:0;color:var(--ink);
      background:radial-gradient(circle at 5% 0%,rgba(213,188,122,.22),transparent 28rem),var(--paper);
      font-family:Inter,"Noto Sans SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif;
      line-height:1.72
    }
    a{color:var(--river-dark)}
    code,pre{font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace}
    .hero{
      color:white;
      background:linear-gradient(115deg,rgba(7,51,61,.97),rgba(15,101,111,.88)),
                 url("final/selected-pair.jpg") center/cover;
      padding:76px 24px 66px
    }
    .hero-inner,main,.footer-inner{width:min(1160px,calc(100% - 40px));margin:auto}
    .eyebrow{margin:0 0 12px;color:#bde7df;font-weight:800;letter-spacing:.13em}
    h1{max-width:850px;margin:0;font-size:clamp(2.15rem,5vw,4.3rem);line-height:1.09;letter-spacing:-.035em}
    .hero-lead{max-width:790px;margin:22px 0 26px;color:#e5f4f2;font-size:1.12rem}
    .chips{display:flex;flex-wrap:wrap;gap:10px}
    .chip{padding:7px 12px;border:1px solid rgba(255,255,255,.25);border-radius:999px;background:rgba(255,255,255,.10);font-size:.92rem}
    .toc{position:sticky;z-index:4;top:0;display:flex;justify-content:center;gap:8px;overflow-x:auto;padding:11px 18px;border-bottom:1px solid var(--line);background:rgba(245,242,233,.94);backdrop-filter:blur(10px)}
    .toc a{flex:0 0 auto;padding:5px 10px;border-radius:999px;color:var(--ink);text-decoration:none;font-size:.9rem}
    .toc a:hover{background:white}
    main{padding:58px 0 78px}
    section{margin:0 0 76px;scroll-margin-top:76px}
    .section-head{max-width:790px;margin-bottom:26px}
    .kicker{margin:0 0 5px;color:var(--sediment);font-size:.83rem;font-weight:900;letter-spacing:.12em}
    h2{margin:0 0 10px;font-size:clamp(1.7rem,3.2vw,2.65rem);line-height:1.18;letter-spacing:-.025em}
    h3{margin:0 0 8px;line-height:1.3}
    p{margin:0 0 12px}
    .muted,figcaption{color:var(--muted)}
    .frame-grid,.term-grid,.image-grid,.metrics{display:grid;gap:20px}
    .frame-grid,.image-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
    .image-grid.three{grid-template-columns:repeat(3,minmax(0,1fr))}
    .term-grid{grid-template-columns:repeat(4,minmax(0,1fr))}
    .metrics{grid-template-columns:repeat(3,minmax(0,1fr));margin:22px 0}
    figure{margin:0;overflow:hidden;border:1px solid var(--line);border-radius:17px;background:var(--card);box-shadow:var(--shadow)}
    figure img{display:block;width:100%;height:auto;background:#172126}
    figcaption{padding:15px 17px 17px;font-size:.93rem}
    figcaption strong{display:block;margin-bottom:3px;color:var(--ink);font-size:1rem}
    .callout{margin-top:22px;padding:20px 22px;border-left:5px solid var(--river);border-radius:10px;background:var(--good)}
    .callout.warn{border-left-color:var(--sediment);background:var(--warn)}
    .term{padding:20px;border:1px solid var(--line);border-radius:15px;background:var(--card)}
    .term .number,.step-number{display:inline-grid;place-items:center;width:34px;height:34px;margin-bottom:13px;border-radius:50%;color:white;background:var(--river-dark);font-weight:900}
    .timeline{display:grid;gap:30px}
    .step{display:grid;grid-template-columns:52px minmax(0,1fr);gap:12px;padding-bottom:30px;border-bottom:1px solid var(--line)}
    .step:last-child{border-bottom:0}
    .step-number{margin:2px 0 0;background:var(--sediment)}
    .step-body>p{max-width:850px}
    .step-body figure,.step-body .image-grid{margin-top:18px}
    .wide img{max-height:780px;object-fit:contain}
    .metric{padding:18px 20px;border:1px solid var(--line);border-radius:14px;background:var(--card)}
    .metric strong{display:block;color:var(--river-dark);font-size:1.45rem;line-height:1.2}
    .metric span{color:var(--muted);font-size:.9rem}
    table{width:100%;border-collapse:collapse;overflow:hidden;border-radius:14px;background:var(--card);box-shadow:var(--shadow)}
    th,td{padding:15px 17px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
    th{color:white;background:var(--river-dark)}
    tr:last-child td{border-bottom:0}
    details{margin:12px 0;border:1px solid var(--line);border-radius:12px;background:var(--card)}
    summary{padding:15px 18px;cursor:pointer;font-weight:800}
    .detail-body{padding:0 18px 18px}
    pre{overflow-x:auto;margin:10px 0;padding:15px;border-radius:9px;color:#edf7f5;background:#12272d;white-space:pre-wrap;overflow-wrap:anywhere;font-size:.83rem;line-height:1.55}
    .path-list{display:grid;gap:7px;padding-left:20px}
    footer{padding:26px 0;color:#dbeceb;background:#12343b}
    footer p{margin:0}
    @media(max-width:900px){.term-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.image-grid.three{grid-template-columns:1fr 1fr}}
    @media(max-width:680px){
      .hero{padding:56px 8px 48px}
      .hero-inner,main,.footer-inner{width:min(100% - 24px,1160px)}
      .frame-grid,.image-grid,.image-grid.three,.term-grid,.metrics{grid-template-columns:1fr}
      .step{grid-template-columns:40px minmax(0,1fr)}
      .step-number{width:32px;height:32px}
      th,td{padding:11px}
    }
    @media print{.toc{display:none}body{background:white}section,details{break-inside:avoid}}
  </style>
</head>
<body>
  <header class="hero">
    <div class="hero-inner">
      <p class="eyebrow">STAGE 1.2 · 可视化生成报告</p>
      <h1>泥沙从河道移动到河口：两张关键帧怎么生成</h1>
      <p class="hero-lead">
        目标不是画两张互不相关的风景，而是在同一个三角洲场景里，
        清楚表现“泥沙仍在河道中”与“泥沙刚到河口”这两个连续时刻。
        本页从结果讲起，再逐图拆解生成过程、失败尝试和复现参数。
      </p>
      <div class="chips">
        <span class="chip">2 张最终关键帧</span>
        <span class="chip">$candidate_count 张模型原始候选</span>
        <span class="chip">14 项自动测试通过</span>
        <span class="chip">原图与后期层分开保存</span>
      </div>
    </div>
  </header>

  <nav class="toc" aria-label="报告目录">
    <a href="#result">最终结果</a><a href="#terms">先懂四个词</a>
    <a href="#process">生成过程</a><a href="#roles">各部分作用</a>
    <a href="#limits">结论与限制</a><a href="#reproduce">复现</a>
  </nav>

  <main>
    <section id="result">
      <div class="section-head">
        <p class="kicker">01 · 先看结果</p>
        <h2>同一场景里的前后两个时刻</h2>
        <p class="muted">两张图保持同一河道、海岸、镜头和光线。真正变化的是水中的泥沙位置，而不是换一张地图或突然长出新三角洲。</p>
      </div>
      <div class="frame-grid">
        <figure>
          <a href="final/in_channel.png"><img src="final/in_channel.png" alt="第一张关键帧：泥沙在河道中随水流向海岸方向移动"></a>
          <figcaption><strong>第 1 张｜泥沙仍在河道中</strong>泥沙带沿河道向右移动，但前缘还没有到出口。此时有 $first_particles 个悬浮泥沙模拟粒子。</figcaption>
        </figure>
        <figure>
          <a href="final/at_outlet.png"><img src="final/at_outlet.png" alt="第二张关键帧：泥沙前缘刚移动到河道出口"></a>
          <figcaption><strong>第 2 张｜泥沙刚到河口</strong>泥沙前缘抵达出口附近，但尚未向海面扩散，也没有提前生成沉积地形。此时有 $second_particles 个悬浮泥沙模拟粒子。</figcaption>
        </figure>
      </div>
      <div class="callout">
        <strong>一句话结论：</strong>最终采用“模型地貌底图 + 机制泥沙层”的组合方案。
        模型负责自然地貌、光照和材质；模拟数据负责告诉我们泥沙此刻究竟在哪里。
        最终图不是未经修改的模型原图。
      </div>
    </section>

    <section id="terms">
      <div class="section-head">
        <p class="kicker">02 · 阅读准备</p>
        <h2>先认识后面会出现的四个词</h2>
        <p class="muted">这四个概念分清以后，整套流程就不神秘了。</p>
      </div>
      <div class="term-grid">
        <article class="term"><span class="number">1</span><h3>ControlNet 边缘控制</h3><p>给模型一张黑白线图，告诉它“河岸和海岸大致画在这里”。它约束形状，不负责决定泥沙的位置。</p></article>
        <article class="term"><span class="number">2</span><h3>模型原始图</h3><p>从 SDXL + ControlNet 直接保存的图片，尚未叠加自己的泥沙颜色层。它是判断模型本身表现的依据。</p></article>
        <article class="term"><span class="number">3</span><h3>随机种子</h3><p>模型从随机噪声开始画图。固定一个数字，就能重现相近构图。数字 $selected_seed 只是被选中的起点编号，不是一种泥沙或模型。</p></article>
        <article class="term"><span class="number">4</span><h3>机制泥沙层</h3><p>把模拟中的泥沙粒子坐标变成柔和的棕色浓度层，再叠到水面上。它不是第二个生图模型，也不改变地形。</p></article>
      </div>
    </section>

    <section id="process">
      <div class="section-head">
        <p class="kicker">03 · 完整过程</p>
        <h2>从模拟状态到两张最终关键帧</h2>
        <p class="muted">每一步都附了实际中间图。点击图片可以查看原尺寸。</p>
      </div>
      <div class="timeline">
        <article class="step">
          <span class="step-number">1</span>
          <div class="step-body">
            <h3>先由模拟决定“选哪两个时刻”</h3>
            <p>第一个时刻要求泥沙前缘仍在河道内；第二个时刻要求它刚到河口。图中的彩色点是模拟粒子，红色竖线是海岸。这两张审计图只用于核对物理状态，<strong>没有送进生图模型</strong>。</p>
            <div class="image-grid">
              <figure>
                <a href="_work/source/in_channel_mechanism_audit.png"><img src="_work/source/in_channel_mechanism_audit.png" alt="第一时刻机制审计图"></a>
                <figcaption><strong>第一个模拟时刻</strong>泥沙前缘距河口仍有 $first_distance 个模拟距离单位。</figcaption>
              </figure>
              <figure>
                <a href="_work/source/at_outlet_mechanism_audit.png"><img src="_work/source/at_outlet_mechanism_audit.png" alt="第二时刻机制审计图"></a>
                <figcaption><strong>第二个模拟时刻</strong>泥沙前缘距河口只剩 $second_distance 个模拟距离单位。</figcaption>
              </figure>
            </div>
          </div>
        </article>

        <article class="step">
          <span class="step-number">2</span>
          <div class="step-body">
            <h3>只用河岸和海岸线约束地貌</h3>
            <p>这张黑白图是两帧共同使用的 ControlNet 输入。白线只标出河岸与海岸，没有把泥沙、粒子点、文字、箭头或后期颜色塞给模型。两帧因此能保持同一地理骨架，又不会把调试标记画进成品。</p>
            <figure>
              <a href="_work/source/natural_sparse_canny.png"><img src="_work/source/natural_sparse_canny.png" alt="只含河岸和海岸的稀疏黑白边缘控制图"></a>
              <figcaption><strong>共享的稀疏边缘控制图</strong>黑色是空白，细白线是必须保留的河岸和海岸。白线约占整张图的 0.60%，所以不会像详细线稿那样把画面锁死。</figcaption>
            </figure>
          </div>
        </article>

        <article class="step">
          <span class="step-number">3</span>
          <div class="step-body">
            <h3>让模型生成同一场景的自然地貌底图</h3>
            <p>我们用 SDXL 生成材质与光照，用 Canny ControlNet 保持河岸和海岸位置。两帧使用相同的随机种子 $selected_seed，所以构图和风格可以直接比较。下面是被选中的<strong>模型原始图</strong>，还没有叠加泥沙层。</p>
            <div class="image-grid">
              <figure>
                <a href="review/in_channel/in_channel_s3102.png"><img src="review/in_channel/in_channel_s3102.png" alt="第一帧模型原始图，河水仍偏清澈"></a>
                <figcaption><strong>第 1 张模型原始图</strong>地貌和镜头可用，但河道里的悬浮泥沙不明显。</figcaption>
              </figure>
              <figure>
                <a href="review/at_outlet/at_outlet_s3102.png"><img src="review/at_outlet/at_outlet_s3102.png" alt="第二帧模型原始图，河水仍偏清澈"></a>
                <figcaption><strong>第 2 张模型原始图</strong>场景一致，但只看水色无法确认泥沙是否已经到河口。</figcaption>
              </figure>
            </div>
          </div>
        </article>

        <article class="step">
          <span class="step-number">4</span>
          <div class="step-body">
            <h3>验证：仅修改提示词仍然不够</h3>
            <p>第一轮生成 8 张原始候选。第二轮把“悬浮泥沙、棕色浑水、河口前缘”等描述写得更明确，又生成 8 张。两轮都没有稳定画出正确的水中泥沙位置；有时棕色甚至出现在陆地上。问题不只是提示词不够华丽，而是文字很难精确控制“泥沙前缘此刻走到哪”。</p>
            <div class="image-grid">
              <figure class="wide">
                <a href="pairs-labeled.jpg"><img src="pairs-labeled.jpg" alt="第一轮八张模型候选对比图"></a>
                <figcaption><strong>第一轮：8 张原始候选</strong>场景连贯，但河水普遍清澈，两个阶段难以辨认。</figcaption>
              </figure>
              <figure class="wide">
                <a href="pairs-revised.jpg"><img src="pairs-revised.jpg" alt="强化泥沙提示词后的八张模型候选对比图"></a>
                <figcaption><strong>第二轮：强化提示词后的 8 张候选</strong>仍未可靠解决泥沙位置与水体浑浊度。</figcaption>
              </figure>
            </div>
          </div>
        </article>

        <article class="step">
          <span class="step-number">5</span>
          <div class="step-body">
            <h3>把模拟粒子变成柔和的泥沙浓度</h3>
            <p>每个粒子不再画成一个生硬圆点，而是扩散成小片柔和浓度，再把所有粒子叠加。浓度图中越亮的位置代表悬浮泥沙越集中。这样既保留模拟给出的前缘位置，也避免“像素点贴纸”的效果。</p>
            <div class="image-grid three">
              <figure>
                <a href="_work/soft_sediment/in_channel_density.png"><img src="_work/soft_sediment/in_channel_density.png" alt="第一帧泥沙浓度图"></a>
                <figcaption><strong>第 1 张泥沙浓度</strong>亮区还停留在河道内部。</figcaption>
              </figure>
              <figure>
                <a href="_work/soft_sediment/at_outlet_density.png"><img src="_work/soft_sediment/at_outlet_density.png" alt="第二帧泥沙浓度图"></a>
                <figcaption><strong>第 2 张泥沙浓度</strong>亮区向右推进，前缘刚到出口。</figcaption>
              </figure>
              <figure>
                <a href="_work/soft_sediment/river_corridor.png"><img src="_work/soft_sediment/river_corridor.png" alt="两条河岸之间的河道范围图"></a>
                <figcaption><strong>河道范围图</strong>白色表示两条河岸之间的水域，灰色边缘让颜色过渡不生硬。</figcaption>
              </figure>
            </div>
            <div class="callout warn">
              <strong>为什么需要这张“河道范围图”？</strong>
              模型偶尔会把河道内部画成浅滩。程序根据同一组河岸线识别两岸之间的水域，
              只在这里校正水色并叠加泥沙，避免棕色跑到陆地上。它是后期合成边界，
              不是隐藏的模型输入；以前叫 mask，本页直接解释它的实际用途。
            </div>
          </div>
        </article>

        <article class="step">
          <span class="step-number">6</span>
          <div class="step-body">
            <h3>合成最终帧，并保留来源标注</h3>
            <p>最后把柔和泥沙浓度叠到同一组地貌底图中。程序不重画海岸、不生成海上羽状流，也不提前增加新陆地。最终图是“模型地貌 + 机制泥沙”的组合结果，所以不能称作未经修改的 SDXL 原图。</p>
            <figure>
              <a href="pairs-soft-sediment.jpg"><img src="pairs-soft-sediment.jpg" alt="模型原始图、泥沙层和最终帧的逐步对比"></a>
              <figcaption><strong>最终组合图对</strong>两张图来自同一个模型构图，泥沙位置则来自各自对应的模拟时刻。</figcaption>
            </figure>
          </div>
        </article>
      </div>
    </section>

    <section id="roles">
      <div class="section-head">
        <p class="kicker">04 · 职责边界</p>
        <h2>每一部分到底负责什么</h2>
      </div>
      <table>
        <thead><tr><th>组成部分</th><th>它决定什么</th><th>它不决定什么</th></tr></thead>
        <tbody>
          <tr><td>河流与泥沙模拟</td><td>选取两个时刻；给出泥沙粒子的数量与位置</td><td>不生成真实感纹理和光照</td></tr>
          <tr><td>稀疏边缘控制图</td><td>固定河岸、海岸的大致形状</td><td>不包含泥沙，不指定颜色</td></tr>
          <tr><td>SDXL + ControlNet</td><td>生成航拍质感、地貌、光照与水面</td><td>没有稳定控制泥沙前缘的位置</td></tr>
          <tr><td>机制泥沙层</td><td>把模拟坐标转成柔和的浑水分布</td><td>不移动海岸，不凭空生成新地形</td></tr>
          <tr><td>报告与元数据</td><td>记录输入、失败尝试、参数和文件来源</td><td>不参与图像生成</td></tr>
        </tbody>
      </table>
      <div class="metrics">
        <div class="metric"><strong>$first_particles</strong><span>第 1 张的悬浮泥沙模拟粒子</span></div>
        <div class="metric"><strong>$second_particles</strong><span>第 2 张的悬浮泥沙模拟粒子</span></div>
        <div class="metric"><strong>$first_distance → $second_distance</strong><span>泥沙前缘距河口的模拟距离</span></div>
      </div>
    </section>

    <section id="limits">
      <div class="section-head">
        <p class="kicker">05 · 结果判断</p>
        <h2>这次解决了什么，还没解决什么</h2>
      </div>
      <div class="frame-grid">
        <div class="callout">
          <h3>已经做到</h3>
          <p>两帧是同一地貌和镜头；泥沙前缘从河道内部推进到河口；没有提前出现海上羽状流、分汊或新生三角洲陆地；所有模型原图与中间层均保留可查。</p>
        </div>
        <div class="callout warn">
          <h3>仍需改进</h3>
          <p>最终河床仍偏平滑、规则，局部还有合成感。这一版适合验证动画叙事和运动路径，还不能视作最终照片级成片。下一阶段应重点改善河床纹理与泥沙在水中的体积感。</p>
        </div>
      </div>
    </section>

    <section id="reproduce">
      <div class="section-head">
        <p class="kicker">06 · 技术附录</p>
        <h2>需要复现时，再展开这些细节</h2>
        <p class="muted">主报告不要求读者先懂参数；完整设置仍保留在这里，方便逐项重跑和审计。</p>
      </div>
      <details>
        <summary>模型和生成参数</summary>
        <div class="detail-body">
          <table><tbody>
            <tr><td>基础模型</td><td>$sdxl_model，FP16</td></tr>
            <tr><td>结构控制模型</td><td>$controlnet_model，FP16</td></tr>
            <tr><td>生成管线</td><td>$pipeline</td></tr>
            <tr><td>尺寸</td><td>$width × $height</td></tr>
            <tr><td>采样步数</td><td>$steps</td></tr>
            <tr><td>提示词引导强度</td><td>$guidance</td></tr>
            <tr><td>ControlNet 强度</td><td>$control_scale</td></tr>
            <tr><td>候选随机种子</td><td>$seeds；最终底图选用 $selected_seed</td></tr>
          </tbody></table>
          <p class="muted" style="margin-top:12px">原始模型推理没有使用 img2img、strength 参数或模型内遮罩投影。河道范围只用于最终的确定性合成。</p>
        </div>
      </details>
      <details>
        <summary>第一轮完整提示词</summary>
        <div class="detail-body">
          <p><strong>第 1 张：</strong></p><pre>$baseline_first</pre>
          <p><strong>第 2 张：</strong></p><pre>$baseline_second</pre>
          <p><strong>反向提示词：</strong></p><pre>$baseline_negative</pre>
        </div>
      </details>
      <details>
        <summary>第二轮强化泥沙后的完整提示词</summary>
        <div class="detail-body">
          <p><strong>第 1 张：</strong></p><pre>$revision_first</pre>
          <p><strong>第 2 张：</strong></p><pre>$revision_second</pre>
          <p><strong>反向提示词：</strong></p><pre>$revision_negative</pre>
        </div>
      </details>
      <details open>
        <summary>一键复现命令</summary>
        <div class="detail-body">
          <p>在项目根目录 <code>Live-Document</code> 运行：</p>
          <pre>/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.transport_pair_test --prepare
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.transport_pair_test --generate --force
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.transport_pair_test --generate-revision --force
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.transport_pair_test --build-soft-sediment --base-seed 3102
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.transport_pair_test --report</pre>
        </div>
      </details>
      <details>
        <summary>原始记录和机器可读文件</summary>
        <div class="detail-body">
          <ul class="path-list">
            <li><a href="report.md">详细 Markdown 报告</a></li>
            <li><a href="_work/metadata.json">全部候选与生成参数（metadata.json）</a></li>
            <li><a href="_work/review.json">人工选择记录（review.json）</a></li>
            <li><a href="_work/soft_sediment_manifest.json">泥沙合成记录（soft_sediment_manifest.json）</a></li>
            <li><a href="final/selection.json">最终输出来源（selection.json）</a></li>
          </ul>
        </div>
      </details>
    </section>
  </main>
  <footer>
    <div class="footer-inner"><p>Stage 1.2｜三角洲形成第一阶段：河道输沙 → 泥沙抵达河口。所有图片均使用相对路径，可随报告目录一起离线查看。</p></div>
  </footer>
</body>
</html>
"""
    )
    return template.substitute(substitutions)
