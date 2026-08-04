import { useState } from 'react'
import { api } from '../api'
import type { EngineName, Spec, SpecsResponse } from '../types'
import SpecCard from '../components/SpecCard'

interface Props {
  onSubmitted: (jobId: string) => void
}

const ENGINES: { id: EngineName; label: string; hint: string; icon: string; tag: string }[] = [
  {
    id: 'deterministic',
    label: 'Deterministic',
    tag: 'Manim 教学动画',
    hint: '规则化确定性渲染，最适合比赛演示；渲染较慢',
    icon: '◈',
  },
  {
    id: 'generative',
    label: 'Generative',
    tag: 'LTX / Wan 视频生成',
    hint: '生成式视频模型，需 ROCm/GPU；不可用时自动回退程序化渲染',
    icon: '✦',
  },
  {
    id: 'procedural',
    label: 'Procedural',
    tag: 'PIL 程序化 GIF',
    hint: '轻量兜底渲染，无外部模型依赖，速度最快',
    icon: '⚡',
  },
]

const STEPS = ['输入文档', '选择片段', '微调与生成']

function StepTitle({ n, title, hint }: { n: string; title: string; hint?: string }) {
  return (
    <div className="mb-4 flex items-center gap-3">
      <span className="font-mono text-sm font-bold text-radeon-400">{n}</span>
      <h2 className="whitespace-nowrap text-base font-bold text-fg">{title}</h2>
      {hint && <span className="whitespace-nowrap text-xs text-dim">{hint}</span>}
      <span className="energy-line h-px flex-1 opacity-30" />
    </div>
  )
}

function Stepper({ current }: { current: number }) {
  return (
    <div className="clip-angled mb-6 flex items-center gap-2 border border-line bg-ink-900 px-4 py-3 sm:gap-3 sm:px-6">
      {STEPS.map((label, i) => {
        const n = i + 1
        const done = n < current
        const active = n === current
        return (
          <div key={label} className="flex min-w-0 flex-1 items-center gap-2 sm:gap-3">
            <span
              className={`clip-angled-sm flex h-7 w-7 shrink-0 items-center justify-center font-mono text-xs font-bold transition-all ${
                active
                  ? 'bg-gradient-to-br from-radeon-500 to-ember-600 text-white glow-red-sm'
                  : done
                    ? 'border border-radeon-500/50 text-radeon-400'
                    : 'border border-line text-dim'
              }`}
            >
              {done ? '✓' : n}
            </span>
            <span
              className={`truncate text-xs sm:text-sm ${
                active ? 'font-semibold text-fg' : done ? 'text-mut' : 'text-dim'
              }`}
            >
              {label}
            </span>
            {n < STEPS.length && (
              <span
                className={`h-px min-w-3 flex-1 ${done || active ? 'energy-line opacity-60' : 'bg-line'}`}
              />
            )}
          </div>
        )
      })}
    </div>
  )
}

export default function PlannerPage({ onSubmitted }: Props) {
  const [text, setText] = useState('')
  const [filename, setFilename] = useState('')
  const [loading, setLoading] = useState(false)
  const [resp, setResp] = useState<SpecsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [engine, setEngine] = useState<EngineName>('deterministic')
  const [goalEdit, setGoalEdit] = useState('')
  const [varsEdit, setVarsEdit] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  function selectSpec(r: SpecsResponse, idx: number) {
    setSelected(idx)
    setGoalEdit(r.specs[idx].learning_goal ?? '')
    setVarsEdit(r.specs[idx].state_variables.join(', '))
    setSubmitError(null)
  }

  async function analyze() {
    setError(null)
    setSubmitError(null)
    setResp(null)
    setSelected(null)
    if (!text.trim()) {
      setError('请先输入需要处理的文档文本')
      return
    }
    setLoading(true)
    try {
      const r = await api.plan(text.trim(), filename.trim() || undefined)
      setResp(r)
      if (r.suitable > 0) {
        const idx = r.specs.findIndex((s) => s.fallback_reason == null)
        if (idx >= 0) selectSpec(r, idx)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  async function submit() {
    if (resp == null || selected == null) return
    const base = resp.specs[selected]
    const spec: Spec = {
      ...base,
      learning_goal: goalEdit.trim() || null,
      state_variables: varsEdit
        .split(/[,，]/)
        .map((s) => s.trim())
        .filter(Boolean),
    }
    setSubmitting(true)
    setSubmitError(null)
    try {
      const job = await api.createJob({ spec, engine })
      onSubmitted(job.id)
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e))
    } finally {
      setSubmitting(false)
    }
  }

  const editedSpec: Spec | null =
    resp != null && selected != null
      ? {
          ...resp.specs[selected],
          learning_goal: goalEdit.trim() || null,
          state_variables: varsEdit
            .split(/[,，]/)
            .map((s) => s.trim())
            .filter(Boolean),
        }
      : null

  const currentStep = editedSpec != null ? 3 : resp != null ? 2 : 1

  return (
    <section className="animate-fade-up">
      <Stepper current={currentStep} />

      {/* ① 输入文档 */}
      <div className="clip-angled border border-line bg-ink-900 p-6">
        <StepTitle n="01" title="输入文档" hint="粘贴教学文档 / 章节文本" />
        <textarea
          rows={8}
          placeholder="粘贴需要动态化展示的教学文档/章节文本，例如：梯度下降沿负梯度方向更新参数，直到收敛到最小值。…"
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="clip-angled-sm w-full resize-y border border-line bg-ink-950 px-4 py-3 text-sm text-fg outline-none transition-all placeholder:text-dim focus:border-radeon-500/70 focus:glow-red-sm"
        />
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <input
            placeholder="可选：文档文件名（如 gradient-descent.md）"
            value={filename}
            onChange={(e) => setFilename(e.target.value)}
            className="clip-angled-sm min-w-56 flex-1 border border-line bg-ink-950 px-4 py-2.5 text-sm text-fg outline-none transition-all placeholder:text-dim focus:border-radeon-500/70"
          />
          <button
            onClick={analyze}
            disabled={loading}
            className="clip-angled-sm bg-gradient-to-r from-radeon-600 via-radeon-500 to-ember-600 px-6 py-2.5 text-sm font-bold text-white transition-all hover:glow-red disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? '分析中…' : '分析并生成 LearningSpec →'}
          </button>
        </div>
        {error && <p className="mt-3 text-sm text-danger">{error}</p>}
      </div>

      {/* ② 选择片段 */}
      {resp && (
        <div className="clip-angled animate-fade-up mt-5 border border-line bg-ink-900 p-6">
          <StepTitle
            n="02"
            title="选择片段"
            hint={`共 ${resp.count} 段 · 适合动画 ${resp.suitable} 段 · 点击卡片选择`}
          />
          {resp.suitable === 0 && (
            <p className="mb-3 text-sm text-warn">
              文档中没有适合动态化的段落，建议更换文本，或改用 Procedural 引擎尝试。
            </p>
          )}
          <div className="flex flex-col gap-3">
            {resp.specs.map((s, i) => {
              const disabled = s.fallback_reason != null
              const isSel = selected === i
              return (
                <div
                  key={i}
                  role="button"
                  aria-pressed={isSel}
                  aria-disabled={disabled}
                  tabIndex={disabled ? -1 : 0}
                  onClick={() => !disabled && selectSpec(resp, i)}
                  onKeyDown={(e) => {
                    if (!disabled && (e.key === 'Enter' || e.key === ' ')) {
                      e.preventDefault()
                      selectSpec(resp, i)
                    }
                  }}
                  className={`clip-angled cursor-pointer border p-4 transition-all ${
                    isSel
                      ? 'border-radeon-500/70 bg-ink-850 glow-red'
                      : disabled
                        ? 'cursor-not-allowed border-line bg-ink-950 opacity-60'
                        : 'border-line bg-ink-950 hover:border-line-strong hover:bg-ink-850'
                  }`}
                >
                  <SpecCard spec={s} index={i} selected={isSel} />
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* ③ 微调 + 引擎 + 提交 */}
      {editedSpec && (
        <div className="clip-angled animate-fade-up mt-5 border border-line bg-ink-900 p-6">
          <StepTitle n="03" title="微调片段并选择渲染引擎" />
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="flex flex-col gap-2">
              <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-dim">
                学习目标（可编辑）
              </span>
              <textarea
                rows={3}
                value={goalEdit}
                onChange={(e) => setGoalEdit(e.target.value)}
                className="clip-angled-sm resize-y border border-line bg-ink-950 px-4 py-2.5 text-sm text-fg outline-none transition-all focus:border-radeon-500/70"
              />
            </label>
            <label className="flex flex-col gap-2">
              <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-dim">
                状态变量（逗号分隔，可编辑）
              </span>
              <input
                value={varsEdit}
                onChange={(e) => setVarsEdit(e.target.value)}
                className="clip-angled-sm border border-line bg-ink-950 px-4 py-2.5 text-sm text-fg outline-none transition-all focus:border-radeon-500/70"
              />
            </label>
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-3">
            {ENGINES.map((en) => {
              const isSel = engine === en.id
              return (
                <button
                  key={en.id}
                  onClick={() => setEngine(en.id)}
                  aria-pressed={isSel}
                  className={`clip-angled relative border p-4 text-left transition-all ${
                    isSel
                      ? 'border-radeon-500/70 bg-ink-850 glow-red'
                      : 'border-line bg-ink-950 hover:border-line-strong hover:bg-ink-850'
                  }`}
                >
                  {isSel && <span className="energy-line absolute inset-y-0 left-0 w-1" />}
                  <div className="flex items-center gap-2">
                    <span className={`text-lg ${isSel ? 'text-radeon-400' : 'text-dim'}`}>
                      {en.icon}
                    </span>
                    <span className="text-sm font-bold text-fg">{en.label}</span>
                  </div>
                  <div className="mt-0.5 font-mono text-[10px] uppercase tracking-[0.14em] text-radeon-400/90">
                    {en.tag}
                  </div>
                  <p className="mt-1.5 text-xs leading-relaxed text-mut">{en.hint}</p>
                </button>
              )
            })}
          </div>

          <button
            onClick={submit}
            disabled={submitting}
            className="clip-angled mt-5 w-full bg-gradient-to-r from-radeon-600 via-radeon-500 to-ember-600 px-6 py-3.5 text-base font-bold tracking-wide text-white transition-all hover:glow-red disabled:cursor-not-allowed disabled:opacity-50 sm:w-auto"
          >
            {submitting ? '提交中…' : '⚡ 提交动画生成任务'}
          </button>
          {submitError && <p className="mt-3 text-sm text-danger">{submitError}</p>}
        </div>
      )}
    </section>
  )
}