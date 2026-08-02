import { useState } from 'react'
import { api } from '../api'
import type { EngineName, Spec, SpecsResponse } from '../types'
import SpecCard from '../components/SpecCard'

interface Props {
  onSubmitted: (jobId: string) => void
}

const ENGINES: { id: EngineName; label: string; hint: string }[] = [
  {
    id: 'deterministic',
    label: 'Deterministic（Manim 教学动画）',
    hint: '规则化确定性渲染，最适合比赛演示；渲染较慢',
  },
  {
    id: 'generative',
    label: 'Generative（LTX / Wan 视频生成）',
    hint: '生成式视频模型，需 ROCm/GPU；不可用时自动回退到程序化渲染',
  },
  {
    id: 'procedural',
    label: 'Procedural（PIL 程序化 GIF）',
    hint: '轻量兜底渲染，无外部模型依赖，速度最快',
  },
]

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

  return (
    <section className="page">
      <div className="panel">
        <h2>① 输入文档</h2>
        <textarea
          className="doc-input"
          rows={8}
          placeholder="粘贴需要动态化展示的教学文档/章节文本，例如：梯度下降沿负梯度方向更新参数，直到收敛到最小值。…"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <div className="row">
          <input
            className="text-input"
            placeholder="可选：文档文件名（如 gradient-descent.md）"
            value={filename}
            onChange={(e) => setFilename(e.target.value)}
          />
          <button className="btn btn-primary" onClick={analyze} disabled={loading}>
            {loading ? '分析中…' : '② 分析并生成 LearningSpec'}
          </button>
        </div>
        {error && <p className="error-text">{error}</p>}
      </div>

      {resp && (
        <div className="panel">
          <h2>
            ③ 选择片段{' '}
            <span className="muted">
              （共 {resp.count} 段，适合动画 {resp.suitable} 段）
            </span>
          </h2>
          {resp.suitable === 0 && (
            <p className="error-text">
              文档中没有适合动态化的段落，建议更换文本，或改用 Procedural 引擎尝试。
            </p>
          )}
          <div className="spec-list">
            {resp.specs.map((s, i) => (
              <div key={i} className={selected === i ? 'spec-item selected' : 'spec-item'}>
                <label className="spec-pick">
                  <input
                    type="radio"
                    name="spec"
                    checked={selected === i}
                    onChange={() => selectSpec(resp, i)}
                    disabled={s.fallback_reason != null}
                  />
                  <span>选择此片段</span>
                </label>
                <SpecCard spec={s} index={i} />
              </div>
            ))}
          </div>
        </div>
      )}

      {editedSpec && (
        <div className="panel">
          <h2>④ 微调片段并选择渲染引擎</h2>
          <div className="edit-grid">
            <label className="field">
              <span className="field-label">学习目标（可编辑）</span>
              <textarea
                className="doc-input compact"
                rows={3}
                value={goalEdit}
                onChange={(e) => setGoalEdit(e.target.value)}
              />
            </label>
            <label className="field">
              <span className="field-label">状态变量（逗号分隔，可编辑）</span>
              <input
                className="text-input"
                value={varsEdit}
                onChange={(e) => setVarsEdit(e.target.value)}
              />
            </label>
          </div>
          <div className="engine-list">
            {ENGINES.map((en) => (
              <label key={en.id} className={`engine-option ${engine === en.id ? 'selected' : ''}`}>
                <input
                  type="radio"
                  name="engine"
                  checked={engine === en.id}
                  onChange={() => setEngine(en.id)}
                />
                <div>
                  <strong>{en.label}</strong>
                  <p className="muted">{en.hint}</p>
                </div>
              </label>
            ))}
          </div>
          <button className="btn btn-primary btn-lg" onClick={submit} disabled={submitting}>
            {submitting ? '提交中…' : '⑤ 提交动画生成任务'}
          </button>
          {submitError && <p className="error-text">{submitError}</p>}
        </div>
      )}
    </section>
  )
}
