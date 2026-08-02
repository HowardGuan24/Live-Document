import type { Spec } from '../types'

export default function SpecCard({ spec, index }: { spec: Spec; index: number }) {
  const unsuitable = spec.fallback_reason != null
  return (
    <article className={`spec-card ${unsuitable ? 'spec-unsuitable' : ''}`}>
      <header className="spec-head">
        <span className="spec-index">片段 #{index + 1}</span>
        {unsuitable ? (
          <span className="badge badge-warn" title={spec.fallback_reason ?? ''}>
            不适合动态化
          </span>
        ) : (
          <span className="badge badge-ok">适合动画</span>
        )}
      </header>
      {spec.learning_goal && <p className="spec-goal">{spec.learning_goal}</p>}
      <div className="spec-section">
        <h4>实体 Entities</h4>
        {spec.entities.length ? (
          <div className="chips">
            {spec.entities.map((e) => (
              <span className="chip" key={e}>{e}</span>
            ))}
          </div>
        ) : (
          <p className="muted">—</p>
        )}
      </div>
      <div className="spec-section">
        <h4>状态变量 State Variables</h4>
        {spec.state_variables.length ? (
          <div className="chips">
            {spec.state_variables.map((v) => (
              <span className="chip chip-alt" key={v}>{v}</span>
            ))}
          </div>
        ) : (
          <p className="muted">—</p>
        )}
      </div>
      {spec.causal_steps.length > 0 && (
        <div className="spec-section">
          <h4>因果链 Causal Steps</h4>
          <ol className="causal-list">
            {spec.causal_steps.map((s, i) => (
              <li key={i}>
                <span className="causal-cause">{s.cause ?? ''}</span>
                <span className="causal-arrow">→</span>
                <span className="causal-change">{s.change ?? ''}</span>
                {s.visual_evidence && (
                  <span className="causal-evidence">（{s.visual_evidence}）</span>
                )}
              </li>
            ))}
          </ol>
        </div>
      )}
      {spec.invariants.length > 0 && (
        <div className="spec-section">
          <h4>不变式 Invariants</h4>
          <ul className="plain-list">
            {spec.invariants.map((v) => (
              <li key={v}>{v}</li>
            ))}
          </ul>
        </div>
      )}
      {spec.comprehension_questions.length > 0 && (
        <div className="spec-section">
          <h4>理解题 Comprehension</h4>
          <ul className="plain-list">
            {spec.comprehension_questions.map((q) => (
              <li key={q}>• {q}</li>
            ))}
          </ul>
        </div>
      )}
      {unsuitable && <p className="fallback-reason">{spec.fallback_reason}</p>}
    </article>
  )
}
