import type { Spec } from '../types'

export default function SpecCard({
  spec,
  index,
  selected = false,
}: {
  spec: Spec
  index: number
  selected?: boolean
}) {
  const unsuitable = spec.fallback_reason != null
  return (
    <article className="text-sm" style={{ pointerEvents: 'none' }}>
      <header className="flex items-center gap-2">
        <span className="font-mono text-xs font-bold text-radeon-400">#{String(index + 1).padStart(2, '0')}</span>
        {selected && !unsuitable && (
          <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-radeon-400">
            ◆ 已选
          </span>
        )}
        {unsuitable ? (
          <span
            className="clip-angled-sm border border-warn/50 bg-warn/10 px-2 py-0.5 text-[11px] text-warn"
            title={spec.fallback_reason ?? ''}
          >
            不适合动态化
          </span>
        ) : (
          <span className="clip-angled-sm border border-ok/40 bg-ok/10 px-2 py-0.5 text-[11px] text-ok">
            适合动画
          </span>
        )}
      </header>

      {spec.learning_goal && (
        <p className="mt-2 text-[15px] font-semibold leading-snug text-fg">{spec.learning_goal}</p>
      )}

      <div className="mt-3">
        <h4 className="font-mono text-[10px] uppercase tracking-[0.16em] text-dim">实体 Entities</h4>
        {spec.entities.length ? (
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {spec.entities.map((e) => (
              <span
                key={e}
                className="clip-angled-sm border border-radeon-500/35 bg-radeon-500/10 px-2 py-0.5 text-[11px] text-radeon-400"
              >
                {e}
              </span>
            ))}
          </div>
        ) : (
          <p className="mt-1 text-dim">—</p>
        )}
      </div>

      <div className="mt-3">
        <h4 className="font-mono text-[10px] uppercase tracking-[0.16em] text-dim">状态变量 State Variables</h4>
        {spec.state_variables.length ? (
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {spec.state_variables.map((v) => (
              <span
                key={v}
                className="clip-angled-sm border border-ember-500/35 bg-ember-500/10 px-2 py-0.5 text-[11px] text-ember-400"
              >
                {v}
              </span>
            ))}
          </div>
        ) : (
          <p className="mt-1 text-dim">—</p>
        )}
      </div>

      {spec.causal_steps.length > 0 && (
        <div className="mt-3">
          <h4 className="font-mono text-[10px] uppercase tracking-[0.16em] text-dim">因果链 Causal Steps</h4>
          <ol className="mt-1.5 list-none space-y-1 pl-0">
            {spec.causal_steps.map((s, i) => (
              <li key={i} className="flex flex-wrap items-baseline gap-x-1.5 text-[13px]">
                <span className="font-mono text-[10px] text-dim">{String(i + 1).padStart(2, '0')}</span>
                <span className="text-fg">{s.cause ?? ''}</span>
                <span className="font-bold text-radeon-400">→</span>
                <span className="text-ember-400">{s.change ?? ''}</span>
                {s.visual_evidence && <span className="text-dim">（{s.visual_evidence}）</span>}
              </li>
            ))}
          </ol>
        </div>
      )}

      {spec.invariants.length > 0 && (
        <div className="mt-3">
          <h4 className="font-mono text-[10px] uppercase tracking-[0.16em] text-dim">不变式 Invariants</h4>
          <ul className="mt-1.5 list-disc space-y-0.5 pl-4 text-[13px] text-fg">
            {spec.invariants.map((v) => (
              <li key={v}>{v}</li>
            ))}
          </ul>
        </div>
      )}

      {spec.comprehension_questions.length > 0 && (
        <div className="mt-3">
          <h4 className="font-mono text-[10px] uppercase tracking-[0.16em] text-dim">理解题 Comprehension</h4>
          <ul className="mt-1.5 list-disc space-y-0.5 pl-4 text-[13px] text-fg">
            {spec.comprehension_questions.map((q) => (
              <li key={q}>{q}</li>
            ))}
          </ul>
        </div>
      )}

      {unsuitable && <p className="mt-2 text-xs text-warn">{spec.fallback_reason}</p>}
    </article>
  )
}