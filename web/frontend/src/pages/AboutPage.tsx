import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import type { HealthResponse } from '../types'

export default function AboutPage({ active }: { active: boolean }) {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)

  const load = useCallback(async () => {
    setRefreshing(true)
    try {
      const h = await api.health()
      setHealth(h)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  // Keep status fresh while the tab is visible (probe is cached server-side).
  useEffect(() => {
    if (!active) return
    const timer = window.setInterval(() => void load(), 20000)
    return () => window.clearInterval(timer)
  }, [active, load])

  const gpuOk = health?.gpu.available

  return (
    <section className="animate-fade-up flex flex-col gap-5">
      <div className="clip-angled border border-line bg-ink-900 p-6">
        <div className="mb-4 flex items-center gap-3">
          <h2 className="flex items-center gap-3 text-base font-bold text-fg">
            <span className="clip-angled-sm flex h-7 w-7 items-center justify-center bg-gradient-to-br from-radeon-500 to-ember-600 text-sm text-white glow-red-sm">
              ◎
            </span>
            运行状态
          </h2>
          <span className="energy-line h-px flex-1 opacity-30" />
          <button
            onClick={() => void load()}
            disabled={refreshing}
            className="clip-angled-sm border border-line bg-ink-850 px-3 py-1.5 text-xs text-mut transition-all hover:border-radeon-500/60 hover:text-fg disabled:opacity-50"
          >
            {refreshing ? '刷新中…' : '↻ 刷新'}
          </button>
        </div>

        {error && <p className="text-sm text-danger">{error}</p>}
        {!health && !error && <p className="text-sm text-dim">正在读取健康信息…</p>}

        {health && (
          <>
            <div className="grid gap-3 sm:grid-cols-3">
              {/* 服务 */}
              <div className="clip-angled border border-line bg-ink-950 p-4">
                <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-dim">服务</div>
                <div className="mt-1 text-xl font-bold text-ok">{health.status}</div>
                <div className="mt-1 text-xs text-mut">API · FastAPI</div>
              </div>
              {/* Python */}
              <div className="clip-angled border border-line bg-ink-950 p-4">
                <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-dim">Python</div>
                <div className="mt-1 text-xl font-bold text-fg">{health.python}</div>
                <div className="mt-1 text-xs text-mut">运行时</div>
              </div>
              {/* GPU / ROCm — 突出 */}
              <div
                className={`clip-angled relative border p-4 ${
                  gpuOk
                    ? 'border-ok/40 bg-ok/5'
                    : 'border-warn/40 bg-warn/5'
                }`}
              >
                <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-dim">GPU / ROCm</div>
                <div className={`mt-1 flex items-center gap-2 text-xl font-bold ${gpuOk ? 'text-ok' : 'text-warn'}`}>
                  <span
                    className={`inline-block h-2.5 w-2.5 rounded-full ${
                      gpuOk ? 'bg-ok animate-pulse-glow' : 'bg-warn'
                    }`}
                  />
                  {gpuOk ? '可用' : '不可用'}
                </div>
                <div className="mt-1 text-xs text-mut">{health.gpu.detail}</div>
              </div>
            </div>

            {/* 引擎可用性 */}
            <h3 className="mt-6 mb-3 font-mono text-[10px] uppercase tracking-[0.18em] text-dim">
              渲染引擎
            </h3>
            <div className="flex flex-col gap-2">
              {Object.keys(health.engines).length === 0 ? (
                <p className="text-sm text-dim">暂无引擎信息。</p>
              ) : (
                Object.entries(health.engines).map(([name, info]) => (
                  <div
                    key={name}
                    className="clip-angled-sm flex items-center gap-3 border border-line bg-ink-950 px-4 py-2.5"
                  >
                    <code className="font-mono text-sm font-bold text-radeon-400">{name}</code>
                    <span
                      className={`clip-angled-sm border px-2 py-0.5 text-[11px] font-semibold ${
                        info.available
                          ? 'border-ok/50 bg-ok/10 text-ok'
                          : 'border-warn/50 bg-warn/10 text-warn'
                      }`}
                    >
                      {info.available ? 'available' : 'unavailable'}
                    </span>
                    <span className="ml-auto text-right text-xs text-mut">{info.detail}</span>
                  </div>
                ))
              )}
            </div>
          </>
        )}
      </div>

      {/* 关于本项目 */}
      <div className="clip-angled border border-line bg-ink-900 p-6">
        <h2 className="mb-3 flex items-center gap-3 text-base font-bold text-fg">
          关于本项目
          <span className="energy-line h-px flex-1 opacity-30" />
        </h2>
        <p className="text-sm leading-relaxed text-mut">
          Live-Document 是一个面向{' '}
          <span className="font-semibold text-radeon-400">AMD DevMaster Hackathon · Track 1</span>{' '}
          的<b className="text-fg">多模态内容创作工具</b>：把教学文档解析为结构化的
          LearningSpec（实体、状态变量、因果链、不变式），再交由多种渲染引擎生成教学动画或短视频。
        </p>
        <div className="mt-4 grid gap-2.5 sm:grid-cols-2">
          {[
            ['◆ 架构', '前后端分离：React (Vite) + FastAPI + SQLite 任务存储'],
            ['◈ Deterministic', 'Manim 渲染，关键推理过程在本地完成'],
            ['✦ Generative', 'LTX / Wan 视频生成（需 AMD Radeon + ROCm）'],
            ['⚡ Procedural', 'PIL 程序化 GIF 兜底，任何环境都能出片'],
          ].map(([t, d]) => (
            <div key={t} className="clip-angled-sm border border-line bg-ink-950 px-4 py-3">
              <div className="font-mono text-xs font-bold text-radeon-400">{t}</div>
              <div className="mt-1 text-xs text-mut">{d}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}