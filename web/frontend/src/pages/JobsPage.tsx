import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api'
import type { Job } from '../types'
import JobCard from '../components/JobCard'

export default function JobsPage({ trackedJobId }: { trackedJobId: string | null }) {
  const [jobs, setJobs] = useState<Job[]>([])
  const [total, setTotal] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const trackedRef = useRef<HTMLDivElement | null>(null)
  const timer = useRef<number | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const r = await api.listJobs(50, 0)
      setJobs(r.jobs)
      setTotal(r.total)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
    return () => {
      if (timer.current) window.clearInterval(timer.current)
    }
  }, [load])

  const hasActive = jobs.some((j) => j.status === 'pending' || j.status === 'running')
  useEffect(() => {
    if (timer.current) window.clearInterval(timer.current)
    timer.current = null
    if (hasActive) {
      timer.current = window.setInterval(() => {
        void load()
      }, 3000)
    }
  }, [hasActive, load])

  useEffect(() => {
    if (trackedJobId) trackedRef.current?.scrollIntoView({ block: 'center', behavior: 'smooth' })
  }, [jobs, trackedJobId])

  const activeCount = jobs.filter((j) => j.status === 'pending' || j.status === 'running').length
  const doneCount = jobs.filter((j) => j.status === 'completed').length

  return (
    <section className="animate-fade-up">
      <div className="clip-angled border border-line bg-ink-900 p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="flex items-center gap-3 text-base font-bold text-fg">
              <span className="clip-angled-sm flex h-7 w-7 items-center justify-center bg-gradient-to-br from-radeon-500 to-ember-600 text-sm text-white glow-red-sm">
                ⚡
              </span>
              任务中心
            </h2>
            <p className="mt-1 font-mono text-[11px] uppercase tracking-[0.14em] text-dim">
              共 {total} 个 ·{' '}
              <span className={activeCount ? 'text-radeon-400' : 'text-dim'}>{activeCount} 渲染中</span>{' '}
              ·{' '}
              <span className={doneCount ? 'text-ok' : 'text-dim'}>{doneCount} 已完成</span>
            </p>
          </div>
          <button
            onClick={() => void load()}
            disabled={loading}
            className="clip-angled-sm border border-line bg-ink-850 px-4 py-2 text-sm text-mut transition-all hover:border-radeon-500/60 hover:text-fg disabled:opacity-50"
          >
            {loading ? '刷新中…' : '↻ 刷新'}
          </button>
        </div>

        {error && <p className="mt-4 text-sm text-danger">{error}</p>}
        {jobs.length === 0 && !loading && (
          <p className="mt-6 text-center text-sm text-dim">
            暂无任务 —— 去「文档规划」提交第一个动画生成任务。
          </p>
        )}
        <div className="mt-4 flex flex-col gap-3">
          {jobs.map((job) => (
            <JobCard
              key={job.id}
              job={job}
              highlighted={job.id === trackedJobId}
              ref={job.id === trackedJobId ? trackedRef : undefined}
            />
          ))}
        </div>
      </div>
    </section>
  )
}