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

  return (
    <section className="page">
      <div className="panel">
        <h2>
          任务中心{' '}
          <span className="muted">（共 {total} 个任务，{hasActive ? '有任务渲染中…' : ''}）</span>
        </h2>
        <button className="btn" onClick={() => void load()} disabled={loading}>
          {loading ? '刷新中…' : '刷新'}
        </button>
        {error && <p className="error-text">{error}</p>}
        {jobs.length === 0 && !loading && <p className="muted">暂无任务，去「文档规划」提交第一个任务吧。</p>}
        <div className="job-list">
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
