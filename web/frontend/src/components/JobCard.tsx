import { forwardRef } from 'react'
import { formatTime, resolveArtifactUrl } from '../api'
import type { Job } from '../types'

function artifactLabel(key: string): string {
  const labels: Record<string, string> = {
    video: '视频 MP4',
    gif: '动图 GIF',
    preview: '预览',
    normalized_spec: '规范 JSON',
    result: '结果 JSON',
  }
  return labels[key] ?? key
}

const JobCard = forwardRef<HTMLDivElement, { job: Job; highlighted: boolean }>(
  function JobCard({ job, highlighted }, ref) {
    const active = job.status === 'pending' || job.status === 'running'
    return (
      <article
        ref={ref}
        className={`job-card ${highlighted ? 'job-highlight' : ''} status-${job.status}`}
      >
        <header className="job-head">
          <span className="job-id" title={job.id}>#{job.id}</span>
          <span className="badge badge-engine">{job.engine}</span>
          <span className={`badge status-${job.status}`}>{job.status}</span>
          <span className="muted job-time">{formatTime(job.created_at)}</span>
        </header>
        {job.spec?.learning_goal && <p className="job-goal">{job.spec.learning_goal}</p>}
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${Math.round(job.progress * 100)}%` }} />
          <span className="progress-label">{Math.round(job.progress * 100)}%</span>
        </div>
        <p className="job-message muted">
          {job.message ?? ''}
          {active && '（正在渲染，页面每 3 秒自动刷新…）'}
        </p>
        {job.error && (
          <div className="job-error">
            <strong>错误：</strong>
            {job.error.message || job.error.type}
          </div>
        )}
        {Object.keys(job.metrics).length > 0 && (
          <div className="job-metrics muted">
            {Object.entries(job.metrics).map(([k, v]) => (
              <span key={k} className="metric-pill">
                {k}: {typeof v === 'object' ? JSON.stringify(v) : String(v)}
              </span>
            ))}
          </div>
        )}
        {Object.keys(job.artifacts).length > 0 && (
          <div className="artifact-list">
            {Object.entries(job.artifacts).map(([key, url]) => (
              <a
                key={key}
                className="artifact-link"
                href={resolveArtifactUrl(url)}
                target="_blank"
                rel="noreferrer"
                download
              >
                ⬇ {artifactLabel(key)}
              </a>
            ))}
          </div>
        )}
      </article>
    )
  },
)

export default JobCard
