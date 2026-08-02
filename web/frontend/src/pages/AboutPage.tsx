import { useEffect, useState } from 'react'
import { api } from '../api'
import type { HealthResponse } from '../types'

export default function AboutPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  return (
    <section className="page">
      <div className="panel">
        <h2>运行状态</h2>
        {error && <p className="error-text">{error}</p>}
        {!health && !error && <p className="muted">正在读取健康信息…</p>}
        {health && (
          <>
            <div className="health-grid">
              <div className="health-card">
                <span className="muted">服务</span>
                <strong>{health.status}</strong>
              </div>
              <div className="health-card">
                <span className="muted">Python</span>
                <strong>{health.python}</strong>
              </div>
              <div className="health-card">
                <span className="muted">GPU / ROCm</span>
                <strong className={health.gpu.available ? 'ok' : 'warn'}>
                  {health.gpu.available ? '可用' : '不可用'}
                </strong>
                <p className="muted">{health.gpu.detail}</p>
              </div>
            </div>
            <table className="engine-table">
              <thead>
                <tr>
                  <th>引擎</th>
                  <th>可用</th>
                  <th>说明</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(health.engines).map(([name, info]) => (
                  <tr key={name}>
                    <td><code>{name}</code></td>
                    <td>
                      <span className={`badge ${info.available ? 'badge-ok' : 'badge-warn'}`}>
                        {info.available ? 'available' : 'unavailable'}
                      </span>
                    </td>
                    <td className="muted">{info.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>
      <div className="panel">
        <h2>关于本项目</h2>
        <p>
          Live-Document 是一个<b>多模态内容创作工具</b>：把教学文档解析为结构化的
          LearningSpec（实体、状态变量、因果链、不变式），再交由多种渲染引擎生成教学动画或短视频。
        </p>
        <ul className="plain-list">
          <li>• 前后端分离：React (Vite) 前端 + FastAPI 后端 + SQLite 任务存储</li>
          <li>• Deterministic 引擎：Manim 渲染，关键推理过程在本地完成</li>
          <li>• Generative 引擎：LTX / Wan 视频生成（需 AMD Radeon + ROCm）</li>
          <li>• Procedural 引擎：PIL 程序化 GIF 兜底，保证任何环境都能出片</li>
        </ul>
      </div>
    </section>
  )
}
