import { useCallback, useEffect, useState } from 'react'
import PlannerPage from './pages/PlannerPage'
import JobsPage from './pages/JobsPage'
import AboutPage from './pages/AboutPage'
import LoginPage from './pages/LoginPage'
import { clearToken, getToken, setToken, setUnauthorizedHandler } from './api'

export type Tab = 'planner' | 'jobs' | 'about'

export default function App() {
  const [token, setTokenState] = useState<string | null>(() => getToken())
  const [tab, setTab] = useState<Tab>('planner')
  const [trackedJobId, setTrackedJobId] = useState<string | null>(null)

  useEffect(() => {
    setUnauthorizedHandler(() => setTokenState(null))
    return () => setUnauthorizedHandler(null)
  }, [])

  const handleLogin = useCallback((t: string) => {
    setToken(t)
    setTokenState(t)
  }, [])

  const handleLogout = useCallback(() => {
    clearToken()
    setTokenState(null)
    setTab('planner')
  }, [])

  const goJobs = useCallback((jobId?: string) => {
    if (jobId) setTrackedJobId(jobId)
    setTab('jobs')
  }, [])

  if (!token) {
    return (
      <div className="app">
        <header className="app-header">
          <div className="brand">
            <span className="brand-mark">LD</span>
            <div>
              <h1>Live-Document</h1>
              <p className="subtitle">文档 → LearningSpec → 教学动画 · AMD 多模态内容创作</p>
            </div>
          </div>
        </header>
        <main className="app-main">
          <LoginPage onLogin={handleLogin} />
        </main>
        <footer className="app-footer">Live-Document Web · 公网访问需登录（Radeon Cloud 规范）</footer>
      </div>
    )
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark">LD</span>
          <div>
            <h1>Live-Document</h1>
            <p className="subtitle">文档 → LearningSpec → 教学动画 · AMD 多模态内容创作</p>
          </div>
        </div>
        <nav className="tabs">
          <button className={tab === 'planner' ? 'tab active' : 'tab'} onClick={() => setTab('planner')}>
            文档规划
          </button>
          <button className={tab === 'jobs' ? 'tab active' : 'tab'} onClick={() => setTab('jobs')}>
            任务中心
          </button>
          <button className={tab === 'about' ? 'tab active' : 'tab'} onClick={() => setTab('about')}>
            运行状态
          </button>
          <button className="tab tab-logout" onClick={handleLogout} title="退出登录">
            退出
          </button>
        </nav>
      </header>
      <main className="app-main">
        {tab === 'planner' && <PlannerPage onSubmitted={goJobs} />}
        {tab === 'jobs' && <JobsPage trackedJobId={trackedJobId} />}
        {tab === 'about' && <AboutPage />}
      </main>
      <footer className="app-footer">
        Live-Document Web · 本地推理 / Radeon ROCm 就绪 · 前后端分离架构
      </footer>
    </div>
  )
}
