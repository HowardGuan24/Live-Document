import { useCallback, useEffect, useState } from 'react'
import PlannerPage from './pages/PlannerPage'
import JobsPage from './pages/JobsPage'
import AboutPage from './pages/AboutPage'
import LoginPage from './pages/LoginPage'
import { clearToken, getToken, setToken, setUnauthorizedHandler } from './api'

export type Tab = 'planner' | 'jobs' | 'about'

const TABS: { id: Tab; label: string }[] = [
  { id: 'planner', label: '文档规划' },
  { id: 'jobs', label: '任务中心' },
  { id: 'about', label: '运行状态' },
]

function Brand() {
  return (
    <div className="flex items-center gap-3">
      <div className="clip-angled-sm flex h-9 w-9 shrink-0 items-center justify-center bg-gradient-to-br from-radeon-500 to-ember-600 text-[15px] font-black text-white glow-red-sm">
        LD
      </div>
      <div className="leading-tight">
        <div className="text-[15px] font-black italic tracking-tight text-fg">
          LIVE-DOCUMENT
        </div>
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-dim">
          文档动态化引擎 · AMD Radeon
        </div>
      </div>
    </div>
  )
}

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
    return <LoginPage onLogin={handleLogin} />
  }

  return (
    <div className="flex min-h-full flex-col">
      <header className="sticky top-0 z-20 border-b border-line bg-ink-950/85 backdrop-blur">
        <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-3">
          <Brand />
          <nav className="flex items-center gap-1">
            {TABS.map((t) => {
              const active = tab === t.id
              return (
                <button
                  key={t.id}
                  onClick={() => setTab(t.id)}
                  className={`relative px-4 py-2 text-sm transition-colors ${
                    active ? 'font-semibold text-fg' : 'text-mut hover:text-fg'
                  }`}
                >
                  {t.label}
                  {active && (
                    <span className="energy-line absolute inset-x-3 -bottom-px h-[2px] glow-red-sm" />
                  )}
                </button>
              )
            })}
            <button
              onClick={handleLogout}
              title="退出登录"
              className="clip-angled-sm ml-2 border border-line px-3.5 py-1.5 text-sm text-mut transition-all hover:border-radeon-500/60 hover:text-radeon-400"
            >
              退出
            </button>
          </nav>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6">
        {tab === 'planner' && <PlannerPage onSubmitted={goJobs} />}
        {tab === 'jobs' && <JobsPage trackedJobId={trackedJobId} />}
        {tab === 'about' && <AboutPage />}
      </main>

      <footer className="border-t border-line py-3 text-center font-mono text-[10px] uppercase tracking-[0.16em] text-dim">
        Live-Document · 本地推理 / Radeon ROCm 就绪 · AMD DevMaster Track 1
      </footer>
    </div>
  )
}
