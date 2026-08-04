import { useState } from 'react'
import { api } from '../api'

const FEATURES = ['本地推理', 'ROCm GPU 加速', '三引擎渲染', 'Manim 教学动画']

export default function LoginPage({ onLogin }: { onLogin: (token: string) => void }) {
  const [value, setValue] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const token = value.trim()
    if (!token) return
    setBusy(true)
    setError(null)
    try {
      const res = await api.login(token)
      onLogin(res.token)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden px-4 py-14">
      {/* 背景脉动红光 */}
      <div className="animate-pulse-glow pointer-events-none absolute left-1/2 top-1/3 h-[420px] w-[420px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-radeon-500/15 blur-[110px]" />

      <div className="animate-fade-up relative flex w-full max-w-xl flex-col items-center text-center">
        <div className="clip-angled mb-6 flex h-16 w-16 items-center justify-center bg-gradient-to-br from-radeon-500 to-ember-600 text-2xl font-black text-white glow-red">
          LD
        </div>

        <p className="font-mono text-[11px] uppercase tracking-[0.3em] text-radeon-400">
          AMD DevMaster Hackathon · Track 1
        </p>
        <h1 className="mt-3 text-5xl font-black leading-tight tracking-tight text-fg sm:text-6xl">
          让<span className="text-energy">文档</span>动起来
        </h1>
        <p className="mt-4 max-w-md text-[15px] leading-relaxed text-mut">
          粘贴教学文档，AI 自动拆解知识结构，一键渲染成教学动画与短视频 ——
          全程本地推理，Radeon ROCm 加速。
        </p>

        {/* 登录卡 */}
        <div className="clip-angled mt-9 w-full max-w-md border border-line bg-ink-900/80 p-7 backdrop-blur">
          <h2 className="flex items-center justify-center gap-2 text-sm font-semibold tracking-wide text-fg">
            <span className="energy-line inline-block h-[2px] w-6" />
            访问授权
            <span className="energy-line inline-block h-[2px] w-6" />
          </h2>
          <p className="mt-2 text-xs text-dim">
            本服务通过 Radeon Cloud 公网隧道对外提供，按官方规范须登录后使用。
          </p>
          <form onSubmit={submit} className="mt-5 flex flex-col gap-3 text-left">
            <label
              htmlFor="access-token"
              className="font-mono text-[10px] uppercase tracking-[0.2em] text-dim"
            >
              Access Token
            </label>
            <input
              id="access-token"
              type="password"
              autoComplete="current-password"
              placeholder="输入管理员提供的访问令牌"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              disabled={busy}
              className="clip-angled-sm w-full border border-line bg-ink-950 px-4 py-3 text-sm text-fg outline-none transition-all placeholder:text-dim focus:border-radeon-500/70 focus:glow-red-sm"
            />
            {error && <p className="text-sm text-danger">{error}</p>}
            <button
              type="submit"
              disabled={busy || !value.trim()}
              className="clip-angled-sm mt-1 bg-gradient-to-r from-radeon-600 via-radeon-500 to-ember-600 px-4 py-3 text-sm font-bold tracking-wide text-white transition-all hover:glow-red disabled:cursor-not-allowed disabled:opacity-50"
            >
              {busy ? '验证中…' : '进入工作台 →'}
            </button>
          </form>
        </div>

        {/* 特性 chips */}
        <div className="mt-8 flex flex-wrap items-center justify-center gap-2">
          {FEATURES.map((f) => (
            <span
              key={f}
              className="clip-angled-sm border border-line bg-ink-900/60 px-3 py-1 font-mono text-[10px] uppercase tracking-[0.14em] text-mut"
            >
              {f}
            </span>
          ))}
        </div>

        <p className="mt-6 text-[11px] text-dim">
          令牌显示在服务启动日志中；管理员可用环境变量{' '}
          <code className="border border-line bg-ink-900 px-1.5 py-0.5 font-mono text-[10px] text-radeon-400">
            LIVE_DOC_AUTH_TOKEN
          </code>{' '}
          自定义。
        </p>
      </div>
    </div>
  )
}
