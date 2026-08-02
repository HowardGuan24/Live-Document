import { useState } from 'react'
import { api } from '../api'

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
    <section className="page">
      <div className="panel login-panel">
        <h2>访问授权</h2>
        <p className="muted">
          本服务通过 AMD Radeon Cloud 公网隧道对外提供，按官方规范必须登录后才能使用。
        </p>
        <form onSubmit={submit} className="login-form">
          <label htmlFor="access-token">访问令牌（Access Token）</label>
          <input
            id="access-token"
            type="password"
            autoComplete="current-password"
            placeholder="输入管理员提供的访问令牌"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            disabled={busy}
          />
          {error && <p className="error-text">{error}</p>}
          <button type="submit" className="btn-primary" disabled={busy || !value.trim()}>
            {busy ? '验证中…' : '登录'}
          </button>
        </form>
        <p className="muted small">
          提示：令牌显示在服务启动日志中；管理员可用环境变量{' '}
          <code>LIVE_DOC_AUTH_TOKEN</code> 自定义。
        </p>
      </div>
    </section>
  )
}
