# Live-Document Web（前后端分离版）

面向 **AMD Dev Contest 2026 · Track 1（Multimodal Content Creation Tools）** 的 Web 交付形态：
把教学文档解析为轻量 **LearningSpec**，再交给现有渲染流程生成教学动画/短视频。

```
┌─────────────────────┐         ┌──────────────────────────────┐
│  React SPA (Vite)   │  /api   │  FastAPI 后端                 │
│  web/frontend/dist  │ ──────▶ │  - 文档规划 → LearningSpec     │
│  （dev: Vite 代理）  │ ◀────── │  - 任务队列（asyncio + worker）│
└─────────────────────┘  JSON   │  - SQLite 任务存储（JobStore） │
                                │  - 渲染引擎：deterministic /   │
                                │    generative / procedural     │
                                └──────────────────────────────┘
```

## 目录结构

| 路径 | 说明 |
|---|---|
| `backend/app` | FastAPI 应用（API、服务、任务管理） |
| `backend/requirements.txt` | 后端完整依赖（FastAPI / Manim / Pillow 等） |
| `frontend/` | React + Vite + TypeScript 前端（SPA） |
| `frontend/dist/` | 前端构建产物（由后端静态托管） |
| `backend/data/` | 运行时数据（SQLite + 任务产物，已 gitignore） |

## 快速开始

### 1. 后端

```bash
cd web/backend
python -m venv .venv && . .venv/bin/activate   # 或复用仓库根 .venv
pip install -r requirements.txt

# 仅 API（前端另起 dev server）
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# 或同时托管构建好的前端（生产模式）
LIVE_DOC_SERVE_FRONTEND=1 python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `LIVE_DOC_SERVE_FRONTEND` | `0` | `1` 时在 `/` 托管 `frontend/dist` |
| `LIVE_DOC_DATA_DIR` | `backend/data` | 运行时数据目录（DB + 产物） |
| `LIVE_DOC_HOST` / `LIVE_DOC_PORT` | `127.0.0.1:8000` | 监听地址 |
| `LIVE_DOC_CORS_ORIGINS` | `*` | 允许的 CORS 源（逗号分隔） |

### 2. 前端（开发模式）

```bash
cd web/frontend
npm install
npm run dev        # http://localhost:5173，/api 代理到 127.0.0.1:8000
```

### 3. 前端（生产构建）

```bash
cd web/frontend
npm install
npm run build      # 产出 dist/，由后端 LIVE_DOC_SERVE_FRONTEND=1 托管
```

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/v1/health` | 服务 / GPU(ROCm) / 引擎可用性 |
| `POST` | `/api/v1/specs` | 文本 → LearningSpec 列表（`suitable` 为可动画化数） |
| `POST` | `/api/v1/jobs` | 提交渲染任务（`engine` + `text` 或 `spec`） |
| `GET` | `/api/v1/jobs` | 任务列表（`?limit=&offset=`） |
| `GET` | `/api/v1/jobs/{id}` | 任务状态（进度 / 产物 / 错误） |
| `GET` | `/api/v1/jobs/{id}/files/{name}` | 下载产物（mp4 / gif / json） |

交互式文档：`http://localhost:8000/docs`

## 渲染引擎

| 引擎 | 实现 | 说明 |
|---|---|---|
| `deterministic` | procedural 兼容回退 | 旧 Manim 模块已移除；请求会明确回退到 PIL GIF |
| `procedural` | PIL 程序化渲染 | 无模型依赖的轻量兜底 GIF，始终可用 |
| `generative` | 三阶段流程（`modules/`，M3 接入中） | 生成式视频，需要 AMD Radeon GPU + ROCm；当前不可用时自动回退 procedural（`metrics.fallback_reason`） |

任务状态机：`pending → running → completed / failed`。服务重启后遗留的 pending/running 任务会被标记为 failed（`StaleJob`）。

## 部署到公网（Radeon Cloud）

官方 Radeon Cloud 提供 `rc-tunnel` 公网暴露，一次只能暴露**一个端口**，因此推荐**单端口一体化部署**（FastAPI 同时托管 API + 前端静态资源）：

```bash
cd web/backend
LIVE_DOC_SERVE_FRONTEND=1 python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 另开终端（Radeon Cloud 环境）
rc-tunnel expose --port 8000
# → 得到 https://rc-*.radeon.firstdg.ai 公网地址
```

- 该公网 URL 任何人都可访问，系统内**必须**加登录/鉴权后才能对外开放（建议在 FastAPI 层加简单 Bearer Token 中间件，前端登录后携带 token）。
- 前端 `VITE_API_BASE` 默认 `/api/v1`（同源），公网单端口部署无需额外配置。

## 比赛规范对照（Track 1）

- ✅ Web UI 交付形态：React SPA + FastAPI，前后端分离
- ✅ 关键推理在本地：轻量文档规划与程序化渲染均在本地完成，无封闭在线 API 依赖
- ✅ AMD Radeon GPU + ROCm：`generative` 引擎面向 `modules/` 三阶段流程，`/api/v1/health` 上报 GPU 可用性
- ✅ 提交物配套：源码仓库（本目录）+ 演示视频（可用 deterministic 引擎生成）+ PPT/Poster

## 公网部署（Radeon Cloud rc-tunnel）

官方要求（Radeon Cloud 用户指南）：公网 URL 可达互联网，**应用自身必须强制登录**。
本系统已内置 Bearer Token 登录：所有 `/api/*`（除 `/api/v1/auth/login`）都要求
`Authorization: Bearer <token>`；前端未登录只显示登录页。

### 前置条件

- Notebook 必须是**新创建**的 Pod（旧 Pod 无 `FRP_BROKER_URL` 环境变量，`rc-tunnel` 无法安装）。
  验证：`env | grep FRP_BROKER_URL`。
- 每 Pod 只能暴露一个端口 → 使用一体化模式（后端托管前端，端口 8000）。

### 一键部署

```bash
git clone https://github.com/HowardGuan24/Live-Document.git
cd Live-Document
bash web/deploy-public.sh
```

脚本会：构建前端 → 启动后端（自动生成/读取访问令牌）→ 安装 rc-tunnel → 暴露端口 8000，
最后打印公网地址（形如 `https://rc-xxxx.radeon.firstdg.ai`）与访问令牌。

### 手动部署

```bash
cd web/frontend && npm install && npm run build && cd ..
cd backend
LIVE_DOC_AUTH_TOKEN=your-token LIVE_DOC_SERVE_FRONTEND=1 \
  /path/to/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# 新终端：
$HOME/.local/bin/rc-tunnel expose --port 8000
```

- 令牌来源：优先环境变量 `LIVE_DOC_AUTH_TOKEN`；未设置则自动生成并持久化到
  `web/backend/data/auth_token.txt`，启动日志也会打印。
- 产物（视频/GIF）下载接受 `?access_token=` 查询参数（浏览器 `<video>/<a>` 无法带请求头）。
- 本地开发可设 `LIVE_DOC_AUTH_DISABLED=1` 关闭鉴权。
