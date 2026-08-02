export type EngineName = 'deterministic' | 'generative' | 'procedural'

export interface CausalStep {
  cause?: string
  change?: string
  visual_evidence?: string
  [k: string]: unknown
}

export interface Spec {
  learning_goal: string | null
  entities: string[]
  state_variables: string[]
  causal_steps: CausalStep[]
  invariants: string[]
  comprehension_questions: string[]
  fallback_reason: string | null
}

export interface SpecsResponse {
  specs: Spec[]
  count: number
  suitable: number
}

export interface JobCreate {
  text?: string
  spec?: Spec
  engine: EngineName
  style?: Record<string, unknown>
}

export type JobStatus = 'pending' | 'running' | 'completed' | 'failed'

export interface Job {
  id: string
  engine: string
  status: JobStatus
  progress: number
  message: string | null
  created_at: string | null
  updated_at: string | null
  spec: Spec | null
  artifacts: Record<string, string>
  metrics: Record<string, unknown>
  error: { type: string; message: string } | null
}

export interface JobListResponse {
  jobs: Job[]
  total: number
}

export interface EngineHealth {
  available: boolean
  detail: string
}

export interface HealthResponse {
  status: string
  python: string
  gpu: { available: boolean; detail: string } & Record<string, unknown>
  engines: Record<string, EngineHealth>
}
