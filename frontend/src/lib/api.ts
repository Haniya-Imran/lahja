// Centralised API client. Same origin by default (Vercel serves the API at /api);
// set VITE_API_URL only if the API lives elsewhere. Vite proxies /api in development.
/* eslint-disable @typescript-eslint/no-explicit-any */
export const API_URL: string = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')

export type Sentiment = 'positive' | 'neutral' | 'negative'

export interface LanguageInfo {
  language: string
  label: string
  script: string
  confidence: number
  arabic_share: number
  roman_urdu_words: number
  english_words: number
  note: string | null
}

export interface Explanation {
  available: boolean
  reason?: string
  method?: string
  explains?: Sentiment
  tokens?: { word: string; start: number; end: number; score: number }[]
  top?: { word: string; score: number }[]
  note?: string
}

export interface Result {
  index: number
  text: string
  normalized: string
  language: LanguageInfo
  sentiment: Sentiment | null
  confidence: number | null
  probabilities: Record<Sentiment, number> | null
  low_confidence: boolean
  windows?: number
  skipped?: string
  explanation?: Explanation
}

export class ApiError extends Error {
  code: string
  status: number
  constructor(status: number, code: string, message: string) { super(message); this.status = status; this.code = code }
}

async function request<T>(path: string, init: RequestInit = {}, timeoutMs = 60000): Promise<T> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), timeoutMs)
  let res: Response
  try { res = await fetch(API_URL + path, { ...init, signal: ctrl.signal }) } catch (e: any) {
    clearTimeout(timer)
    throw new ApiError(0, e?.name === 'AbortError' ? 'timeout' : 'network',
      e?.name === 'AbortError' ? 'The analysis server took too long. Please try again.' : 'Cannot reach the analysis server. Check your connection and try again.')
  }
  clearTimeout(timer)
  const text = await res.text()
  let body: any = null
  try { body = text ? JSON.parse(text) : null } catch { body = null }
  if (!res.ok) throw new ApiError(res.status, body?.error?.code || 'http_error', body?.error?.message || `Request failed (${res.status})`)
  return body as T
}

const json = (b: unknown): RequestInit => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) })

export interface SummaryItem { text: string; sentiment: Sentiment | null; confidence: number | null; language: string; low_confidence: boolean; probabilities: Record<string, number> | null }

// Text is capped at 2,000 characters so a full session fits within the 4.5 MB request limit.
export const toItem = (r: Result): SummaryItem => ({
  text: r.text.length > 2000 ? r.text.slice(0, 2000) : r.text, sentiment: r.sentiment, confidence: r.confidence, language: r.language.language,
  low_confidence: r.low_confidence, probabilities: r.probabilities,
})

export const api = {
  health: () => request<{ status: string; model_loaded: boolean }>('/api/health'),
  modelInfo: () => request<any>('/api/model-info', {}, 90000),
  examples: () => request<{ note: string; examples: { language: string; text: string }[] }>('/api/examples'),
  analyze: (text: string, explain = true) => request<Result & { model: any; elapsed_ms: number }>('/api/analyze', json({ text, explain }), 90000),
  batch: (reviews: string[]) => request<{ results: Result[]; model: any; elapsed_ms: number }>('/api/analyze/batch', json({ reviews }), 120000),
  inspect: (file: File) => { const fd = new FormData(); fd.append('file', file); return request<any>('/api/reviews/inspect', { method: 'POST', body: fd }, 90000) },
  summary: (items: SummaryItem[]) => request<any>('/api/analytics/summary', json({ items }), 90000),
}

export async function exportReport(items: SummaryItem[], format: 'csv' | 'html') {
  const res = await fetch(API_URL + '/api/reports/export', json({ items, format }))
  if (!res.ok) {
    let msg = `Export failed (${res.status})`
    try { msg = (await res.json()).error.message } catch { /* ignore */ }
    throw new ApiError(res.status, 'export_failed', msg)
  }
  const blob = await res.blob()
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = format === 'csv' ? 'lahja-review-analysis.csv' : 'lahja-review-report.html'
  document.body.appendChild(a); a.click(); a.remove()
  setTimeout(() => URL.revokeObjectURL(a.href), 3000)
}

export const hasUrdu = (s: string) => /[؀-ۿ]/.test(s)
