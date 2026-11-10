import { useEffect, useRef, useState, type ReactNode } from 'react'
import { SENT_LABEL } from '../brand'
import { hasUrdu } from '../lib/api'

export function Logo({ size = 34 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" aria-hidden="true">
      <rect width="64" height="64" rx="16" fill="#1B1A3A" />
      <path d="M14 40c6-14 12-14 18 0s12 14 18 0" fill="none" stroke="#E8A33D" strokeWidth="6" strokeLinecap="round" />
      <circle cx="32" cy="20" r="5" fill="#fff" />
    </svg>
  )
}

export function Info({ text }: { text: string }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    if (!open) return
    const close = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false) }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [open])
  return (
    <span className="info" ref={ref} onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <button type="button" className="info-btn" aria-label="What does this mean?" aria-expanded={open} onClick={() => setOpen((o) => !o)}>i</button>
      {open && <span role="tooltip" className="info-tip">{text}</span>}
    </span>
  )
}

export const Loading = ({ label = 'Loading…' }: { label?: string }) => (
  <div className="loading" role="status"><span className="spinner" />{label}</div>
)

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="state state-error" role="alert">
      <b>Something went wrong</b><span>{message}</span>
      {onRetry && <button className="btn btn-sm" onClick={onRetry}>Try again</button>}
    </div>
  )
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return <div className="state state-empty"><b>{title}</b>{children}</div>
}

export function SentBadge({ s }: { s: string | null }) {
  if (!s) return <span className="badge b-line">Skipped</span>
  const cls = s === 'positive' ? 'b-pos' : s === 'negative' ? 'b-neg' : 'b-neu'
  return <span className={`badge ${cls}`}>{SENT_LABEL[s]}</span>
}

/** Renders review text with the right font and direction for Urdu script. */
export function ReviewText({ text, className = '' }: { text: string; className?: string }) {
  return <span dir="auto" className={`${hasUrdu(text) ? 'ur' : ''} ${className}`}>{text}</span>
}

export function PageHead({ eyebrow, title, lead, actions }: { eyebrow: string; title: string; lead?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="page-head">
      <div><span className="eyebrow">{eyebrow}</span><h1>{title}</h1>{lead && <p className="lead">{lead}</p>}</div>
      {actions && <div className="row">{actions}</div>}
    </header>
  )
}

export function Kpi({ label, value, hint, ink, info }: { label: string; value: ReactNode; hint?: ReactNode; ink?: boolean; info?: string }) {
  return (
    <div className={`kpi ${ink ? 'ink' : ''}`}>
      <span>{label}{info && <Info text={info} />}</span><b>{value}</b>{hint && <small>{hint}</small>}
    </div>
  )
}

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]) {
  const [state, setState] = useState<{ data?: T; error?: string; loading: boolean }>({ loading: true })
  const [tick, setTick] = useState(0)
  useEffect(() => {
    let alive = true
    setState((s) => ({ ...s, loading: true, error: undefined }))
    fn().then((data) => alive && setState({ data, loading: false }))
      .catch((e) => alive && setState({ error: e?.message || 'Request failed', loading: false }))
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick])
  return { ...state, retry: () => setTick((t) => t + 1) }
}
