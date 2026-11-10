import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Empty, ErrorState, Kpi, PageHead } from '../components/ui'
import { exportReport, toItem } from '../lib/api'
import { useSession } from '../state/Session'

export default function Reports() {
  const { entries, clear } = useSession()
  const [scope, setScope] = useState<'all' | 'bulk' | 'single'>('all')
  const [busy, setBusy] = useState<string>()
  const [err, setErr] = useState<string>()
  const items = useMemo(() => entries.filter((e) => scope === 'all' || e.origin === scope), [entries, scope])
  const go = async (f: 'csv' | 'html') => {
    setBusy(f); setErr(undefined)
    try { await exportReport(items.map(toItem), f) } catch (e) { setErr((e as Error).message) } finally { setBusy(undefined) }
  }
  return (
    <>
      <PageHead eyebrow="Export" title="Reports"
        lead="Download what you analysed in this session. The CSV opens in Excel with Urdu text intact; the report is a printable page — use your browser's Print → Save as PDF to keep it as a PDF." />
      {!entries.length ? <Empty title="Nothing to export yet"><p>Analyse reviews on the <Link to="/app/analyze">Review Analyzer</Link> or <Link to="/app/bulk">Bulk Analysis</Link> page first.</p></Empty> : (
        <div className="stack">
          <div className="row"><div className="segmented">{(['all', 'bulk', 'single'] as const).map((s) =>
            <button key={s} className={scope === s ? 'on' : ''} onClick={() => setScope(s)}>{s === 'all' ? 'All results' : s === 'bulk' ? 'Bulk only' : 'Single only'}</button>)}</div></div>
          <div className="kpis"><Kpi ink label="Reviews to export" value={items.length.toLocaleString()} />
            <Kpi label="Single" value={entries.filter((e) => e.origin === 'single').length.toLocaleString()} />
            <Kpi label="Bulk" value={entries.filter((e) => e.origin === 'bulk').length.toLocaleString()} /></div>
          <div className="grid g2">
            <div className="card"><h3 style={{ marginBottom: 6 }}>Results CSV</h3><p className="small muted" style={{ marginBottom: 14 }}>One row per review: text, detected language, sentiment, confidence, the three probabilities and the low-confidence flag.</p>
              <button className="btn btn-primary" disabled={!items.length || !!busy} onClick={() => go('csv')}>{busy === 'csv' ? 'Preparing…' : 'Download CSV'}</button></div>
            <div className="card"><h3 style={{ marginBottom: 6 }}>Summary report</h3><p className="small muted" style={{ marginBottom: 14 }}>Sentiment and language breakdown, distinctive words, low-confidence reviews and the model's limitations, ready to print.</p>
              <button className="btn btn-saffron" disabled={!items.length || !!busy} onClick={() => go('html')}>{busy === 'html' ? 'Preparing…' : 'Download report'}</button></div>
          </div>
          {err && <ErrorState message={err} />}
          <div className="note ink small">Results are kept only in this browser tab and are cleared when you close it. <button className="btn btn-ghost btn-sm" onClick={() => clear()}>Clear all results now</button></div>
        </div>
      )}
    </>
  )
}
