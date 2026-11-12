import Papa from 'papaparse'
import { useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { LANG_LABEL } from '../brand'
import { ReviewCell } from '../components/ResultView'
import { Empty, ErrorState, Info, Kpi, PageHead, SentBadge } from '../components/ui'
import { api, exportReport, toItem, type Result } from '../lib/api'
import { useSession } from '../state/Session'

/* eslint-disable @typescript-eslint/no-explicit-any */
const BATCH = 48
const MAX_ROWS = 5000
const MAX_MB = 4

type Parsed = { rows: Record<string, string>[]; columns: string[]; filename: string; inspect: any }

export default function Bulk() {
  const { entries, add, clear } = useSession()
  const [parsed, setParsed] = useState<Parsed | null>(null)
  const [column, setColumn] = useState('')
  const [pasted, setPasted] = useState('')
  const [mode, setMode] = useState<'csv' | 'paste'>('csv')
  const [over, setOver] = useState(false)
  const [err, setErr] = useState<string>()
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState({ done: 0, total: 0 })
  const cancel = useRef(false)
  const [filter, setFilter] = useState({ q: '', sentiment: '', language: '', low: false })
  const [page, setPage] = useState(0)
  const results = useMemo(() => entries.filter((e) => e.origin === 'bulk'), [entries])

  const onFile = async (f?: File | null) => {
    setErr(undefined); setParsed(null)
    if (!f) return
    if (!/\.(csv|tsv|txt)$/i.test(f.name)) { setErr('Please choose a .csv file (comma, semicolon or tab separated).'); return }
    if (f.size > MAX_MB * 1024 * 1024) { setErr(`This file is larger than ${MAX_MB} MB. Split it into smaller files.`); return }
    try {
      const inspect = await api.inspect(f)
      const text = await f.text()
      const out = Papa.parse<Record<string, string>>(text.replace(/^﻿/, ''), { header: true, skipEmptyLines: 'greedy', delimiter: inspect.delimiter })
      const columns = (out.meta.fields || []).filter(Boolean)
      if (!columns.length || !out.data.length) { setErr('The file has no rows we could read.'); return }
      setParsed({ rows: out.data, columns, filename: f.name, inspect })
      setColumn(columns.includes(inspect.suggested_column) ? inspect.suggested_column : columns[0])
    } catch (e) { setErr((e as Error).message) }
  }

  const texts = (): string[] => {
    if (mode === 'paste') return pasted.split(/\r?\n/).map((s) => s.trim()).filter(Boolean)
    return (parsed?.rows || []).map((r) => String(r[column] ?? '').trim()).filter(Boolean)
  }

  const run = async () => {
    const all = texts()
    if (!all.length) { setErr(mode === 'paste' ? 'Paste at least one review (one per line).' : 'The chosen column has no text.'); return }
    if (all.length > MAX_ROWS) { setErr(`Please analyse at most ${MAX_ROWS.toLocaleString()} reviews at a time (this file has ${all.length.toLocaleString()}).`); return }
    setErr(undefined); setBusy(true); cancel.current = false; setProgress({ done: 0, total: all.length })
    const collected: Result[] = []
    try {
      for (let i = 0; i < all.length; i += BATCH) {
        if (cancel.current) break
        const chunk = all.slice(i, i + BATCH).map((t) => t.slice(0, 5000))
        let tries = 0
        for (;;) {
          try { const r = await api.batch(chunk); collected.push(...r.results.map((x) => ({ ...x, index: x.index + i }))); break }
          catch (e) { if (++tries >= 3) throw e; await new Promise((ok) => setTimeout(ok, 1200 * tries)) }
        }
        setProgress({ done: Math.min(i + BATCH, all.length), total: all.length })
      }
    } catch (e) { setErr(`Stopped after ${collected.length.toLocaleString()} reviews: ${(e as Error).message}`) }
    if (collected.length) { clear('bulk'); add([...collected].reverse(), 'bulk'); setPage(0) }
    setBusy(false)
  }

  const view = useMemo(() => results.filter((r) =>
    (!filter.sentiment || r.sentiment === filter.sentiment) && (!filter.language || r.language.language === filter.language) &&
    (!filter.low || r.low_confidence) && (!filter.q || r.text.toLowerCase().includes(filter.q.toLowerCase()))
  ).sort((a, b) => a.index - b.index), [results, filter])
  const PER = 50
  const counts = useMemo(() => ({
    pos: results.filter((r) => r.sentiment === 'positive').length, neg: results.filter((r) => r.sentiment === 'negative').length,
    neu: results.filter((r) => r.sentiment === 'neutral').length, low: results.filter((r) => r.low_confidence).length,
  }), [results])
  const langs = [...new Set(results.map((r) => r.language.language))]
  const dl = async (f: 'csv' | 'html') => { try { await exportReport(view.map(toItem), f) } catch (e) { setErr((e as Error).message) } }

  return (
    <>
      <PageHead eyebrow="Many reviews" title="Bulk Analysis"
        lead="Upload a CSV of reviews (up to 5,000 rows, 4 MB) or paste one review per line. LingoAI suggests the review column, analyses in small batches and keeps the results in this browser tab." />
      <div className="card">
        <div className="card-head">
          <div className="segmented" role="tablist">
            <button role="tab" aria-selected={mode === 'csv'} className={mode === 'csv' ? 'on' : ''} onClick={() => setMode('csv')}>Upload CSV</button>
            <button role="tab" aria-selected={mode === 'paste'} className={mode === 'paste' ? 'on' : ''} onClick={() => setMode('paste')}>Paste reviews</button>
          </div>
          <a className="btn btn-sm" href="/sample-reviews.csv" download>Download sample CSV</a>
        </div>
        {mode === 'csv' ? (
          <>
            <label className={`drop ${over ? 'over' : ''}`} style={{ display: 'block', cursor: 'pointer' }}
              onDragOver={(e) => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)}
              onDrop={(e) => { e.preventDefault(); setOver(false); onFile(e.dataTransfer.files?.[0]) }}>
              <input type="file" accept=".csv,.tsv,.txt,text/csv" className="sr-only" onChange={(e) => onFile(e.target.files?.[0])} />
              <b>{parsed ? parsed.filename : 'Drop a CSV here or click to choose'}</b>
              <p className="small muted">UTF-8 or Windows encodings · comma, semicolon or tab separated · first row must be the column names</p>
            </label>
            {parsed && (
              <div className="grid g2" style={{ marginTop: 14, alignItems: 'end' }}>
                <label className="field"><span>Review text column <Info text="Suggested automatically: the column with the longest, most letter-rich values." /></span>
                  <select className="input" value={column} onChange={(e) => setColumn(e.target.value)}>
                    {parsed.columns.map((c) => <option key={c} value={c}>{c}{c === parsed.inspect.suggested_column ? ' (suggested)' : ''}</option>)}
                  </select></label>
                <div className="small muted num">{parsed.rows.length.toLocaleString()} rows · {texts().length.toLocaleString()} with text in “{column}”
                  {parsed.inspect.ragged_rows ? ` · ${parsed.inspect.ragged_rows} rows had a different number of columns` : ''}</div>
              </div>
            )}
            {parsed && (
              <div className="table-wrap" style={{ marginTop: 12, maxHeight: 220 }}><table className="tbl">
                <thead><tr>{parsed.columns.slice(0, 6).map((c) => <th key={c} style={c === column ? { color: 'var(--indigo)' } : undefined}>{c}</th>)}</tr></thead>
                <tbody>{parsed.rows.slice(0, 5).map((r, i) => <tr key={i}>{parsed.columns.slice(0, 6).map((c) => <td key={c} className="clip"><ReviewCell text={String(r[c] ?? '')} /></td>)}</tr>)}</tbody>
              </table></div>
            )}
          </>
        ) : (
          <textarea className="input" dir="auto" rows={8} value={pasted} onChange={(e) => setPasted(e.target.value)}
            placeholder={'One review per line\nkhana bohat mazedar tha\nThe phone stopped working after a week'} />
        )}
        <div className="row" style={{ marginTop: 14, justifyContent: 'space-between' }}>
          <div style={{ flex: 1, minWidth: 220 }}>
            {busy && <><div className="progress" aria-label="progress"><i style={{ width: `${(progress.done / Math.max(progress.total, 1)) * 100}%` }} /></div>
              <div className="small muted num" style={{ marginTop: 4 }}>{progress.done.toLocaleString()} / {progress.total.toLocaleString()} reviews</div></>}
          </div>
          <div className="row">
            {busy && <button className="btn" onClick={() => { cancel.current = true }}>Stop</button>}
            <button className="btn btn-primary" onClick={run} disabled={busy || (mode === 'csv' ? !parsed : !pasted.trim())}>{busy ? 'Analysing…' : 'Analyse all'}</button>
          </div>
        </div>
        {err && <div style={{ marginTop: 12 }}><ErrorState message={err} /></div>}
      </div>

      {results.length > 0 ? (
        <div className="stack" style={{ marginTop: 16 }}>
          <div className="kpis">
            <Kpi ink label="Reviews analysed" value={results.length.toLocaleString()} />
            <Kpi label="Positive" value={counts.pos.toLocaleString()} hint={`${((counts.pos / results.length) * 100).toFixed(1)}%`} />
            <Kpi label="Neutral" value={counts.neu.toLocaleString()} hint={`${((counts.neu / results.length) * 100).toFixed(1)}%`} />
            <Kpi label="Negative" value={counts.neg.toLocaleString()} hint={`${((counts.neg / results.length) * 100).toFixed(1)}%`} />
            <Kpi label="Low confidence" value={counts.low.toLocaleString()} hint="read these yourself" />
          </div>
          <div className="card">
            <div className="card-head">
              <h3>Results</h3>
              <div className="row">
                <Link className="btn btn-sm" to="/app/analytics">Open analytics</Link>
                <button className="btn btn-sm" onClick={() => dl('csv')}>Download CSV</button>
                <button className="btn btn-sm" onClick={() => dl('html')}>Download report</button>
                <button className="btn btn-ghost btn-sm" onClick={() => clear('bulk')}>Clear</button>
              </div>
            </div>
            <div className="row" style={{ marginBottom: 12 }}>
              <input className="input" style={{ maxWidth: 260 }} placeholder="Search text" value={filter.q} onChange={(e) => { setFilter({ ...filter, q: e.target.value }); setPage(0) }} />
              <select className="input" style={{ maxWidth: 170 }} value={filter.sentiment} onChange={(e) => { setFilter({ ...filter, sentiment: e.target.value }); setPage(0) }}>
                <option value="">All sentiments</option><option value="positive">Positive</option><option value="neutral">Neutral</option><option value="negative">Negative</option></select>
              <select className="input" style={{ maxWidth: 230 }} value={filter.language} onChange={(e) => { setFilter({ ...filter, language: e.target.value }); setPage(0) }}>
                <option value="">All languages</option>{langs.map((l) => <option key={l} value={l}>{LANG_LABEL[l]}</option>)}</select>
              <label className="row small" style={{ gap: 6 }}><input type="checkbox" checked={filter.low} onChange={(e) => { setFilter({ ...filter, low: e.target.checked }); setPage(0) }} />Low confidence only</label>
              <span className="small muted num">{view.length.toLocaleString()} shown</span>
            </div>
            <div className="table-wrap"><table className="tbl">
              <thead><tr><th className="n">#</th><th>Review</th><th>Language</th><th>Sentiment</th><th className="n">Confidence</th></tr></thead>
              <tbody>{view.slice(page * PER, page * PER + PER).map((r) => (
                <tr key={r.id}><td className="n">{r.index + 1}</td><td className="clip"><ReviewCell text={r.text} /></td><td>{LANG_LABEL[r.language.language]}</td>
                  <td><SentBadge s={r.sentiment} />{r.low_confidence && <span className="badge b-saf" style={{ marginLeft: 6 }}>Low</span>}</td>
                  <td className="n">{r.confidence == null ? '—' : `${Math.round(r.confidence * 100)}%`}</td></tr>))}</tbody>
            </table></div>
            {view.length > PER && (
              <div className="row" style={{ justifyContent: 'flex-end', marginTop: 10 }}>
                <button className="btn btn-sm" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button>
                <span className="small muted num">Page {page + 1} of {Math.ceil(view.length / PER)}</span>
                <button className="btn btn-sm" disabled={(page + 1) * PER >= view.length} onClick={() => setPage(page + 1)}>Next</button>
              </div>
            )}
          </div>
        </div>
      ) : !busy && <div style={{ marginTop: 16 }}><Empty title="No bulk results yet"><p>Upload a file or paste reviews, then press <b>Analyse all</b>.</p></Empty></div>}
    </>
  )
}
