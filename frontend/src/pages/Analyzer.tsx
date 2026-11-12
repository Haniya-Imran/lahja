import { useState } from 'react'
import { Link } from 'react-router-dom'
import { LANG_LABEL } from '../brand'
import { Highlight, LanguageCard, Probabilities, ReviewCell, Verdict } from '../components/ResultView'
import { Empty, ErrorState, Info, Loading, PageHead, SentBadge, useAsync } from '../components/ui'
import { api, hasUrdu, type Result } from '../lib/api'
import { useSession } from '../state/Session'

const MAX = 5000

export default function Analyzer() {
  const { entries, add, remove, clear } = useSession()
  const ex = useAsync(() => api.examples(), [])
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string>()
  const [res, setRes] = useState<(Result & { elapsed_ms: number }) | null>(null)
  const [copied, setCopied] = useState(false)
  const history = entries.filter((e) => e.origin === 'single').slice(0, 30)

  const run = async () => {
    const t = text.trim()
    if (!t) { setErr('Type or paste a review first.'); return }
    setBusy(true); setErr(undefined)
    try {
      const r = await api.analyze(t, true)
      setRes(r); add([r], 'single')
    } catch (e) { setErr((e as Error).message) } finally { setBusy(false) }
  }
  const copy = async () => {
    if (!res) return
    const s = `Review: ${res.text}\nSentiment: ${res.sentiment} (${Math.round((res.confidence || 0) * 100)}%)\nLanguage: ${LANG_LABEL[res.language.language]}`
    try { await navigator.clipboard.writeText(s); setCopied(true); setTimeout(() => setCopied(false), 1500) } catch { /* clipboard blocked */ }
  }

  return (
    <>
      <PageHead eyebrow="Single review" title="Review Analyzer"
        lead="Paste one review in Urdu, Roman Urdu, English or a mix. LingoAI detects the language, cleans the text, runs the fine-tuned Transformer and shows which words drove the result." />
      <div className="grid g2" style={{ alignItems: 'start' }}>
        <div className="stack">
          <div className="card">
            <label className="field">
              <span>Review text</span>
              <textarea className={`input ${hasUrdu(text) ? 'ur' : ''}`} dir="auto" rows={7} maxLength={MAX} value={text}
                placeholder="e.g. Delivery late thi lekin product bohat acha hai" onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') run() }} />
            </label>
            <div className="row" style={{ justifyContent: 'space-between', marginTop: 10 }}>
              <span className="small muted num">{text.length.toLocaleString()} / {MAX.toLocaleString()} characters</span>
              <div className="row">
                <button className="btn btn-ghost btn-sm" onClick={() => { setText(''); setRes(null); setErr(undefined) }} disabled={!text && !res}>Clear</button>
                <button className="btn btn-primary" onClick={run} disabled={busy}>{busy ? 'Analysing…' : 'Analyse review'}</button>
              </div>
            </div>
          </div>
          <div className="card">
            <div className="card-head"><div><h3>Try an example</h3><p className="sub">{ex.data?.note || 'Short reviews written for this demo.'}</p></div></div>
            {ex.loading && <Loading />}
            {ex.error && <ErrorState message={ex.error} onRetry={ex.retry} />}
            <div className="grid g2" style={{ gap: 8 }}>
              {ex.data?.examples.map((e, i) => (
                <button key={i} className="example" onClick={() => { setText(e.text); setRes(null) }}>
                  <small>{LANG_LABEL[e.language] || e.language}</small><span dir="auto" className={hasUrdu(e.text) ? 'ur' : ''}>{e.text}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="stack" aria-live="polite">
          {busy && <div className="card"><Loading label="Running the Transformer… the first request can take a few seconds while the model loads." /></div>}
          {err && <ErrorState message={err} />}
          {!busy && !res && !err && <Empty title="No result yet"><p>Results appear here. Press <b>Analyse review</b> or Ctrl + Enter.</p></Empty>}
          {res && !busy && (
            <>
              <div className="card">
                <div className="card-head"><h3>Sentiment</h3>
                  <div className="row"><span className="small muted">{res.elapsed_ms} ms</span><button className="btn btn-sm" onClick={copy}>{copied ? 'Copied' : 'Copy result'}</button></div></div>
                <Verdict r={res} />
                {res.probabilities && <div style={{ marginTop: 16 }}><Probabilities p={res.probabilities} /></div>}
              </div>
              <div className="card"><div className="card-head"><h3>Language & script<Info text="Detected from the share of Urdu-script letters and from character patterns learned on the training reviews." /></h3></div><LanguageCard r={res} /></div>
              {res.sentiment && <div className="card"><div className="card-head"><h3>Why this result<Info text="Each word is removed in turn and the review re-scored. Green words pushed towards the predicted label; red words pushed against it." /></h3></div><Highlight r={res} /></div>}
              <div className="card"><div className="card-head"><h3>Cleaned text sent to the model</h3></div><p dir="auto" className={hasUrdu(res.normalized) ? 'ur' : ''}>{res.normalized}</p></div>
            </>
          )}
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <div className="card-head">
          <div><h3>History in this session</h3><p className="sub">Kept in this browser tab only. Bulk results are on the <Link to="/app/bulk">Bulk Analysis</Link> page.</p></div>
          {history.length > 0 && <button className="btn btn-sm" onClick={() => clear('single')}>Clear history</button>}
        </div>
        {history.length === 0 ? <Empty title="Nothing analysed yet" /> : (
          <div className="table-wrap"><table className="tbl">
            <thead><tr><th>Review</th><th>Language</th><th>Sentiment</th><th className="n">Confidence</th><th /></tr></thead>
            <tbody>{history.map((h) => (
              <tr key={h.id}>
                <td className="clip"><ReviewCell text={h.text} /></td><td>{LANG_LABEL[h.language.language]}</td>
                <td><SentBadge s={h.sentiment} />{h.low_confidence && <span className="badge b-saf" style={{ marginLeft: 6 }}>Low</span>}</td>
                <td className="n">{h.confidence == null ? '—' : `${Math.round(h.confidence * 100)}%`}</td>
                <td><div className="row" style={{ flexWrap: 'nowrap' }}>
                  <button className="btn btn-ghost btn-sm" onClick={() => { setText(h.text); setRes({ ...h, elapsed_ms: 0 }) }}>Open</button>
                  <button className="btn btn-ghost btn-sm" aria-label="Delete" onClick={() => remove(h.id)}>✕</button></div></td>
              </tr>))}</tbody>
          </table></div>
        )}
      </div>
    </>
  )
}
