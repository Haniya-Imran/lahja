import { useMemo } from 'react'
import { LANG_COLOR, LANG_LABEL, SENT_COLOR, SENT_LABEL, pct } from '../brand'
import Plot from '../components/Plot'
import { ErrorState, Info, Loading, PageHead } from '../components/ui'
import { useModelInfo } from '../lib/useModel'
import { useSession } from '../state/Session'

/* eslint-disable @typescript-eslint/no-explicit-any */
const S = ['positive', 'neutral', 'negative'] as const
const SRC_LANG: Record<string, string> = { urdu: 'Urdu script', roman_urdu: 'Roman Urdu', english: 'English' }

export default function Languages() {
  const q = useModelInfo()
  const { entries } = useSession()
  const session = useMemo(() => {
    const m: Record<string, { n: number; low: number; conf: number }> = {}
    entries.forEach((e) => { const k = e.language.language; m[k] ??= { n: 0, low: 0, conf: 0 }; m[k].n++; m[k].low += +e.low_confidence; m[k].conf += e.confidence || 0 })
    return Object.entries(m).sort((a, b) => b[1].n - a[1].n)
  }, [entries])
  if (q.loading) return <Loading label="Loading language statistics…" />
  if (q.error) return <ErrorState message={q.error} onRetry={q.retry} />
  const c = q.data.card
  const det = Object.entries(c.data.detected_language as Record<string, number>).sort((a, b) => b[1] - a[1])
  const tl = Object.entries(c.data.test_language_label as Record<string, Record<string, number>>).filter(([, v]) => Object.values(v).reduce((a, b) => a + b, 0) >= 20)
  const byLang = c.test_by_language as Record<string, any>

  return (
    <>
      <PageHead eyebrow="Language & script intelligence" title="Language Insights"
        lead="How LingoAI tells Urdu script, Roman Urdu, English and code-switched text apart, what the training data looks like per language, and how well the model does on each." />
      <div className="stack">
        <div className="grid g3">
          {[
            ['Urdu script', 'If at least 85% of letters are Urdu/Arabic script, the review is Urdu. Between 15% and 85% it is mixed Urdu + English.'],
            ['Roman Urdu vs English', 'Both use Latin letters, so LingoAI uses a character n-gram model and word lists learned only from the training reviews, plus a public Roman Urdu dictionary.'],
            ['Code-switching', 'A Latin-script review is mixed when it has enough known words of both languages and the smaller one is at least a quarter of them. Very short or unknown text is marked uncertain.'],
          ].map(([t, d]) => <div className="card" key={t}><h3 style={{ fontSize: 16, marginBottom: 6 }}>{t}</h3><p className="small muted">{d}</p></div>)}
        </div>

        <div className="grid g2">
          <div className="card"><div className="card-head"><div><h3>Training data by detected language</h3><p className="sub">{c.data.final_rows.toLocaleString()} reviews after cleaning and de-duplication.</p></div></div>
            <Plot label="Reviews per detected language" height={280} data={[{ type: 'bar', orientation: 'h', y: det.map(([k]) => LANG_LABEL[k]), x: det.map(([, v]) => v),
              marker: { color: det.map(([k]) => LANG_COLOR[k]) }, hovertemplate: '%{y}: %{x:,}<extra></extra>' }]} layout={{ yaxis: { autorange: 'reversed' }, xaxis: { type: 'log', title: { text: 'Reviews (log scale)' } }, margin: { l: 10, r: 16, t: 10, b: 44 } }} /></div>
          <div className="card"><div className="card-head"><h3>Label balance in the test split</h3></div>
            <Plot label="Sentiment share per language in the test split" height={280} data={S.map((s) => ({ type: 'bar', orientation: 'h', name: SENT_LABEL[s], marker: { color: SENT_COLOR[s] },
              y: tl.map(([k, v]) => `${LANG_LABEL[k]} (${Object.values(v).reduce((a, b) => a + b, 0)})`), x: tl.map(([, v]) => ((v[s] || 0) / Object.values(v).reduce((a, b) => a + b, 0)) * 100),
              hovertemplate: `%{y}<br>${SENT_LABEL[s]}: %{x:.1f}%<extra></extra>` }))} layout={{ barmode: 'stack', xaxis: { range: [0, 100], ticksuffix: '%' }, yaxis: { autorange: 'reversed' }, margin: { l: 10, r: 16, t: 10, b: 40 } }} /></div>
        </div>

        <div className="card">
          <div className="card-head"><div><h3>Model quality per language<Info text="Measured on the held-out test split. Fewer than 30 test reviews means the number is not reliable." /></h3></div></div>
          <div className="kpis">{Object.entries(byLang).sort((a, b) => b[1].n - a[1].n).map(([k, v]) => (
            <div className="kpi" key={k}><span>{LANG_LABEL[k]}</span><b>{v.macro_f1.toFixed(2)}</b><small>macro-F1 · {v.n} test reviews{v.reliable ? '' : ' · not reliable'}</small></div>))}</div>
        </div>

        <div className="card">
          <div className="card-head"><div><h3>Your session</h3><p className="sub">Languages detected in the reviews you analysed in this tab.</p></div></div>
          {session.length === 0 ? <p className="small muted">Nothing analysed yet.</p> : (
            <div className="table-wrap"><table className="tbl"><thead><tr><th>Language</th><th className="n">Reviews</th><th className="n">Mean confidence</th><th className="n">Low confidence</th></tr></thead>
              <tbody>{session.map(([k, v]) => <tr key={k}><td><span className="badge" style={{ background: LANG_COLOR[k] + '22', color: 'var(--ink)' }}>{LANG_LABEL[k]}</span></td><td className="n">{v.n.toLocaleString()}</td><td className="n">{pct(v.conf / v.n, 0)}</td><td className="n">{v.low}</td></tr>)}</tbody></table></div>)}
        </div>

        <div className="card">
          <div className="card-head"><div><h3>Data sources</h3><p className="sub">Public datasets downloaded from their original repositories at fixed versions. Labels were mapped to positive / neutral / negative.</p></div></div>
          <div className="table-wrap"><table className="tbl"><thead><tr><th>Dataset</th><th>Language</th><th className="n">Rows loaded</th><th>Labels</th><th>License</th></tr></thead>
            <tbody>{(c.sources || []).map((s: any) => <tr key={s.key}><td><a href={s.homepage} target="_blank" rel="noreferrer">{s.name}</a></td><td>{SRC_LANG[s.language] || s.language}</td>
              <td className="n">{(c.data.loaded_by_source[s.key] || 0).toLocaleString()}</td><td className="small">{s.labels}</td><td className="small">{s.license}</td></tr>)}</tbody></table></div>
          <p className="small muted" style={{ marginTop: 10 }}>No public, labelled set of code-switched reviews was large enough to train on separately; mixed-language reviews come from inside these datasets and are found by the language detector.</p>
        </div>
      </div>
    </>
  )
}
