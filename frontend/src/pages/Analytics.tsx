import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { LANG_COLOR, SENT_COLOR, SENT_LABEL } from '../brand'
import Plot from '../components/Plot'
import { Empty, ErrorState, Info, Kpi, Loading, PageHead, useAsync } from '../components/ui'
import { api, toItem } from '../lib/api'
import { useSession } from '../state/Session'

/* eslint-disable @typescript-eslint/no-explicit-any */
const S = ['positive', 'neutral', 'negative'] as const

export default function Analytics() {
  const { entries } = useSession()
  const [scope, setScope] = useState<'all' | 'bulk' | 'single'>('all')
  const items = useMemo(() => entries.filter((e) => scope === 'all' || e.origin === scope).map(toItem), [entries, scope])
  const key = `${scope}:${items.length}:${entries[0]?.id || ''}`
  const q = useAsync(() => (items.length ? api.summary(items) : Promise.resolve(null)), [key])
  const d: any = q.data

  return (
    <>
      <PageHead eyebrow="Session analytics" title="Review Analytics"
        lead="Statistics over the reviews you analysed in this browser tab: sentiment, languages, confidence and the words that set each sentiment apart."
        actions={<div className="segmented">{(['all', 'bulk', 'single'] as const).map((s) =>
          <button key={s} className={scope === s ? 'on' : ''} onClick={() => setScope(s)}>{s === 'all' ? 'All' : s === 'bulk' ? 'Bulk' : 'Single'}</button>)}</div>} />
      {!items.length ? (
        <Empty title="No reviews analysed yet"><p>Analyse reviews on the <Link to="/app/analyze">Review Analyzer</Link> or upload a file in <Link to="/app/bulk">Bulk Analysis</Link>, then come back.</p></Empty>
      ) : q.loading && !d ? <Loading label="Computing statistics…" /> : q.error ? <ErrorState message={q.error} onRetry={q.retry} /> : d && (
        <div className="stack">
          <div className="kpis">
            <Kpi ink label="Reviews" value={d.scored.toLocaleString()} hint={d.skipped ? `${d.skipped} skipped (no text)` : 'all scored'} />
            <Kpi label="Net sentiment" value={d.net_sentiment == null ? '—' : `${d.net_sentiment > 0 ? '+' : ''}${d.net_sentiment}`}
              info="Positive share minus negative share, in percentage points. Ranges from −100 to +100." hint="positive % − negative %" />
            <Kpi label="Positive" value={`${d.sentiment_pct.positive}%`} hint={`${d.sentiment.positive.toLocaleString()} reviews`} />
            <Kpi label="Negative" value={`${d.sentiment_pct.negative}%`} hint={`${d.sentiment.negative.toLocaleString()} reviews`} />
            <Kpi label="Mean confidence" value={d.confidence.mean == null ? '—' : `${Math.round(d.confidence.mean * 100)}%`} hint={`${d.confidence.low_confidence} low-confidence`} />
          </div>

          {d.business_insights.length > 0 && (
            <div className="card"><div className="card-head"><h3>What the numbers say</h3></div>
              <ul style={{ margin: 0, paddingLeft: 20, display: 'grid', gap: 6 }}>{d.business_insights.map((b: string, i: number) => <li key={i}>{b}</li>)}</ul></div>
          )}

          <div className="grid g2">
            <div className="card"><div className="card-head"><h3>Sentiment distribution</h3></div>
              <Plot label="Sentiment distribution bar chart" height={260} data={[{ type: 'bar', x: S.map((s) => SENT_LABEL[s]), y: S.map((s) => d.sentiment[s]),
                marker: { color: S.map((s) => SENT_COLOR[s]) }, text: S.map((s) => `${d.sentiment_pct[s]}%`), textposition: 'outside', cliponaxis: false,
                hovertemplate: '%{x}: %{y:,} reviews<extra></extra>' }]} layout={{ yaxis: { title: { text: 'Reviews' } } }} /></div>
            <div className="card"><div className="card-head"><h3>Language mix</h3></div>
              <Plot label="Language distribution bar chart" height={260} data={[{ type: 'bar', orientation: 'h', y: d.language.map((l: any) => l.label), x: d.language.map((l: any) => l.count),
                marker: { color: d.language.map((l: any) => LANG_COLOR[l.language]) }, hovertemplate: '%{y}: %{x:,}<extra></extra>' }]}
                layout={{ yaxis: { autorange: 'reversed' }, margin: { l: 10, r: 16, t: 10, b: 40 } }} /></div>
          </div>

          <div className="card"><div className="card-head"><div><h3>Sentiment by language</h3><p className="sub">Share of each sentiment within each detected language.</p></div></div>
            <Plot label="Sentiment share by language stacked bar chart" height={Math.max(220, 60 + d.sentiment_by_language.length * 46)}
              data={S.map((s) => ({ type: 'bar', orientation: 'h', name: SENT_LABEL[s], marker: { color: SENT_COLOR[s] },
                y: d.sentiment_by_language.map((r: any) => `${r.label} (${(r.positive + r.neutral + r.negative).toLocaleString()})`),
                x: d.sentiment_by_language.map((r: any) => (r[s] / Math.max(1, r.positive + r.neutral + r.negative)) * 100),
                hovertemplate: `%{y}<br>${SENT_LABEL[s]}: %{x:.1f}%<extra></extra>` }))}
              layout={{ barmode: 'stack', xaxis: { range: [0, 100], ticksuffix: '%' }, yaxis: { autorange: 'reversed' }, margin: { l: 10, r: 16, t: 10, b: 40 } }} /></div>

          <div className="grid g2">
            <div className="card"><div className="card-head"><h3>Confidence<Info text="How sure the model was. Reviews under 55% are flagged as low confidence." /></h3></div>
              <Plot label="Confidence histogram" height={240} data={[{ type: 'bar', x: d.confidence.histogram.map((h: any) => h.range), y: d.confidence.histogram.map((h: any) => h.count),
                marker: { color: '#3B36A8' }, hovertemplate: '%{x}: %{y:,}<extra></extra>' }]} layout={{ yaxis: { title: { text: 'Reviews' } } }} /></div>
            <div className="card"><div className="card-head"><h3>Review length</h3>{d.text.length && <span className="small muted">median {d.text.length.median} words</span>}</div>
              {d.text.length ? <Plot label="Review length histogram" height={240} data={[{ type: 'bar', x: d.text.length.buckets.map((h: any) => h.range), y: d.text.length.buckets.map((h: any) => h.count),
                marker: { color: '#E8A33D' }, hovertemplate: '%{x} words: %{y:,}<extra></extra>' }]} layout={{ xaxis: { title: { text: 'Words' } } }} /> : <Empty title="No data" />}</div>
          </div>

          <div className="card">
            <div className="card-head"><div><h3>Words that set each sentiment apart<Info text="Ranked by weighted log-odds: words used much more in this sentiment than in the others, not just the most common words." /></h3><p className="sub">{d.text.note}</p></div></div>
            <div className="grid g3">{S.map((s) => (
              <div key={s}>
                <div className="row" style={{ marginBottom: 8 }}><span className={`badge ${s === 'positive' ? 'b-pos' : s === 'negative' ? 'b-neg' : 'b-neu'}`}>{SENT_LABEL[s]}</span></div>
                {(d.text.words[s] || []).length === 0 ? <p className="small muted">Not enough reviews.</p> : (
                  <div className="row" style={{ gap: 6 }}>{d.text.words[s].slice(0, 14).map((w: any) => <span key={w.term} className="badge b-line" dir="auto" title={`${w.count} reviews`}>{w.term} <span className="muted num">{w.count}</span></span>)}</div>)}
                {(d.text.phrases[s] || []).length > 0 && <>
                  <div className="label" style={{ margin: '12px 0 6px' }}>Phrases</div>
                  <div className="row" style={{ gap: 6 }}>{d.text.phrases[s].slice(0, 8).map((w: any) => <span key={w.term} className="badge b-ink" dir="auto">{w.term}</span>)}</div></>}
              </div>))}</div>
          </div>
        </div>
      )}
    </>
  )
}
