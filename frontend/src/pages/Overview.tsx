import { Link } from 'react-router-dom'
import { pct } from '../brand'
import { ErrorState, Kpi, Loading, PageHead } from '../components/ui'
import { useModelInfo } from '../lib/useModel'
import { useSession } from '../state/Session'

const STAGES = [
  ['Review input', 'Type one review or upload a CSV'],
  ['Language & script', 'Urdu, Roman Urdu, English or mixed'],
  ['Normalisation', 'Unicode, Urdu letters, URLs, elongation'],
  ['Transformer', 'Fine-tuned multilingual MiniLM'],
  ['Sentiment', 'Calibrated probabilities + explanation'],
  ['Analytics', 'Trends, languages, distinctive words'],
  ['Reports', 'CSV and printable report'],
]

export default function Overview() {
  const q = useModelInfo()
  const { entries } = useSession()
  const t = q.data?.card?.served_model_test
  const base = q.data?.card?.baseline
  const pos = entries.filter((e) => e.sentiment === 'positive').length
  const neg = entries.filter((e) => e.sentiment === 'negative').length
  return (
    <>
      <PageHead eyebrow="Workspace" title="Overview"
        lead="LingoAI reads customer reviews the way people in Pakistan actually write them — in Urdu script, in Roman Urdu, in English, and in all of them at once."
        actions={<><Link className="btn btn-primary" to="/app/analyze">Analyse a review</Link><Link className="btn" to="/app/bulk">Upload a CSV</Link></>} />
      <div className="stack">
        <div className="card">
          <div className="card-head"><div><h3>How a review flows through LingoAI</h3><p className="sub">Each stage is a separate module in the code, from input to report.</p></div></div>
          <ol className="pipeline">{STAGES.map(([t, d], i) => <li key={t}><span className="num">{i + 1}</span><b>{t}</b><small>{d}</small></li>)}</ol>
        </div>
        {q.loading ? <Loading label="Loading model facts…" /> : q.error ? <ErrorState message={q.error} onRetry={q.retry} /> : t && (
          <div className="kpis">
            <Kpi ink label="Test macro-F1" value={t.macro_f1.toFixed(3)} hint={base ? `TF-IDF baseline ${base.test.macro_f1.toFixed(3)}` : undefined} info="Measured on held-out test reviews never seen in training." />
            <Kpi label="Test accuracy" value={pct(t.accuracy)} hint={`${t.n.toLocaleString()} test reviews`} />
            <Kpi label="Training reviews" value={(q.data.metadata.training?.train_rows || 0).toLocaleString()} hint={`${q.data.card.sources?.length || 8} public datasets`} />
            <Kpi label="Served model" value={`${q.data.metadata.size_mb} MB`} hint="int8 ONNX, runs on CPU" />
          </div>
        )}
        <div className="grid g3">
          <Link to="/app/analyze" className="card tile"><h3>Review Analyzer</h3><p className="small muted">One review, with language, confidence and the words that drove the result.</p></Link>
          <Link to="/app/bulk" className="card tile"><h3>Bulk Analysis</h3><p className="small muted">Upload a CSV, pick the review column, analyse up to 5,000 reviews.</p></Link>
          <Link to="/app/analytics" className="card tile"><h3>Review Analytics</h3><p className="small muted">{entries.length ? `${entries.length.toLocaleString()} reviews in this session: ${pos.toLocaleString()} positive, ${neg.toLocaleString()} negative.` : 'Charts appear once you analyse reviews.'}</p></Link>
          <Link to="/app/languages" className="card tile"><h3>Language Insights</h3><p className="small muted">Script detection, data per language and quality per language.</p></Link>
          <Link to="/app/model" className="card tile"><h3>Model Performance</h3><p className="small muted">Test metrics, confusion matrix, baseline comparison and error analysis.</p></Link>
          <Link to="/app/reports" className="card tile"><h3>Reports</h3><p className="small muted">Download results as CSV or a printable summary report.</p></Link>
        </div>
      </div>
    </>
  )
}
