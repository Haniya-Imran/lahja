import { lazy, Suspense, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { BRAND, pct } from '../brand'
import { Logo } from '../components/ui'
import { fetchModelInfo } from '../lib/useModel'
import { api, hasUrdu, type Result } from '../lib/api'
import { SENT_COLOR, SENT_LABEL, LANG_LABEL } from '../brand'
import './landing.css'

const Hero3D = lazy(() => import('./Hero3D'))
/* eslint-disable @typescript-eslint/no-explicit-any */

function useReducedMotion() {
  const [r] = useState(() => typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches)
  return r
}

function LiveDemo() {
  const [text, setText] = useState('Delivery late thi lekin product bohat acha hai')
  const [res, setRes] = useState<Result | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string>()
  const run = async () => {
    if (!text.trim()) return
    setBusy(true); setErr(undefined)
    try { setRes(await api.analyze(text.trim(), false)) } catch (e) { setErr((e as Error).message) } finally { setBusy(false) }
  }
  return (
    <div className="l-demo">
      <textarea dir="auto" className={`input ${hasUrdu(text) ? 'ur' : ''}`} rows={3} maxLength={600} value={text} onChange={(e) => setText(e.target.value)} aria-label="Try a review" />
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <div className="row" style={{ gap: 6 }}>
          {['کھانا بہت مزیدار تھا', 'bilkul bekaar, paise zaya', 'Good phone but battery theek nahi'].map((s) =>
            <button key={s} className="btn btn-sm" dir="auto" onClick={() => { setText(s); setRes(null) }}><span className={hasUrdu(s) ? 'ur' : ''}>{s}</span></button>)}
        </div>
        <button className="btn btn-primary" onClick={run} disabled={busy}>{busy ? 'Reading…' : 'Analyse'}</button>
      </div>
      <div aria-live="polite">
        {err && <p className="small" style={{ color: '#a3293d' }}>{err}</p>}
        {res && res.sentiment && (
          <div className="l-demo-out">
            <span className="badge" style={{ background: SENT_COLOR[res.sentiment], color: '#fff' }}>{SENT_LABEL[res.sentiment]} · {Math.round((res.confidence || 0) * 100)}%</span>
            <span className="badge b-saf">{LANG_LABEL[res.language.language]}</span>
            {res.low_confidence && <span className="small muted">low confidence</span>}
            <Link to="/app/analyze" className="small">See why →</Link>
          </div>
        )}
      </div>
    </div>
  )
}

export default function Landing() {
  const reduced = useReducedMotion()
  const [card, setCard] = useState<any>(null)
  useEffect(() => { document.title = `${BRAND.name} — Multilingual Review Intelligence`; fetchModelInfo().then(setCard).catch(() => setCard(null)) }, [])
  const t = card?.card?.served_model_test
  const base = card?.card?.baseline
  return (
    <div className="landing">
      <header className="l-nav">
        <Link to="/" className="brand"><Logo size={34} /><b>{BRAND.name}</b></Link>
        <nav className="l-links" aria-label="Sections"><a href="#how">How it works</a><a href="#proof">Results</a><a href="#limits">Limits</a></nav>
        <Link to="/app" className="btn btn-primary">Open workspace</Link>
      </header>

      <section className="l-hero">
        <div className="l-hero-copy">
          <span className="eyebrow">Urdu · Roman Urdu · English · Mixed</span>
          <h1>Hear the <em>tone</em> in every review.</h1>
          <p className="l-lead">{BRAND.name} — <span className="ur">لہجہ</span>, “tone of voice” — reads customer reviews in the scripts people really use and tells you how they feel, how sure it is, and which words decided it.</p>
          <div className="row"><Link to="/app/analyze" className="btn btn-primary btn-lg">Analyse a review</Link><Link to="/app/bulk" className="btn btn-lg">Upload a CSV</Link></div>
          <p className="small muted" style={{ marginTop: 14 }}>No sign-up. Reviews are processed in memory and not stored.</p>
        </div>
        <div className="l-hero-3d">
          {reduced ? <div className="l-static" aria-hidden="true"><span className="ur">بہت زبردست</span><span>bohat acha</span><span>not worth it</span></div>
            : <Suspense fallback={<div className="l-static" />}><Hero3D /></Suspense>}
        </div>
      </section>

      <section className="l-section" id="try">
        <div className="l-split">
          <div><span className="eyebrow">Try it now</span><h2>Mixed language is not an edge case.</h2>
            <p className="l-body">“Delivery late thi lekin product bohat acha hai” switches language mid-sentence. Keyword lists break on text like this; a multilingual Transformer fine-tuned on Urdu, Roman Urdu and English reviews does not need them.</p></div>
          <LiveDemo />
        </div>
      </section>

      <section className="l-section" id="how">
        <span className="eyebrow">How it works</span><h2>From raw review to report, in seven steps.</h2>
        <div className="l-steps">
          {[
            ['Detect', 'Script share and a character n-gram model tell Urdu, Roman Urdu, English and code-switched text apart.'],
            ['Normalise', 'Unicode and Urdu letter forms unified, diacritics, links and stretched letters (“acchaaaa”) cleaned.'],
            ['Understand', 'A multilingual MiniLM Transformer, fine-tuned on about 31,000 labelled public reviews, reads the review.'],
            ['Calibrate', 'Probabilities are temperature-calibrated so “80% sure” means something; unsure results are flagged.'],
            ['Explain', 'Each word is removed in turn and the review re-scored to show which words moved the decision.'],
            ['Report', 'Session analytics, distinctive words per sentiment, CSV export and a printable report.'],
          ].map(([h, d], i) => <article key={h} className="l-step"><span className="num">0{i + 1}</span><h3>{h}</h3><p>{d}</p></article>)}
        </div>
      </section>

      <section className="l-section" id="proof">
        <span className="eyebrow">Measured, not claimed</span><h2>Tested on reviews it never saw.</h2>
        {t ? (
          <div className="l-proof">
            <div><b>{t.macro_f1.toFixed(2)}</b><span>macro-F1 on {t.n.toLocaleString()} held-out test reviews</span></div>
            <div><b>{pct(t.accuracy, 0)}</b><span>accuracy{base ? ` (TF-IDF baseline ${pct(base.test.accuracy, 0)})` : ''}</span></div>
            <div><b>{Object.keys(card.card.test_by_language).length}</b><span>language groups evaluated separately</span></div>
            <div><b>{card.metadata.size_mb} MB</b><span>compressed model, runs on a CPU</span></div>
          </div>
        ) : <p className="muted">Model metrics load from the live API.</p>}
        <Link to="/app/model" className="btn" style={{ marginTop: 18 }}>See full evaluation</Link>
      </section>

      <section className="l-section" id="limits">
        <div className="l-limits">
          <div><span className="eyebrow">Honest limits</span><h2>Where it can be wrong.</h2></div>
          <ul>
            <li>Sarcasm and irony are often misread.</li>
            <li>Very short reviews (“ok”, “hmm”) carry little signal and are usually low-confidence.</li>
            <li>Training data comes from public datasets of product reviews and social media; other domains may score lower.</li>
            <li>Mixed Urdu-script + English reviews are rare in the data, so their test numbers are less reliable.</li>
          </ul>
        </div>
      </section>

      <section className="l-cta">
        <h2>Start with one review — or five thousand.</h2>
        <div className="row" style={{ justifyContent: 'center' }}><Link to="/app/analyze" className="btn btn-saffron btn-lg">Open the analyzer</Link><Link to="/app" className="btn btn-lg">Explore the workspace</Link></div>
      </section>
      <footer className="l-foot"><span>{BRAND.name} · multilingual review intelligence</span><span>Built by {BRAND.author}</span></footer>
    </div>
  )
}
