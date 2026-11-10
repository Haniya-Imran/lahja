import { LANG_LABEL, SENT_COLOR, SENT_LABEL, pct } from '../brand'
import type { Result } from '../lib/api'
import { hasUrdu } from '../lib/api'
import { Info, ReviewText } from './ui'

const ORDER = ['positive', 'neutral', 'negative'] as const

export function Probabilities({ p }: { p: Record<string, number> }) {
  return (
    <div className="stack" style={{ gap: 8 }}>
      {ORDER.map((k) => (
        <div className="prob" key={k}>
          <span>{SENT_LABEL[k]}</span>
          <span className="bar"><i style={{ width: `${(p[k] * 100).toFixed(1)}%`, background: SENT_COLOR[k] }} /></span>
          <b className="num" style={{ textAlign: 'right' }}>{pct(p[k])}</b>
        </div>
      ))}
    </div>
  )
}

/** Word-level highlight: green pushed towards the predicted label, red pushed away. */
export function Highlight({ r }: { r: Result }) {
  const ex = r.explanation
  if (!ex?.available || !ex.tokens) return <p className="muted small">{ex?.reason || 'No explanation for this review.'}</p>
  const max = Math.max(1e-6, ...ex.tokens.map((t) => Math.abs(t.score)))
  const text = r.normalized
  const parts: React.ReactNode[] = []
  let pos = 0
  ex.tokens.forEach((t, i) => {
    if (t.start > pos) parts.push(text.slice(pos, t.start))
    const a = Math.min(1, Math.abs(t.score) / max)
    const bg = t.score >= 0 ? `rgba(31,157,122,${(0.12 + a * 0.55).toFixed(2)})` : `rgba(216,75,97,${(0.12 + a * 0.5).toFixed(2)})`
    parts.push(<mark key={i} className="tok" style={{ background: Math.abs(t.score) / max < 0.08 ? 'transparent' : bg, color: 'inherit' }}
      title={`${t.score >= 0 ? '+' : ''}${(t.score * 100).toFixed(1)} pts`}>{text.slice(t.start, t.end)}</mark>)
    pos = t.end
  })
  if (pos < text.length) parts.push(text.slice(pos))
  return (
    <div className="stack" style={{ gap: 10 }}>
      <p dir="auto" className={hasUrdu(text) ? 'ur' : ''} style={{ fontSize: 17, lineHeight: hasUrdu(text) ? 2.4 : 1.9 }}>{parts}</p>
      {ex.top && ex.top.length > 0 && (
        <div className="row small">
          <span className="muted">Most influential:</span>
          {ex.top.map((t) => <span key={t.word} className={`badge ${t.score >= 0 ? 'b-pos' : 'b-neg'}`} dir="auto">{t.word} {t.score >= 0 ? '↑' : '↓'}</span>)}
        </div>
      )}
      <p className="muted small">{ex.note}</p>
    </div>
  )
}

export function LanguageCard({ r }: { r: Result }) {
  const l = r.language
  return (
    <div className="stack" style={{ gap: 8 }}>
      <div className="row"><span className="badge b-saf">{LANG_LABEL[l.language] || l.label}</span><span className="muted small">{l.script} script · {pct(l.confidence, 0)} sure</span></div>
      <div className="small muted num">Urdu-script characters: {pct(l.arabic_share, 0)} · Roman Urdu words: {l.roman_urdu_words} · English words: {l.english_words}</div>
      {l.note && <p className="small">{l.note}</p>}
    </div>
  )
}

export function Verdict({ r }: { r: Result }) {
  if (!r.sentiment || !r.probabilities) return <div className="note">{r.skipped || 'This text could not be analysed.'}</div>
  return (
    <div className={`verdict ${r.sentiment}`}>
      <div className="dial" style={{ background: SENT_COLOR[r.sentiment] }}>{Math.round((r.confidence || 0) * 100)}%</div>
      <div>
        <div style={{ fontFamily: 'var(--font-display)', fontSize: 28, lineHeight: 1.1 }}>{SENT_LABEL[r.sentiment]}</div>
        <div className="small muted">
          Confidence {pct(r.confidence)}<Info text="Calibrated probability of the predicted label. Calibration was fitted on the validation set so 80% confident predictions are right about 80% of the time on that data." />
          {r.windows && r.windows > 1 ? ` · long review read in ${r.windows} overlapping parts` : ''}
        </div>
        {r.low_confidence && <div className="small" style={{ color: '#7d4d06', fontWeight: 650, marginTop: 4 }}>Low confidence — the model is unsure, please read this review yourself.</div>}
      </div>
    </div>
  )
}

export function ReviewCell({ text }: { text: string }) {
  return <ReviewText text={text.length > 240 ? text.slice(0, 240) + '…' : text} />
}
