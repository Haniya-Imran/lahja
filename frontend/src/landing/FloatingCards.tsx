// 3D hero: real-looking review cards float in a perspective scene. Each card's sentiment stamp is the
// live model's prediction (fetched from the API), so the hero shows real output, not a mock-up.
import { useEffect, useRef, useState } from 'react'
import { LANG_LABEL, SENT_LABEL } from '../brand'
import { api, hasUrdu, type Result } from '../lib/api'

const CARDS = [
  { who: 'A', text: 'Delivery late thi lekin product bohat acha hai', x: 0, y: 0, z: 50, r: -6 },
  { who: 'S', text: 'یہ فون بہت جلدی گرم ہو جاتا ہے، مایوس کن', x: 47, y: 11, z: -30, r: 5 },
  { who: 'M', text: 'Battery life is amazing, easily lasts two days', x: 3, y: 47, z: 100, r: 3 },
  { who: 'H', text: 'bilkul bekaar cheez, paise zaya ho gaye', x: 49, y: 57, z: 10, r: -4 },
]

export default function FloatingCards() {
  const scene = useRef<HTMLDivElement>(null)
  const [res, setRes] = useState<(Result | null)[]>(CARDS.map(() => null))
  const [shown, setShown] = useState(0)

  useEffect(() => {
    let alive = true
    api.batch(CARDS.map((c) => c.text)).then((r) => alive && setRes(r.results)).catch(() => { /* cards still render without stamps */ })
    return () => { alive = false }
  }, [])
  // Stamps land one after another once results arrive.
  useEffect(() => {
    if (!res[0] || shown >= CARDS.length) return
    const t = setTimeout(() => setShown((s) => s + 1), shown === 0 ? 300 : 650)
    return () => clearTimeout(t)
  }, [res, shown])
  // Gentle parallax tilt following the pointer.
  useEffect(() => {
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return
    let raf = 0, tx = 0, ty = 0, cx = 0, cy = 0
    const move = (e: PointerEvent) => { tx = (e.clientX / window.innerWidth - 0.5) * 2; ty = (e.clientY / window.innerHeight - 0.5) * 2 }
    const loop = () => {
      cx += (tx - cx) * 0.06; cy += (ty - cy) * 0.06
      if (scene.current) scene.current.style.transform = `rotateX(${10 - cy * 6}deg) rotateY(${-16 + cx * 10}deg)`
      raf = requestAnimationFrame(loop)
    }
    window.addEventListener('pointermove', move)
    raf = requestAnimationFrame(loop)
    return () => { window.removeEventListener('pointermove', move); cancelAnimationFrame(raf) }
  }, [])

  return (
    <div className="fc-stage" aria-label="Example reviews analysed live by the model">
      <div className="fc-scene" ref={scene}>
        <div className="fc-core" aria-hidden="true"><span /><span /><span /></div>
        {CARDS.map((c, i) => {
          const r = res[i]
          const on = r && r.sentiment && i < shown
          return (
            <article key={i} className="fc-card" style={{ ['--x' as string]: `${c.x}%`, ['--y' as string]: `${c.y}%`, ['--z' as string]: `${c.z}px`, ['--r' as string]: `${c.r}deg`, animationDelay: `${i * -1.7}s` }}>
              <header>
                <span className="fc-av">{c.who}</span>
                <span className="fc-lang">{r ? LANG_LABEL[r.language.language] : '…'}</span>
              </header>
              <p dir="auto" className={hasUrdu(c.text) ? 'ur' : ''}>{c.text}</p>
              <footer>
                {on ? (
                  <>
                    <span className={`fc-stamp ${r!.sentiment}`}>{SENT_LABEL[r!.sentiment!]}</span>
                    <span className="fc-conf"><i style={{ width: `${Math.round((r!.confidence || 0) * 100)}%` }} /></span>
                    <b className="num">{Math.round((r!.confidence || 0) * 100)}%</b>
                  </>
                ) : <span className="fc-wait">reading…</span>}
              </footer>
            </article>
          )
        })}
      </div>
      <p className="fc-cap">Live: these cards are scored by the real model when the page opens.</p>
    </div>
  )
}
