import { Suspense, useEffect } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'
import { BRAND } from '../brand'
import { Loading, Logo } from '../components/ui'
import { useSession } from '../state/Session'

const I = {
  home: 'M3 10.5 12 3l9 7.5V20a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z',
  one: 'M4 5h16v11H8l-4 4zM8 9h8M8 12h5',
  many: 'M3 6h13v9H3zM8 18h13V9',
  chart: 'M4 20V10M10 20V4M16 20v-7M22 20H2',
  globe: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM3 12h18M12 3c3 3 3 15 0 18M12 3c-3 3-3 15 0 18',
  gauge: 'M4 18a8 8 0 1 1 16 0M12 18l4-6',
  doc: 'M6 3h9l4 4v14H6zM14 3v5h5M9 13h7M9 17h7',
}
const TABS: [string, string, keyof typeof I][] = [
  ['/app', 'Overview', 'home'], ['/app/analyze', 'Review Analyzer', 'one'], ['/app/bulk', 'Bulk Analysis', 'many'],
  ['/app/analytics', 'Review Analytics', 'chart'], ['/app/languages', 'Language Insights', 'globe'],
  ['/app/model', 'Model Performance', 'gauge'], ['/app/reports', 'Reports', 'doc'],
]
const Icon = ({ d }: { d: string }) => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={d} /></svg>
)

export default function Workspace() {
  const { entries } = useSession()
  const { pathname } = useLocation()
  useEffect(() => { window.scrollTo(0, 0) }, [pathname])
  return (
    <div className="ws">
      <a href="#main" className="sr-only">Skip to content</a>
      <aside className="ws-side">
        <Link to="/" className="brand" aria-label={`${BRAND.name} home`}><Logo size={32} /><b>{BRAND.name}</b></Link>
        <nav className="ws-nav" aria-label="Workspace">
          {TABS.map(([to, label, ic]) => <NavLink key={to} to={to} end={to === '/app'}><Icon d={I[ic]} /><span>{label}</span></NavLink>)}
        </nav>
        <div className="ws-session">
          <span className="label">This session</span>
          <b className="num">{entries.length.toLocaleString()}</b>
          <small>reviews analysed, kept in this tab only</small>
        </div>
        <p className="ws-credit">Built by {BRAND.author}</p>
      </aside>
      <div className="ws-body">
        <main id="main" className="ws-main">
          <Suspense fallback={<Loading />}><Outlet /></Suspense>
        </main>
        <footer className="ws-foot">
          Results come from a fine-tuned model and can be wrong; check low-confidence results yourself. Reviews are processed in memory and not stored on the server.
        </footer>
      </div>
    </div>
  )
}
