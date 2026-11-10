import { Suspense, useEffect } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'
import { BRAND } from '../brand'
import { Loading, Logo } from '../components/ui'
import { useSession } from '../state/Session'

const TABS = [
  ['/app', 'Overview'], ['/app/analyze', 'Review Analyzer'], ['/app/bulk', 'Bulk Analysis'], ['/app/analytics', 'Review Analytics'],
  ['/app/languages', 'Language Insights'], ['/app/model', 'Model Performance'], ['/app/reports', 'Reports'],
] as const

export default function Workspace() {
  const { entries } = useSession()
  const { pathname } = useLocation()
  useEffect(() => { window.scrollTo(0, 0) }, [pathname])
  return (
    <>
      <a href="#main" className="sr-only">Skip to content</a>
      <div className="ws-top">
        <div className="ws-top-in">
          <Link to="/" className="brand" aria-label={`${BRAND.name} home`}><Logo size={32} /><b>{BRAND.name}</b></Link>
          <nav className="tabs" aria-label="Workspace">
            {TABS.map(([to, label]) => <NavLink key={to} to={to} end={to === '/app'}>{label}</NavLink>)}
          </nav>
          <span className="badge b-ink num" title="Reviews analysed in this browser tab">{entries.length.toLocaleString()} in session</span>
        </div>
      </div>
      <main id="main" className="ws-main">
        <Suspense fallback={<Loading />}><Outlet /></Suspense>
      </main>
      <footer className="ws-foot">
        {BRAND.name} — multilingual review sentiment. Results come from a fine-tuned model and can be wrong; check low-confidence results yourself.
        Reviews are processed in memory and not stored on the server. Built by {BRAND.author}.
      </footer>
    </>
  )
}
