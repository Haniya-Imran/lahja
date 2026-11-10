import { lazy, Suspense } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { Loading } from './components/ui'
import { SessionProvider } from './state/Session'
import Workspace from './pages/Workspace'

const Landing = lazy(() => import('./landing/Landing'))
const Overview = lazy(() => import('./pages/Overview'))
const Analyzer = lazy(() => import('./pages/Analyzer'))
const Bulk = lazy(() => import('./pages/Bulk'))
const Analytics = lazy(() => import('./pages/Analytics'))
const Languages = lazy(() => import('./pages/Languages'))
const Performance = lazy(() => import('./pages/Performance'))
const Reports = lazy(() => import('./pages/Reports'))

export default function App() {
  return (
    <SessionProvider>
      <BrowserRouter>
        <Suspense fallback={<div style={{ padding: 40 }}><Loading /></div>}>
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/app" element={<Workspace />}>
              <Route index element={<Overview />} />
              <Route path="analyze" element={<Analyzer />} />
              <Route path="bulk" element={<Bulk />} />
              <Route path="analytics" element={<Analytics />} />
              <Route path="languages" element={<Languages />} />
              <Route path="model" element={<Performance />} />
              <Route path="reports" element={<Reports />} />
            </Route>
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </BrowserRouter>
    </SessionProvider>
  )
}
