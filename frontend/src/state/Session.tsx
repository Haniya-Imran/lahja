// Current-session review history. Lives in this browser tab only (sessionStorage);
// nothing is stored on the server.
import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'
import type { Result } from '../lib/api'

export interface Entry extends Result { id: string; at: number; origin: 'single' | 'bulk' }

interface SessionState {
  entries: Entry[]
  add: (rs: Result[], origin: Entry['origin']) => void
  remove: (id: string) => void
  clear: (origin?: Entry['origin']) => void
}

const Ctx = createContext<SessionState | null>(null)
const KEY = 'lahja.session'
const MAX = 20000

function load(): Entry[] {
  try { return JSON.parse(sessionStorage.getItem(KEY) || '[]') } catch { return [] }
}
function save(e: Entry[]) {
  try { sessionStorage.setItem(KEY, JSON.stringify(e.map(({ explanation, ...rest }) => (rest.origin === 'single' ? { ...rest, explanation } : rest)))) } catch { /* quota: keep in memory only */ }
}

export function SessionProvider({ children }: { children: ReactNode }) {
  const [entries, setEntries] = useState<Entry[]>(load)
  const update = useCallback((fn: (e: Entry[]) => Entry[]) => setEntries((prev) => { const n = fn(prev); save(n); return n }), [])
  const add = useCallback((rs: Result[], origin: Entry['origin']) => update((prev) => {
    const now = Date.now()
    const fresh = rs.map((r, i) => ({ ...r, id: `${now}-${i}-${Math.random().toString(36).slice(2, 7)}`, at: now, origin }))
    return [...fresh, ...prev].slice(0, MAX)
  }), [update])
  const remove = useCallback((id: string) => update((prev) => prev.filter((e) => e.id !== id)), [update])
  const clear = useCallback((origin?: Entry['origin']) => update((prev) => origin ? prev.filter((e) => e.origin !== origin) : []), [update])
  const value = useMemo(() => ({ entries, add, remove, clear }), [entries, add, remove, clear])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

// eslint-disable-next-line react-refresh/only-export-components
export function useSession() {
  const c = useContext(Ctx)
  if (!c) throw new Error('useSession outside provider')
  return c
}
