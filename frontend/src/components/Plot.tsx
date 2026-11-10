// Thin wrapper around plotly.js (loaded lazily so the landing page stays light).
import { useEffect, useRef } from 'react'
/* eslint-disable @typescript-eslint/no-explicit-any */

let plotlyPromise: Promise<any> | null = null
const loadPlotly = () => (plotlyPromise ??= import('plotly.js-dist-min').then((m: any) => m.default ?? m))

const BASE: any = {
  font: { family: 'Inter Variable, system-ui, sans-serif', size: 12.5, color: '#3E3C63' },
  paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
  margin: { l: 48, r: 16, t: 10, b: 44 },
  hoverlabel: { bgcolor: '#1B1A3A', bordercolor: '#1B1A3A', font: { color: '#fff', family: 'Inter Variable, sans-serif' } },
  xaxis: { gridcolor: '#EEE8DD', zerolinecolor: '#D8D0C2', linecolor: '#D8D0C2', automargin: true },
  yaxis: { gridcolor: '#EEE8DD', zerolinecolor: '#D8D0C2', linecolor: '#D8D0C2', automargin: true },
  legend: { orientation: 'h', y: -0.22, x: 0 },
  bargap: 0.28,
}

export default function Plot({ data, layout = {}, height = 300, label }: { data: any[]; layout?: any; height?: number; label: string }) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    let alive = true
    const el = ref.current
    loadPlotly().then((P) => {
      if (!alive || !el) return
      const lay = { ...BASE, ...layout, height, xaxis: { ...BASE.xaxis, ...layout.xaxis }, yaxis: { ...BASE.yaxis, ...layout.yaxis } }
      P.react(el, data, lay, { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'] })
    })
    return () => { alive = false }
  }, [data, layout, height])
  useEffect(() => () => { const el = ref.current; if (el) loadPlotly().then((P) => P.purge(el)) }, [])
  return <div ref={ref} role="img" aria-label={label} style={{ width: '100%', minHeight: height }} />
}
