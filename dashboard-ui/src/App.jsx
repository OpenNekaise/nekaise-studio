import React, { useEffect, useMemo, useState } from 'react'

/* ---------- data hooks ---------- */
function usePoll(url, ms, enabled = true) {
  const [data, setData] = useState(null)
  useEffect(() => {
    if (!enabled) return
    let alive = true
    const tick = async () => {
      try { const r = await fetch(url); const j = await r.json(); if (alive) setData(j) } catch { /* keep last */ }
    }
    tick()
    const id = setInterval(tick, ms)
    return () => { alive = false; clearInterval(id) }
  }, [url, ms, enabled])
  return data
}

function useClock(ms = 1000) {
  const [, set] = useState(0)
  useEffect(() => { const id = setInterval(() => set((n) => n + 1), ms); return () => clearInterval(id) }, [ms])
}

/* ---------- format ---------- */
const fmtDur = (s) => {
  if (s == null || isNaN(s)) return '—'
  s = Math.max(0, Math.floor(s))
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = s % 60
  return (h ? `${h}h ` : '') + `${m}m ${String(ss).padStart(2, '0')}s`
}
const fmtNum = (x, d = 3) => {
  if (x == null || isNaN(x)) return '—'
  const a = Math.abs(x)
  if (a >= 1000) return x.toFixed(0)
  if (a >= 1) return x.toFixed(d)
  if (a === 0) return '0'
  return x.toPrecision(3)
}
const clock = (t) => t ? new Date(t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''
const lowerBetter = (metric) => /ppl|perplex|loss/i.test(metric || '')

/* ---------- minimal SVG line chart ---------- */
function LineChart({ points, color = '#6f9183', height = 240 }) {
  if (!points || points.length === 0) return <div className="chart-empty">waiting for the first logged step…</div>
  const W = 760, H = height, pad = { l: 46, r: 18, t: 16, b: 26 }
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y)
  let xmin = Math.min(...xs), xmax = Math.max(...xs)
  let ymin = Math.min(...ys), ymax = Math.max(...ys)
  if (xmin === xmax) { xmin -= 0.5; xmax += 0.5 }
  if (ymin === ymax) { const e = Math.abs(ymin) * 0.1 || 0.1; ymin -= e; ymax += e }
  const padY = (ymax - ymin) * 0.08; ymin -= padY; ymax += padY
  const sx = (x) => pad.l + (W - pad.l - pad.r) * (x - xmin) / (xmax - xmin)
  const sy = (y) => pad.t + (H - pad.t - pad.b) * (1 - (y - ymin) / (ymax - ymin))
  const path = points.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ')
  const area = `${path} L${sx(xmax).toFixed(1)},${sy(ymin).toFixed(1)} L${sx(xmin).toFixed(1)},${sy(ymin).toFixed(1)} Z`
  const grids = [0, 0.25, 0.5, 0.75, 1].map((f) => ymin + (ymax - ymin) * f)
  const last = points[points.length - 1]
  const gid = `g${color.replace('#', '')}`
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.16" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      {grids.map((g, i) => (
        <g key={i}>
          <line x1={pad.l} x2={W - pad.r} y1={sy(g)} y2={sy(g)} stroke="#efebe2" strokeWidth="1" />
          <text x={pad.l - 8} y={sy(g) + 3.5} textAnchor="end" fontSize="10.5" fill="#a39d92">{fmtNum(g, 2)}</text>
        </g>
      ))}
      <path d={area} fill={`url(#${gid})`} />
      <path d={path} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={sx(last.x)} cy={sy(last.y)} r="3.6" fill={color} />
      <circle cx={sx(last.x)} cy={sy(last.y)} r="7" fill={color} opacity="0.16" />
      <text x={W - pad.r} y={pad.t + 2} textAnchor="end" fontSize="11" fill={color} fontWeight="600">{fmtNum(last.y)}</text>
    </svg>
  )
}

/* ---------- pieces ---------- */
function Sidebar({ runs, sel, onSelect }) {
  return (
    <aside className="side">
      <div className="brand">
        <span className="dot" /><h1>nekaise</h1><span>monitor</span>
      </div>
      <div className="label">Runs</div>
      <div className="runlist">
        {(runs || []).map((r) => {
          const active = sel && r.run_id === sel.run_id && r.exp === sel.exp
          return (
            <div key={`${r.exp}/${r.run_id}`} className={`runitem${active ? ' active' : ''}`} onClick={() => onSelect(r)}>
              <div className="top">
                <span className={`sdot ${r.status}`} />
                <span className="rid">{r.run_id}</span>
              </div>
              <div className="sub">{r.metric}{r.status === 'done' && r.after != null ? ` · ${fmtNum(r.after)}` : ''}</div>
              <div className="exp">{r.exp}</div>
            </div>
          )
        })}
        {(!runs || runs.length === 0) && <div className="exp" style={{ padding: '8px 12px' }}>no runs yet</div>}
      </div>
    </aside>
  )
}

function Tile({ k, v, u }) {
  return <div className="tile"><div className="k">{k}</div><div className="v">{v}{u && <span className="u">{u}</span>}</div></div>
}

function latest(events, key) {
  for (let i = events.length - 1; i >= 0; i--) if (events[i][key] != null) return events[i][key]
  return null
}

/* ---------- app ---------- */
export default function App() {
  useClock(1000) // re-render every second for live elapsed
  const runs = usePoll('/api/runs', 3000)
  const [sel, setSel] = useState(null)

  // auto-select newest run once, keep user choice afterward
  useEffect(() => {
    if (runs && runs.length && !sel) setSel({ exp: runs[0].exp, run_id: runs[0].run_id })
  }, [runs, sel])

  const run = usePoll(sel ? `/api/run?exp=${encodeURIComponent(sel.exp)}&id=${encodeURIComponent(sel.run_id)}` : null, 1200, !!sel)
  const meta = run?.meta
  const events = run?.events || []

  const lossPts = useMemo(
    () => events.filter((e) => e.loss != null).map((e) => ({ x: e.step, y: e.loss })), [events])
  const rewardPts = useMemo(
    () => events.filter((e) => e.reward != null).map((e) => ({ x: e.step, y: e.reward })), [events])
  const lrPts = useMemo(
    () => events.filter((e) => e.learning_rate != null).map((e) => ({ x: e.step, y: e.learning_rate })), [events])

  if (!meta) return (
    <div className="app">
      <Sidebar runs={runs} sel={sel} onSelect={setSel} />
      <main className="main"><div className="empty-main">select a run</div></main>
    </div>
  )

  const running = meta.status === 'running'
  const elapsed = (running ? Date.now() / 1000 : (meta.ended || meta.started)) - meta.started
  const curStep = events.length ? events[events.length - 1].step : 0
  const primary = rewardPts.length && !lossPts.length ? rewardPts : lossPts
  const primaryName = rewardPts.length && !lossPts.length ? 'reward' : 'loss'
  const hasResult = meta.status === 'done' && (meta.before != null || meta.after != null)
  const delta = meta.delta
  const good = delta == null ? null : (lowerBetter(meta.metric) ? delta < 0 : delta > 0)

  return (
    <div className="app">
      <Sidebar runs={runs} sel={sel} onSelect={setSel} />
      <main className="main">
        <div className="main-head">
          <div>
            <div className="model">{meta.model}</div>
            <div className="pills">
              <span className="pill">{meta.pack}</span>
              <span className="pill ghost">{meta.metric}</span>
              <span className="pill ghost">{meta.run_id}</span>
            </div>
          </div>
          <div className={`badge ${meta.status}`}>
            <span className={`sdot ${meta.status}`} />
            {running ? 'training' : 'done'} · {fmtDur(elapsed)}
          </div>
        </div>

        <div className="tiles">
          <Tile k="step" v={curStep} />
          <Tile k={primaryName} v={fmtNum(latest(events, primaryName === 'reward' ? 'reward' : 'loss'))} />
          <Tile k="learning rate" v={fmtNum(latest(events, 'learning_rate'), 2)} />
          <Tile k="grad norm" v={fmtNum(latest(events, 'grad_norm'), 2)} />
          <Tile k="started" v={clock(meta.started)} />
        </div>

        {hasResult && (
          <div className="card">
            <h3>Result · {meta.metric}</h3>
            <div className="result">
              <span className="ba">{fmtNum(meta.before)}</span>
              <span className="arrow">→</span>
              <span className="ba">{fmtNum(meta.after)}</span>
              {delta != null && (
                <span className={`delta ${good ? 'good' : 'bad'}`}>
                  {delta > 0 ? '+' : ''}{fmtNum(delta)} {good ? 'improved' : 'regressed'}
                </span>
              )}
            </div>
          </div>
        )}

        <div className="card">
          <h3>Training {primaryName}</h3>
          <div className="hint">{primary.length} logged step{primary.length === 1 ? '' : 's'}{running ? ' · live, updating every 1.2s' : ''}</div>
          <LineChart points={primary} color={primaryName === 'reward' ? '#c2a26a' : '#6f9183'} />
        </div>

        {lrPts.length > 1 && (
          <div className="card">
            <h3>Learning rate</h3>
            <div className="hint">schedule across steps</div>
            <LineChart points={lrPts} color="#6e8ca0" height={150} />
          </div>
        )}

        <div className="foot">reading experiments/{meta.exp}/runs/{meta.run_id}/ · {events.length} events</div>
      </main>
    </div>
  )
}
