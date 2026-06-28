import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// Read training telemetry straight off disk (experiments/<exp>/runs/<id>/{meta.json,events.jsonl}).
// Exposed as a tiny dev-server API so the UI is a single `npm run dev` with no separate backend.
const HERE = path.dirname(fileURLToPath(import.meta.url))
const EXPERIMENTS = path.resolve(HERE, '..', 'experiments')

const rd = (p) => { try { return fs.readdirSync(p) } catch { return [] } }
const rj = (p) => { try { return JSON.parse(fs.readFileSync(p, 'utf8')) } catch { return null } }

function listRuns() {
  const runs = []
  for (const exp of rd(EXPERIMENTS)) {
    const rdir = path.join(EXPERIMENTS, exp, 'runs')
    for (const rid of rd(rdir)) {
      const meta = rj(path.join(rdir, rid, 'meta.json'))
      if (meta) runs.push(meta)
    }
  }
  return runs.sort((a, b) => (b.started || 0) - (a.started || 0))
}

function readRun(exp, id) {
  const dir = path.join(EXPERIMENTS, exp, 'runs', id)
  const meta = rj(path.join(dir, 'meta.json'))
  let events = []
  try {
    events = fs.readFileSync(path.join(dir, 'events.jsonl'), 'utf8')
      .split('\n').filter(Boolean).map((l) => { try { return JSON.parse(l) } catch { return null } })
      .filter(Boolean)
  } catch { /* no events yet */ }
  return { meta, events }
}

function runApi() {
  return {
    name: 'nekaise-run-api',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const u = new URL(req.url, 'http://localhost')
        if (u.pathname === '/api/runs') {
          res.setHeader('content-type', 'application/json')
          return res.end(JSON.stringify(listRuns()))
        }
        if (u.pathname === '/api/run') {
          res.setHeader('content-type', 'application/json')
          return res.end(JSON.stringify(
            readRun(u.searchParams.get('exp') || '', u.searchParams.get('id') || '')))
        }
        next()
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), runApi()],
  // host:true binds 0.0.0.0 (reachable over the Tailscale interface). allowedHosts lets the
  // MagicDNS names through Vite's DNS-rebinding guard ('afk' and any *.tail5ec85b.ts.net).
  server: { port: 5273, host: true, allowedHosts: ['afk', '.tail5ec85b.ts.net'] },
})
