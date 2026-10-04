# Tailscale access

This server's dashboard is **http://afk:8766**, also reachable through
`http://afk.tail5ec85b.ts.net:8766` or `http://100.123.76.107:8766`.
The short name uses Tailscale MagicDNS. Connect your viewing device to the
tailnet. The listener binds to the Tailscale IP, and Tailscale access rules control who
can reach it. There is no separate application login.

The backend accepts only its configured bind address, local hostnames, and explicitly
allowed hostnames. Same-origin checks still apply to campaign controls. Tailscale Serve
HTTPS could be added later; this account cannot change the machine's Serve configuration.

```bash
systemctl --user status nekaise-loop-dashboard nekaise-loop-supervisor
systemctl --user restart nekaise-loop-dashboard
journalctl --user -u nekaise-loop-dashboard -n 50
```

The installed units are copied from `nekaise-loop-dashboard.service` and
`nekaise-loop-supervisor.service`. The supervisor is the persistent scheduler and has an
automatic restart policy; it owns no model weights. It reads machine-local
settings from ignored `workspace/dashboard.env`. This server has user lingering enabled,
so the enabled service can run after logout and start after reboot. Restart retries handle
the Tailscale address becoming available after startup. Stopping the dashboard leaves the
independent supervisor and learning worker running. Use campaign pause/stop to control
learning; `nekaise-loop shutdown-worker` stops the supervisor and owned execution for
maintenance. `start`/`resume` bring the supervisor back. An API restart also reconnects
to outstanding durable work. Do not edit Python/prompts until execution has stopped.

For another server, obtain its IPv4 address with `tailscale ip -4` and DNS name from
`tailscale status --json`. Choose an unused port, and create:

```dotenv
NEKAISE_DASHBOARD_HOST=100.x.y.z
NEKAISE_DASHBOARD_PORT=8766
NEKAISE_DASHBOARD_DNS=your-server.your-tailnet.ts.net
NEKAISE_DASHBOARD_SHORT_DNS=your-server
```

Save those actual values in `workspace/dashboard.env`, then install the service:

The short-name setting explicitly permits its HTTP Host header; DNS resolution alone
does not grant access. If omitted, the unit defaults to the already-allowed `localhost`.

```bash
mkdir -p ~/.config/systemd/user
cp deploy/nekaise-loop-dashboard.service deploy/nekaise-loop-supervisor.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now nekaise-loop-supervisor
systemctl --user enable --now nekaise-loop-dashboard
```

The units assume this checkout lives at `~/Code/nekaise-studio`. Adjust its paths
if needed. For a foreground process instead:

```bash
.venv/bin/nekaise-loop serve --host 100.x.y.z --port 8766 --allow-host your-server.your-tailnet.ts.net --allow-host your-server
```

The server's former `~/Code/nekaise-studio-loop` path is a compatibility symlink to this
checkout. Keep it while historical checkpoints, campaign configs and call logs contain
that absolute prefix. Their immutable contents and hashes have not been rewritten.
Python/prompt changes made during the rename require a fresh continuation on Resume;
the service preserves existing campaigns and chooses their verified checkpoints.

For a custom workspace, update the supervisor unit’s `--workspace` argument and the
dashboard environment together. Foreground/custom installations without an installed
unit use a detached supervisor; commands and recovery timers still persist in SQLite,
but a process manager is needed to restart the supervisor after its own crash or reboot.

## Independent evaluation display

Studio accepts Bench's aggregate `chat-2`, `chat-1` and `completion-1` observations,
including sealed history pages. Keep each release and protocol in its original series.
When Bench moves to a new observer state, update the dashboard's ignored
`workspace/dashboard.env` to its actual projection directory, for example:

```dotenv
NEKAISE_BENCH_PROJECTION_DIR=/home/zengp/Code/nekaise-bench/workspace/monitor-v2/projection
```

Restart only the dashboard after changing this setting. The legacy default remains
`../nekaise-bench/workspace/monitor/projection` for existing installations; Studio does
not guess a series from whichever directory has newer files. Verify the dedicated
`/api/campaigns/{id}/benchmark` endpoint and its history endpoint after deployment.
Missing display data does not establish that no evaluation has run. Actual measurements
stay in Bench storage and never enter teaching, normal Report or recovery inputs.

## Local GPU chat

The Model tab defaults to the latest Kai checkpoint on CPU. To use a separate local
model, pause training through Studio and wait for the worker to stop. Put its configuration
in ignored `workspace/model-chat.json`:

```json
{
  "display_name": "Qwen3.8-27B",
  "model_source": "unsloth/Qwen3.8-27B-GGUF",
  "revision": "4ca720788d1e01f1bff70c033e0d0028fd02e502",
  "quantization": "UD-Q6_K",
  "model_path": "/absolute/path/Qwen3.8-27B-UD-Q6_K.gguf",
  "sha256": "c9c206812fbe4ac7b76a729e25928b63f2ae89d37f69da7a71c20aec763cd436",
  "server_binary": "/absolute/path/llama-server",
  "port": 8791,
  "context_tokens": 8192,
  "max_new_tokens": 2048
}
```

Use a CUDA llama.cpp build that supports the model (validated with b11396). Keep weights
and runtime binaries outside Git. The worker verifies the complete model hash before
loading. If CUDA libraries need a search path, put `LD_LIBRARY_PATH=/absolute/library/path`
in ignored `workspace/model-chat.env`. Install and start the dedicated worker:

```bash
cp deploy/nekaise-model-chat.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start nekaise-model-chat
systemctl --user status nekaise-model-chat
```

Refresh Studio and open Model. Loading and stopped states are unavailable; they never
fall back silently to Kai. Text chat uses the model's native template with thinking off.
The reply limit is 2,048 tokens; total context is 8,192 tokens in this configuration.
Prompt limits include the whole conversation. File/image input and tool execution are
not part of this interface. Messages remain private to the page and in-memory inference.

The chat worker holds the same lock as training. Resume through Studio first stops the
chat server and frees the GPU; the training supervisor then processes the queued command.
Do not enable this chat service at boot. After another explicit training pause, start
the service again. To return the Model tab to Kai:

```bash
systemctl --user stop nekaise-model-chat
mv workspace/model-chat.json workspace/model-chat.disabled.json
```

Stop this worker before code maintenance: it holds a shared source lock. Source changes
still require the usual verified continuation when training later resumes. Chat does not
replace the training student, optimizer or checkpoint lineage.
