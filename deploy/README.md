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
