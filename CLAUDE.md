# comfy-runpod

A CLI that rents a Runpod GPU, runs a batch of ComfyUI generations, downloads the
results, and destroys the machine. Five modes: `t2i`, `i2i`, `t2v`, `i2v`, `v2v`.
Single user, personal use, 10–20 images or 1–2 videos per run.

## Read these before changing anything

- **Design/spec:** `docs/superpowers/specs/2026-09-29-comfy-runpod-harness-design.md` — every
  decision with the evidence behind it. This is the authority.
- **Implementation plan:** `docs/superpowers/plans/2026-09-29-comfy-runpod-harness.md` — task
  breakdown, with the exact code and tests.
- **What's left:** `docs/HANDOFF.md`.

## Hard rules

These are not style preferences. Each one has a live-verified reason.

1. **`https://api.runpod.io/v2` only.** REST v1 retires **2026-11-15**; GraphQL in early 2027.
   `runpodctl` and the `runpod` PyPI SDK both still ride the dying APIs — **do not add them**.
   The live schema is public at `https://api.runpod.io/v2/openapi.json` (no auth); read it rather
   than guessing field names.
2. **PyYAML is the only third-party dependency.** HTTP is stdlib `urllib.request`; SSH is the
   system `ssh` binary via `subprocess`. Do not add `httpx`, `requests`, or `paramiko`.
3. **Port 8188 is never publicly exposed.** `ports` is exactly `["22/tcp"]` and ComfyUI is reached
   through an SSH tunnel to `127.0.0.1:8188`. Runpod's *proxy* SSH cannot forward ports, so the
   direct-TCP endpoint — and therefore `22/tcp` — is load-bearing.
4. **An explicit `User-Agent` header is mandatory.** Runpod's Cloudflare edge answers
   `403 "error code: 1010"` to `urllib`'s default identity. Every API call fails without it.
   This cost a live debugging session to find; don't remove it.
5. **Zero custom ComfyUI nodes.** Core ComfyUI covers all five modes. VideoHelperSuite,
   Kijai's WanVideoWrapper, ComfyUI-Frame-Interpolation and `comfyui_controlnet_aux` are all
   redundant now, and two are stale.
6. **There is no server-side pod TTL in REST v2.** Every property of `CreatePodRequest` was
   enumerated — no `terminateAfter`, and `PodAction` is only `start|stop|restart|terminate`.
   Cost safety comes from `comfy run` terminating on completion by default.
7. **Terminate, never stop.** A stopped pod still bills its disk at $0.20/GB/month.
8. **Do not send `gpu.allowedCudaVersions` or `gpu.minCudaVersion`.** Either one *replaces* the
   template's CUDA constraint; leaving them off lets the template enforce the right CUDA line.
   Pairing the wrong template with a GPU is the most common first-run failure.

## Environment

**There is no `python3` on this machine outside the Nix dev shell.** Everything runs as:

```bash
nix develop -c python -m pytest tests/ -q
nix develop -c comfy --help
```

`nix develop` does **not** inherit `RUNPOD_API_KEY`. Live commands need:

```bash
nix develop -c bash -c 'source ~/set_runpod_key.sh && comfy status'
```

## Layout

| Path | Holds |
| --- | --- |
| `src/comfy_runpod/config.py` | `comfy.yaml` + run-file loading, `.comfy-state.json` |
| `src/comfy_runpod/runpod_api.py` | Runpod REST v2 client |
| `src/comfy_runpod/provision.py` | one-time volume + model download |
| `src/comfy_runpod/tunnel.py` | `ssh -L` lifecycle |
| `src/comfy_runpod/comfyui.py` | ComfyUI HTTP API |
| `src/comfy_runpod/workflow.py` | patches API-format graphs by `_meta.title` |
| `src/comfy_runpod/cli.py` | subcommands; **module-level `handlers` dict** |
| `workflows/*.json` | one API-format graph per mode |
| `comfy.yaml` | **gitignored** — carries the real volume ID |

**Adding a subcommand:** define `cmd_x(args) -> int` and assign `handlers["x"] = cmd_x`.
**Never rewrite `main`** — it dispatches generically and wraps every handler's
`ConfigError`/`RunpodError` into a clean `error: …`. Rewriting it silently drops commands.

## Placement, and why it is not negotiable

**Datacenter `EU-RO-1`.** It is the only datacenter pairing standard network volumes with
better-than-LOW GPU stock — of 67 GPU/datacenter pairs in volume-capable datacenters, 65 are LOW.

| GPU | VRAM | $/hr | Template |
| --- | --- | --- | --- |
| RTX PRO 4500 Blackwell | 32 GB | 0.72 | `wgd3p4n4o6` (CUDA 13) |
| RTX 4090 | 24 GB | 0.74 | `cw3nka7d08` (CUDA 12.8) |

Stock moves within hours — both have been HIGH and MEDIUM on the same day — which is why
`comfy.yaml` holds an **ordered** GPU list and the code takes the first with capacity.

⚠️ `2lv7ev3wfp` appears in older Runpod documentation as the CUDA 13 template. **It is dead** —
the API 404s on it. Discover current ids from `GET /v2/catalog/templates`.

## Money

- Volume: **$0.07/GB/month**, billed whether or not a pod runs.
- Pods bill per second while running. Network volumes force **Secure Cloud** pricing.
- **Measured HF → Runpod throughput: 124.4 MB/s (995 Mbps)** on EU-RO-1, 2026-09-29, plain `curl`.
  The 48.7 GB model set downloads in ~7 minutes. Published community figures are far worse
  (median ~15 MB/s) but come from a support-forum complaints channel; treat them as the bad tail,
  not the expectation. Throughput is a documented per-host lottery.
- At that speed a per-session re-download costs ~$0.08 of GPU, so a $5.25/month volume only pays
  off above roughly 65 sessions/month. **For a few sessions a week, volume-less is cheaper.**
  `create_pod` accepts `volume_id=None` and omits the mount, so both modes work.

## Testing

Every test is offline. The Runpod client is driven through an injected `transport`, and ComfyUI
through overridden `_get_json`/`_post_json`. **Never write a test that creates a paid resource**,
opens a real socket, or spawns a real `ssh`.

Before claiming anything works: `nix develop -c python -m pytest tests/ -q`, and confirm nothing
is billing with `nix develop -c bash -c 'source ~/set_runpod_key.sh && comfy status'`.

## Git

Gitflow: `feature/*` and `fix/*` branch from `develop`; `develop` merges to `main` at a release.
Never fast-forward — always `--no-ff`. Conventional Commits.
