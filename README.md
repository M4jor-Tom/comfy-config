# comfy-runpod

Rent a GPU, generate, download, destroy the machine. Five modes — `t2i`, `i2i`, `t2v`,
`i2v`, `v2v` — driven from a small config file against ComfyUI on a Runpod pod, reached
over an SSH tunnel so nothing is ever exposed to the internet.

```bash
$ comfy up                        # boot a pod, open the tunnel
$ comfy run prompts/fox.yaml      # 16 images, downloaded to ./out/
$ comfy down                      # terminate, report what it cost
```

## Setup

You need a Runpod account with credit and an SSH key registered on it
(`PUT /v2/account/ssh-keys`, or the console). Then:

```bash
export RUNPOD_API_KEY=...         # console.runpod.io/user/settings
cp comfy.example.yaml comfy.yaml  # gitignored; holds your volume id
nix develop                       # the only way to get Python here
```

**Inside `nix develop`, `comfy` is on `PATH`** (`python -m comfy_runpod.cli` still works
too, and behaves identically). `nix develop` does not inherit `RUNPOD_API_KEY`, so live
commands look like:

```bash
nix develop -c bash -c 'source ~/set_runpod_key.sh && comfy status'
```

## Two ways to run, and which you want

**Volume-less (default, and cheaper for most people).** Leave `volume_id` blank. Every
`up` re-downloads the models. Costs nothing while idle.

**With a volume.** `comfy provision` creates a 75 GB network volume and fills it once.
Models are then ready in seconds, but the volume bills **$0.07/GB/month** — about
**$5.25** — whether or not you generate anything, and it forces Secure Cloud pricing.

The measured numbers, from this account on 2026-09-29:

| | |
| --- | --- |
| HuggingFace → Runpod throughput | **124.4 MB/s (995 Mbps)** |
| Full 48.7 GB model set | **~7 minutes** |
| Cost of that download at $0.72/hr | **~$0.08** |
| Volume break-even | **~65 sessions/month** |

So unless you generate most days, **volume-less wins**. Published community figures suggest
a much worse median (~15 MB/s), but they come from a support-forum complaints channel —
treat them as the bad tail. Throughput is a per-host lottery; `comfy provision` is
re-runnable if you draw badly and want a volume after all.

## The config

`comfy.yaml` is infrastructure and rarely changes:

```yaml
datacenter: EU-RO-1
volume_id: g3h3d0v2ee          # blank for volume-less
gpus:                          # ordered; first with capacity wins
  - { id: "NVIDIA RTX PRO 4500 Blackwell", template: "wgd3p4n4o6" }
  - { id: "NVIDIA GeForce RTX 5090",       template: "wgd3p4n4o6" }
  - { id: "NVIDIA GeForce RTX 4090",       template: "cw3nka7d08" }
terminate_after: 3h
```

The ordered list is not decoration. EU-RO-1 stock moved from HIGH to NONE and back within
ninety minutes during development, and one `up` failed outright before a retry succeeded.
The template is tied to the GPU's CUDA line — Blackwell and the 5090 need CUDA 13
(`wgd3p4n4o6`), the 4090 needs CUDA 12.8 (`cw3nka7d08`). Mixing them is the most common
first-run failure.

A run file is per-idea:

```yaml
mode: t2i
prompt: "a red fox in a misty pine forest, cinematic"
negative: "blurry, watermark"
count: 16
size: [1024, 1024]
seed: random
overrides:                     # optional escape hatch
  sampler.cfg: 3.5
```

`i2i`, `i2v` and `v2v` add `input: path/to/file`, resolved relative to the run file.
`overrides` addresses any node input as `title.field`, where `title` is the node's
`_meta.title` in the workflow JSON.

## Models

All from `Comfy-Org/*` mirrors, so **no HuggingFace token is needed**. 48.7 GB total.

| Mode | Model |
| --- | --- |
| `t2i`, `i2i` | Z-Image bf16 (non-distilled) + `qwen_3_4b` + `ae` |
| `t2v`, `i2v` | Wan 2.2 TI2V 5B — one model does both |
| `v2v` | Wan 2.2 Fun Control 5B |

Non-distilled bf16 is deliberate: the Turbo variant is an 8-step distillation and
`int8_convrot` is a quantization, and both trade quality for speed. Dropping both costs
+6.1 GB and no extra generation time.

**Zero custom ComfyUI nodes.** Core covers all five modes.

## Cost control

Runpod REST v2 has **no server-side pod TTL** — `terminateAfter` does not exist, and
`PodAction` is only `start|stop|restart|terminate`. So the guards are:

1. **`comfy run` terminates the pod when the batch finishes**, unless you pass `--keep`.
   The common path never leaves a pod running.
2. A local watchdog from `terminate_after`. Second line only — it dies with your machine.
3. `comfy status` reports real billed spend from `/v2/billing/pods`.
4. `comfy teardown` terminates every pod and lists any volume with its monthly cost.
   Volumes are only ever deleted with the explicit `--delete-volume` flag — that destroys
   every downloaded model irreversibly, so it never happens by default.

**The residual risk, stated plainly:** `--keep` plus a crashed laptop bills ~$17/day until
you notice. Enable low-balance notifications.

Always terminate, never stop — a stopped pod still bills its disk at $0.20/GB/month.

## Privacy

Port 8188 is never exposed. Pods are created with `ports: ["22/tcp"]` only, and ComfyUI is
reached through `ssh -L` at `127.0.0.1:8188` — verified live: the public proxy URL returns
404. Point a browser at `http://localhost:8188` while a pod is up and you get the real
ComfyUI UI, privately.

This requires direct-TCP SSH; Runpod's *proxy* SSH cannot forward ports.

## A cheaper option for plain work

Runpod sells managed per-generation endpoints — roughly $0.005/image and $0.30/clip — that
need no infrastructure at all. For straightforward `t2i` or `t2v` they are competitive with
self-hosting and start instantly. They do **not** do video2video, custom graphs, or your own
LoRAs, which is why this exists. Worth a `curl` on a day you want twenty quick images.

## Development

```bash
nix develop -c python -m pytest tests/ -q
```

Every test is offline — the Runpod client takes an injected transport, ComfyUI's HTTP calls
are overridden at two seams. No test may create a paid resource.

See `CLAUDE.md` for the rules that have live-verified reasons behind them, and
`docs/HANDOFF.md` for what is still unfinished.
