# ComfyUI-on-Runpod harness — design

**Date:** 2026-09-29
**Status:** approved 2026-09-29

## 1. Goal

A CLI that rents a Runpod GPU, runs a batch of generations against ComfyUI on it,
downloads the results, and shuts the machine down. Five modes: `t2i`, `i2i`, `t2v`,
`i2v`, `v2v`. Personal single-user tool. Typical run: 10–20 images or 1–2 videos.

Non-goal: a service, a web UI, multi-user anything, or a general ComfyUI wrapper.

## 2. Decisions already taken

| Decision | Choice | Why |
| --- | --- | --- |
| Interaction | CLI-first, ComfyUI web UI reachable when wanted | user choice |
| Config shape | one curated workflow per mode + `overrides` escape hatch | user choice |
| Model storage | persistent network volume | see §3 |
| Access | SSH tunnel only, port 8188 never public | user choice |
| Control plane | `api.runpod.io/v2` called directly | see §4 |
| Custom ComfyUI nodes | none | core ComfyUI covers all 5 modes (§7) |

## 3. Storage: network volume

**Standard network volume, 75 GB, EU-RO-1, $5.25/month.**

Pricing verified at <https://docs.runpod.io/storage/network-volumes> §Pricing:
first 1 TB at **$0.07/GB/month**. 75 × 0.07 = $5.25.

The volume holds the ComfyUI install (the official template copies itself to
`/workspace` on first boot) *plus* models *plus* output headroom — not models alone:

| | GB |
| --- | --- |
| Models, all 5 modes (§6) | 48.7 |
| ComfyUI install + venv | ~20 |
| Output headroom | ~5 |
| **Provision** | **75** |

Size can be increased later, never decreased.

### Why not re-download each session

Runpod publishes no uplink figure. The only evidence is 13 data points from Runpod's
support Discord — a complaints channel, so a biased sample — spanning 0.3 to 120 MB/s,
median ~15 MB/s, containing exactly one clean size÷time measurement (32 GB in ~10 min).
The spread is a **per-host lottery**, not a datacenter property: the same user, same
files, got 15 MB/s and 120 MB/s on different days, and one person fixed 2 MB/s by
destroying the pod and renting another.

At the median, 48.7 GB is ~54 min of download per session. On a $0.72/hr GPU that is
~$0.65 of rented hardware watching a progress bar — the volume pays for itself at
about five sessions a month, and on the *first* session if a bad host is drawn.
Runpod's own figure for the alternative: 32 GB loaded from a volume in ~21 seconds
(volumes are NVMe, 200–400 MB/s).

### Rejected: Global Volumes

Region-independent and elastic, which would have dodged the Secure Cloud markup. Rejected
on <https://docs.runpod.io/storage/globalstore> §Limitations:

- *"No permission bits… This may produce warnings when downloading models from sources
  such as Hugging Face."* — hits our exact workflow.
- *"Eventual consistency… updates made to the volume after the Pod starts may not be
  immediately visible inside the container."* — breaks download-then-rescan.
- No published pricing at all. Beta. No file locking or atomic rename, so a Python venv
  on it is unsafe.

### Consequences to accept

- **Volumes force Secure Cloud** (*"Network volumes are only available for Pods in the
  Secure Cloud"*), so the GPU costs $0.72–0.74/hr rather than the $0.34 community rate.
- **Volumes pin the datacenter**, which drives §5.
- **At $0 account balance**, storage keeps accruing and the volume *"may eventually be
  terminated and its data cannot be recovered."* Enable low-balance notifications.

## 4. Control plane: REST v2, not runpodctl

Runpod REST v1 **retires 2026-11-15**; GraphQL follows in early 2027. `runpodctl` v2.14.0
still hardcodes `rest.runpod.io/v1` and creates GPU pods over GraphQL, and the Python SDK
1.12.0 is GraphQL too. Both break within weeks of this being written.

The harness therefore calls `https://api.runpod.io/v2` directly. Schema taken from the
live spec at `https://api.runpod.io/v2/openapi.json` (public, no auth). Surface needed:

| Call | Use |
| --- | --- |
| `POST /v2/pods` | create |
| `GET /v2/pods/{id}` | status, `runtime.ports[]`, `ssh` block |
| `POST /v2/pods/{id}/action` `{"action":"terminate"}` | teardown |
| `POST /v2/network-volumes` | one-time volume creation |
| `GET /v2/catalog/gpus?include=AVAILABILITY&product=POD` | pick a GPU that exists |

Notes from the spec: `cloud` defaults to `SECURE`; exactly one of `gpu`/`cpu`; `image`
required unless `templateId`; `unevaluatedProperties: false` so a v1-shaped body fails
loudly; `runtime` is null until status is `RUNNING`; read the response's `actions` array
rather than hardcoding the state machine.

## 5. Placement and hardware

**EU-RO-1.** It is the only datacenter with both standard network volumes and
better-than-LOW GPU stock. Queried live from the catalog: of 67 GPU/datacenter pairs in
volume-capable datacenters, **65 are LOW**, and the two that are not are both here:

| VRAM | GPU | Secure $/hr | Stock | ComfyUI template |
| --- | --- | --- | --- | --- |
| 32 GB | **RTX PRO 4500 Blackwell** | **$0.72** | MEDIUM | CUDA 13 — `2lv7ev3wfp` |
| 24 GB | RTX 4090 | $0.74 | HIGH | CUDA 12.8 — `cw3nka7d08` |

Primary is the **PRO 4500 Blackwell**: cheaper than the 4090 and 8 GB more VRAM, which is
the line between holding a 14B video model resident and streaming it from system RAM
(measured: Wan 2.2 A14B fp8 on a 24 GB card peaked at 6.1 GB VRAM and took 1546 s/clip —
that low peak is the tell that it was streaming).

Because the template depends on the GPU's CUDA line, config carries a small ordered
list of `{gpu, templateId}` pairs and the harness takes the first that has capacity.
Two entries, one loop — this is not an abstraction, it is the fallback that makes
`comfy up` work on a day the primary is out of stock.

**Risk:** the CUDA 13 template is verified only from its metadata and readme, not from a
live run. Phase 1 must boot it for real before we rely on it; the 4090 path is the
fallback if it misbehaves.

## 6. Model set — 48.7 GB, all five modes

Two source repos, both `Comfy-Org/*` mirrors, so **no HuggingFace token is required**.
All Apache-2.0. Every filename and size below was verified against the HuggingFace API
on 2026-09-29; sizes are decimal GB as the API reports them.

Repo **A** = `Comfy-Org/z_image`, repo **B** = `Comfy-Org/Wan_2.2_ComfyUI_Repackaged`.
All paths are under `split_files/` in both repos.

| Mode | Model | Repo | File → target dir | GB |
| --- | --- | --- | --- | --- |
| `t2i` | Z-Image bf16, non-distilled | A | `diffusion_models/z_image_bf16.safetensors` | 12.3 |
| | | A | `text_encoders/qwen_3_4b.safetensors` | 8.0 |
| | | A | `vae/ae.safetensors` | 0.3 |
| `i2i` | *same weights as `t2i`* — classic denoise graph | — | — | **0** |
| `t2v` + `i2v` | Wan 2.2 TI2V 5B (one model does both) | B | `diffusion_models/wan2.2_ti2v_5B_fp16.safetensors` | 10.0 |
| `v2v` | Wan 2.2 Fun Control 5B | B | `diffusion_models/wan2.2_fun_control_5B_bf16.safetensors` | 10.0 |
| shared | Wan text encoder | B | `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6.7 |
| shared | Wan 2.2 VAE | B | `vae/wan2.2_vae.safetensors` | 1.4 |
| | | | **Total** | **48.7** |

Repo **A** is `Comfy-Org/z_image` — the **non-distilled** Z-Image, not `z_image_turbo`.
This implements the quality-over-speed policy (§13.2): the Turbo variant is an 8-step
distillation, and `int8_convrot` is a quantization, and both trade fidelity for
throughput. Taking `z_image_bf16` drops both. It costs +6.1 GB (+$0.43/month) and no
extra generation time — bf16 needs no dequantization step; it needs VRAM, and the
32 GB primary GPU has it (12.3 GB weights + 8.0 GB text encoder leaves ample headroom
at 1024×1024).

Honest caveat: `z_image_turbo` has 8.1M downloads against 216K for the non-distilled
repo, so the variant we are choosing has far less community validation. Download counts
measure convenience rather than quality, and the free fallback is one filename away —
`z_image_int8_convrot.safetensors` is the same 6.2 GB as Turbo.

Note the Wan text encoder lives in the 2.2 repo, not the 2.1 one — every Wan file we
need comes from repo B, which keeps `provision` to two repos.

`i2i` costs zero extra disk because classic image-to-image is a denoise graph over the
`t2i` model: `LoadImage → VAEEncode → KSampler(denoise=0.6) → VAEDecode → SaveImage`,
about eight core nodes. There is no official template for it — the shipped "Image Edit"
templates are *instruction*-edit models ("add a top hat"), not a strength dial — so we
author this one graph by hand in Phase 1.

Wan 2.2 TI2V 5B is chosen over the 14B pairs deliberately: **one model serves both `t2v`
and `i2v`** — 10.0 GB against roughly 57 GB for the four fp8 14B expert files — it is the
variant Wan's own card says fits 24 GB, and it does 720p at 24 fps in "under 9 minutes"
on a consumer GPU by Wan's own measurement.

### Documented upgrades, not installed now

Each requires growing the volume, which is allowed (increase only, never decrease).

| Upgrade | +GB | +$/mo | Buys |
| --- | --- | --- | --- |
| Wan 2.2 Fun Control 14B fp8 pair (verified 14.3 × 2) | 28.6 | 2.00 | better `v2v` on the same text encoder and VAE |
| Wan 2.2 14B t2v+i2v fp8 + lightx2v 4-step LoRAs | ~57 | 3.99 | materially better video. The 4-step LoRAs are not optional — they turn ~40 min/clip into ~2 |
| FLUX.2 klein 4B edit *(size unverified)* | ~7.5 | 0.53 | instruction-editing ("add a hat") alongside denoise `i2i` |
| Bernini-R *(size unverified)* | ~30 | 2.10 | purpose-built video *editing* (outfit swap, insertion) |

## 7. Workflows

**Zero custom nodes.** Core ComfyUI covers all five modes: `LoadVideo`, `SaveVideo`,
`GetVideoComponents`, `WanVaceToVideo`, `Wan22FunControlToVideo`, `FrameInterpolate`,
`Canny`. VideoHelperSuite, Kijai's WanVideoWrapper, ComfyUI-Frame-Interpolation and
`comfyui_controlnet_aux` are all redundant now, and two of them are stale. This is what
keeps the stock official template usable untouched.

**Workflows are committed as API-format JSON**, one per mode, in `workflows/`.

Getting them: the good current templates are wrapped in `definitions.subgraphs`, and
converting UI format to API format client-side is a 1,760-line problem (that is the size
of comfy-cli's converter). We sidestep it — in Phase 1, open each workflow in the browser
over the tunnel and use **File → Export (API)**. Five exports, once, zero code. The
`i2i` graph is authored by hand in the same sitting.

Parameterisation: nodes are located by their `_meta.title`, which we set to stable names
(`positive`, `negative`, `latent`, `sampler`, `output`) when exporting. The harness
overwrites `inputs` on those nodes. `overrides` in the config addresses the same
`title.input` path, which is why the escape hatch costs a dict merge rather than a
feature.

## 8. Config

Two files. `comfy.yaml` is infrastructure and changes almost never:

```yaml
datacenter: EU-RO-1
volume_id: vol_xxxxxxxx
gpus:                          # ordered; first with capacity wins
  - { id: "NVIDIA RTX PRO 4500 Blackwell", template: "2lv7ev3wfp" }
  - { id: "NVIDIA GeForce RTX 4090",       template: "cw3nka7d08" }
terminate_after: 3h            # hard backstop set at creation
```

A run file is per-idea:

```yaml
mode: t2i
prompt: "a fox in a misty forest, cinematic"
negative: "blurry, watermark"
count: 16
size: [1024, 1024]
seed: random
overrides:                     # optional
  sampler.cfg: 3.5
```

`i2i`/`i2v`/`v2v` add `input: path/to/file`, uploaded via ComfyUI's `/upload/image`.

## 9. Commands

| Command | Does |
| --- | --- |
| `comfy provision` | one-time: create the volume, attach it to a **CPU pod at $0.07/hr**, download the model set, destroy the pod |
| `comfy up` | create the GPU pod, open the SSH tunnel, wait for `/system_stats` |
| `comfy run <file>` | resolve mode → workflow, apply params, queue `count` jobs, stream progress, download to `out/<date>-<slug>/` |
| `comfy down` | terminate the pod, report what the session cost |
| `comfy status` | is a pod up, how long, how much so far |

`provision` deliberately uses a CPU pod: the volume attaches to one (`mounts.persistent`
is invalid for CPU pods, network volumes are not), models download at datacenter speed,
and a slow host costs $0.07/hr instead of $0.72. Even a 9-hour worst-case download is
about $0.63, once, ever.

`up` and `run` are separate because the user asked for them to be — but `run` starts a
pod if none is up, so the common case is one command.

## 10. Access and cost safety

**SSH tunnel only.** Port 8188 is never exposed. `ports: ["22/tcp"]` at creation,
`startSsh: true`, and the harness forwards `8188` to localhost. Both the CLI and the
browser talk to `http://localhost:8188`.

Confirmed from the OpenAPI spec: `startSsh` injects `PUBLIC_KEY` from the account's
registered keys and *"with none registered the flag does nothing and the pod has no SSH
access"*; the `ssh.direct` variant additionally requires `22/tcp` exposed. One key is
already registered on this account. Runpod's *proxy* SSH cannot forward ports — only the
direct-TCP flavour can, so `22/tcp` is load-bearing, not incidental.

Two things this buys beyond privacy: no 100-second Cloudflare timeout (we are off the
HTTP proxy entirely), so large video downloads work; and WebSockets work, so progress
comes from ComfyUI's `ws://` stream instead of polling.

**Cost guards, in order of trust:**

1. `terminate_after` passed at pod creation — deletes the pod server-side even if the
   laptop closes. Terminate, not stop: a stopped pod still bills volume disk at
   $0.20/GB/month.
2. `comfy down` terminates and prints the spend.
3. `comfy status` so a forgotten pod is one command from visible.

## 11. Flake

`flake.nix` providing a devShell (Python + `httpx`/`PyYAML`/`websockets`, `openssh`,
`jq`) and `packages.default` so `nix run` gives the CLI. Systems: `x86_64-linux`,
`aarch64-linux`, matching the house style in the user's other repos.

The GPU side needs nothing from Nix — it is Runpod's Docker image.

## 12. Verification

Phase 1 is itself the integration test, and it must produce evidence, not assertions:

1. CUDA 13 template boots on the PRO 4500 and serves `/system_stats`.
2. `ssh -L 8188:localhost:8188` works over direct-TCP SSH.
3. Model download completes; record the observed MB/s — the first real datapoint anyone
   has for this account.
4. One generation per mode, five output files.

Then per-mode runs through the CLI, and `/simplify` + `/ponytail-review` as CLAUDE.md
requires.

## 13. Decisions and remaining questions

**13.1 GPU — settled.** PRO 4500 Blackwell 32 GB at $0.72 primary, RTX 4090 24 GB at
$0.74 as the HIGH-stock fallback. Config holds both in order (§5).

**13.2 Quality over speed, at equal cost — settled (user, 2026-09-29).** The governing
rule for every subsequent choice: *where quality costs no more money, take quality.*
"No more money" is judged on the total monthly bill, not on generation seconds — GPU
time at $0.72/hr is cheap enough that step count is a rounding error, while volume GB
is a standing charge.

Applied so far:

| Choice | Ruling |
| --- | --- |
| `z_image_turbo` int8 → **`z_image` bf16** | taken. +$0.43/mo, no time cost |
| Sampler steps in shipped workflows | set for quality, not the distilled minimum. 20 images at 8 s instead of 2 s is ~$0.03 of GPU |
| Wan 2.2 TI2V 5B → 14B pairs | **not** taken. +57 GB is +$3.99/mo — an 80% rise in the standing bill, so it fails the equal-cost test. Stays the top documented upgrade (§6) |
| `umt5_xxl_fp8` → `umt5_xxl_fp16` | open, see 13.4 |

**13.3 `count` for video — assumed.** Same config field as images; the user sets it low
for video modes. Revisit only if that proves annoying in practice.

**13.4 Open: the Wan text encoder precision.** The spec carries
`umt5_xxl_fp8_e4m3fn_scaled` (6.7 GB); `umt5_xxl_fp16` (11.4 GB) also exists in repo B.
Under 13.2 the fp16 encoder is a candidate at +4.7 GB (+$0.33/mo), but text-encoder
precision affects video output less visibly than the diffusion model's does, and it is
untested here. Decide empirically in Phase 1 — generate the same prompt both ways
before committing another $0.33/month.

## 14. Risks

| Risk | Mitigation |
| --- | --- |
| CUDA 13 template never live-verified | Phase 1 boots it first; 4090 + CUDA 12.8 is the fallback |
| Only 2 GPU/DC pairs above LOW stock — both in one datacenter | ordered GPU list; if both are out, `up` fails clearly rather than hanging |
| Export (API) may not round-trip a subgraph-heavy template | fall back to a simpler official template, or author the graph by hand as with `i2i` |
| WebSocket over SSH tunnel untested | poll `/history` as fallback; both are a few lines |
| Volume silently billing while unused | it is $5.25/mo; `comfy status` surfaces it, low-balance notifications on |
