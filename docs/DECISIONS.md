# SDD ledger — plan: docs/superpowers/plans/2026-09-29-comfy-runpod-harness.md

Spec: docs/superpowers/specs/2026-09-29-comfy-runpod-harness-design.md (read, approved 2026-09-29)
Branch: feature/harness (from develop)

## Pre-flight conflict scan

### Shared-file / shared-interface pairs

| Pair | Produces → consumes | Finding |
| --- | --- | --- |
| T1 → T4, T5, T8 | `cli.py:main` stub → dispatch table → more handlers | **CONFLICT.** T1 writes `main` with a stub; T4 replaces it with a `handlers` dict; T5 and T8 must *extend* that dict, but the plan shows the dict only in T4. A naive T5/T8 implementer rewrites `main` and drops earlier commands. |
| T2 → T3 | `GpuChoice(id, template)` → `pick_gpu` | clean |
| T2 → T7, T8 | `Run` fields → test helper + `apply_run` | clean; T7's helper omits `seed`/`input`/`overrides`, all of which have defaults |
| T2 → T4 | `Infra.volume_id: str \| None` → `provision` | clean for provision, but see Ruling R4 |
| T3 → T4, T5, T8 | `ssh_target → (host, port, user)` | clean; unpacked as 3-tuple in all three |
| T3 → T4 | `create_pod(template_id=None, cpu=…)` | clean; `templateId` only set on the gpu branch |
| T3 → T5, T8 | `create_pod(volume_id: str)` required | **GAP.** No way to create a volume-less pod — see Ruling R4 |
| T5 → T8 | `comfyui.py`: `_get_json` → `_post_json` + 7 methods | clean, but T8 appends to T5's class and must preserve its imports |
| T7 → T8 | `load`/`apply_run`/`apply_overrides`/`seeds_for` | clean; all four used by `cmd_run` |
| T6 → T7, T8 | workflow JSON `_meta.title` → `find_by_title` | clean; T6 Step 5 verifies uniqueness, T7 raises on duplicates |

### Per-task internal agreement

| Task | Finding |
| --- | --- |
| T1 | **DEFECT.** `flake.nix` shellHook contains `${if true then "" else ""}` — meaningless Nix interpolation I left in. |
| T2 | Claims "15 tests"; the parametrize makes it 16 cases. Cosmetic only — tests agree with the code they specify. |
| T3 | 15 tests, count correct; fake transport matches the real `transport` signature. |
| T4 | **DEFECT ×2.** (a) generated bash calls `python3` on the pod, which may not exist; (b) `CPU_POD_IMAGE = "runpod/base:0.6.2-cpu"` is an unverified tag I invented. |
| T5 | **DEFECT.** Step 6 tells the implementer to wire `up`/`down`/`status` into the CLI but supplies no code — the plan's own no-placeholder rule. |
| T6 | Human-in-loop; steps are instructions, not code, which is correct for this task. |
| T7 | 17 tests, count correct; every helper it uses is defined in the same module. |
| T8 | 7 tests, count correct; `FakeComfy` overrides exactly the two transport seams. |

## Rulings

**R1 (T1 flake defect).** Strip the `${if true then "" else ""}` fragment; the shellHook
just reports whether `RUNPOD_API_KEY` is set, in plain bash. Cost if wrong: a cosmetic
shell message.

**R2 (T4 pod-side python3 + image tag).** Generated bash must be pure POSIX + curl with
no `python3`: pre-compute the minimum byte counts in Python at render time and emit them
as literals, and compute MB/s with shell arithmetic. The implementer must also verify the
CPU pod image tag exists before Task 4's live step rather than trusting my constant.
Cost if wrong: the download script dies on a paid pod.

**R3 (T5 missing code).** The implementer writes `cmd_up`/`cmd_down`/`cmd_status` from the
Interfaces block, extending T4's `handlers` dict. Extracting the shared `_bring_up`
helper is part of T5, as the plan's self-review already noted. Cost if wrong: one fix
round.

**R4 (volume-optional pods).** `create_pod` takes `volume_id: str | None`; when `None` it
omits `mounts` entirely. Reason: the user has asked that no recurring charge survive this
session, so the end state is volume-less — and a harness that *cannot* run without a
volume would be useless in exactly the state we are leaving them in. Three lines.
Cost if wrong: an unused parameter.

**R5 (cli.py growth).** `handlers` is a module-level dict built once; each task adds its
entries and never rewrites `main`. Carried into every dispatch that touches `cli.py`.
Cost if wrong: a later task silently drops earlier commands.

**R6 (teardown).** Add Task 10: terminate every pod, then report the volume as the one
standing charge and offer its deletion. The deletion itself is destructive and
irreversible, so it stops and asks — it is not mine to take. Cost if wrong: either a
surprise $5.25/month, or 48.7 GB to re-download.

**R7 (isolation).** Feature branch `feature/harness` off `develop` rather than a git
worktree. Reason: CLAUDE.md mandates gitflow feature branches, the repo is new with no
competing work, and a worktree would add ceremony without isolating anything further.
Cost if wrong: none material; the branch is still isolated from `develop`.

## Progress

Preflight: 5 defects found in the plan itself, all ruled (R1-R5). Plan amended for R4/R6 in 10e7223.
Task 1: dispatched (sonnet). BASE recorded c76a506, but controller committed 10e7223 (plan amendments)
  afterwards — resolve Task 1's review range from `git log` when the report lands, not from the
  stale BASE. Task 1 touches flake.nix/pyproject.toml/src/tests only; no overlap with the docs commit.
Task 1: implementer DONE_WITH_CONCERNS, commit a7a4a77 (range 10e7223..a7a4a77). 3/3 verification cmds
  passed: Python 3.12.14 + PyYAML 6.0.3, `nix develop -c python -m comfy_runpod.cli --help`, `nix run`.
  Implementer found 2 further plan defects I missed and fixed both:
    D1: brief's cli.py never called main() -> added `if __name__ == "__main__": sys.exit(main())`.
    D2: `nix run` broken, pname != console-script name -> added meta.mainProgram = "comfy".
  Ruling: both accepted as correct fixes to plan defects, not scope creep — D2 is the only way Step 5
  (`nix run`) can pass, and Step 5 is a named gate. Cost if wrong: a reviewer disagrees on idiom.
Task 1: task review dispatched (sonnet) on review-10e7223..a7a4a77.diff.
Task 1: review clean — Spec COMPLIANT, Quality APPROVED. Both undirected fixes (D1, D2) endorsed by
  reviewer as necessary for the brief's own Steps 4/5 to pass.
Task 1: minor (deferred): `COMMANDS` tuple in cli.py is dead code (came from my brief, not implementer).
Task 1: minor (deferred): report's PATH-isolation check ran inside `nix develop`, so it did not
  independently prove makeWrapperArgs supplies ssh when the caller's PATH lacks it. Idiom is standard.
Task 1: complete (commits 10e7223..a7a4a77, review clean)
Task 2: implementer DONE, commit 7034603, 16/16 tests, no regressions.

**R8 (dead template ID — found by live API check, discharging R2).** Spec and plan carried
  CUDA 13 ComfyUI template `2lv7ev3wfp`, taken from the local runpod skills-pack reference.
  `GET /v2/templates/2lv7ev3wfp` returns 404 "template not found". The live official catalog
  (`GET /v2/catalog/templates`, 14 entries) gives the real pair:
    cw3nka7d08  ComfyUI - CUDA 12.8  runpod/comfyui:1.3.3-comfyuiv0.30.0-cuda12.8  (alive)
    wgd3p4n4o6  ComfyUI - CUDA 13.0  runpod/comfyui:1.3.3-comfyuiv0.30.0-cuda13.0  (alive)
  Ruling: use wgd3p4n4o6. Spec + plan updated. `comfy.example.yaml` was written by Task 2
  from the stale brief and still says 2lv7ev3wfp — assigned to Task 3, whose tests are where
  template IDs are load-bearing. Task 2 is NOT at fault; it complied with its brief.
  Cost if wrong: Task 5's live GPU boot 404s on the primary GPU path.

**R9 (CPU pod image).** My invented `runpod/base:0.6.2-cpu` exists but dates from 2024-02-27
  (1.3 GB). Replaced with `runpod/base:1.0.2-ubuntu2404`, which is the image RunPod's own
  official `runpod-ubuntu-2404` template uses — so sshd/PUBLIC_KEY handling is proven rather
  than assumed. No `-cpu`-suffixed tags exist any more. Cost if wrong: Task 4's CPU pod has
  no sshd and the download step cannot run.
Task 2: review — Spec COMPLIANT. Quality: 1 Important, 2 Minor.
  **R10 (Important finding vs plan text).** Reviewer: `load_run`'s `int(seed_raw)` and
  `int(size[0..1])` are unguarded, raising raw ValueError instead of ConfigError. The same gap is
  verbatim in the brief's Step 3 code, so this is a finding against plan-mandated code — mine to rule.
  Ruling: finding is CORRECT, plan is wrong. `main` catches ConfigError and prints `error: <msg>`;
  a raw ValueError escapes that and shows the user a traceback, and ConfigError's own docstring
  claims to cover malformed config. Fix dispatched. Cost if wrong: a slightly larger diff than the
  brief specified.
Task 2: minor (deferred -> folded into fix round 1 since each is one line): `_duration_seconds` error
  omits the `{path}:` prefix; `each gpus entry needs an 'id'` does not echo the offending entry.
Task 2: reviewer ⚠️ "cannot verify from diff" on comfy.example.yaml's pricing/CUDA comments.
  RESOLVED by controller against the live API: 4090 $0.34 community / $0.74 secure, PRO 4500
  Blackwell $0.72 secure, both confirmed from GET /v2/catalog/gpus. CUDA pairing confirmed from
  GET /v2/catalog/templates. Not a gap.
Task 2: fix round 1 dispatched (resumed original implementer, haiku).
Task 2: fix round 1/5 (3 addressed, 0 open; commits 7034603..59e72bd, 19/19 tests)
Task 2: minor (deferred): test_load_infra_rejects_bad_terminate_after asserts only the substring
  "terminate_after", not the {path}: prefix, so it would pass without the finding-2 fix.
Task 2: minor (deferred): no test exercises the "gpus entry missing id entirely" branch.
Task 2: complete (commits a7a4a77..59e72bd, review clean)
Task 3: dispatched (haiku). BASE 59e72bd. Carries R8 (fix comfy.example.yaml stale template ID).

Task 4 pre-flight, all verified free against live APIs (discharges R2 fully):
  - cpu5c: HIGH availability in EU-RO-1 (and EUR-IS-1 only). $0.035/vCPU/hr secure, 2 GB RAM per
    vCPU, min 2 vCPU -> 2 vCPU = $0.07/hr, 4 GB RAM. The cheap-download plan is viable in the one
    datacenter the volume can live in. cpu3c is $0.03 but older generation; keeping cpu5c.
  - runpod/base:1.0.2-ubuntu2404 confirmed as the image behind RunPod's official runpod-ubuntu-2404
    template, so startSsh/PUBLIC_KEY handling is proven rather than assumed.
  - All 7 model URLs return HTTP 200 with content-length summing to EXACTLY 48.7 GB
    (12.3+8.0+0.3+10.0+10.0+6.7+1.4). The spec's figure is confirmed to the decimal.
  - EU-RO-1 offers STANDARD network volumes, GDPR+HIPAA, globalNetwork true.
  Note: /v2/catalog/cpus requires `product=POD` alongside `include=AVAILABILITY` or it 400s.
Task 3: implementer DONE, commit 5d66a68, 35 tests (19 + 16 new), comfy.example.yaml fixed to wgd3p4n4o6.
Task 3: review — Spec COMPLIANT. Quality: 2 Important, 2 Minor.
Task 3: reviewer ⚠️ "cannot verify query params from diff" — RESOLVED by controller live:
  GET /v2/catalog/gpus?include=AVAILABILITY&product=POD -> 200. GET /v2/billing/pods -> 200, and
  ?podId= is an accepted (optional) param. Both shapes as the client assumes. Not a gap.

**R11 (Critical, controller-found — my plan was wrong).** `pod_spend` sums `r.get("amount", 0.0)`
  over billing records. There IS no `amount` field: the live v2 schema's `PodBillingRecord` =
  BillingTimeRange + PodBillingAmounts + {podId}, and PodBillingAmounts requires exactly
  totalAmount/gpuAmount/cpuAmount/diskAmount. So `pod_spend` would report $0.00 forever, in the one
  module whose job is tracking money. The brief's test passed only because I fabricated an
  `{"amount": 0.21}` payload the API never returns — the test encoded my own wrong assumption.
  Ruling: read `metadata.totals.totalAmount` (verified live; present as 0 even with records: []),
  return 0.0 for genuine zero spend and None only on request failure or a missing totals object.
  This also subsumes the reviewer's Important finding that empty-vs-missing were conflated.
  Cost if wrong: every `comfy down` / `comfy status` silently under-reports spend as $0.00.
Task 3: minor (deferred): FakeTransport matches response keys by substring (`key in path`), so a
  future test registering both `/pods` and `/pods/x/action` on one instance could silently get the
  wrong canned response. Latent only — no current test collides.
Task 3: fix round 1 dispatched (resumed original implementer, haiku): R11 Critical, the dishonest
  ssh_target test, and the `volume_id: str | None` annotation.
Task 3: fix round 1/5 (3 addressed, 0 open; commits 5d66a68..4ff9249, 37 tests)
Task 3: minor (deferred): zero-spend test fails via TypeError rather than a clean assertion diff.
Task 3: minor (deferred): `if not totals` also returns None for an explicit empty-dict totals.
Task 3: complete (commits 59e72bd..4ff9249, review clean)
=== STOPPING for user approval: Task 4 is the first money step (~$0.15 + $5.25/mo standing). ===
=== USER APPROVED Task 4 (2026-09-29): create volume, delete it in Task 10. ~$0.35 real cost. ===
Task 4: dispatched (sonnet). BASE 4ff9249. LIVE — user-approved money step.

Task 5 pre-flight, from the live v2 schema's templateId description (free, no spend):
  - "explicit body fields override the template's, except `env`, which is merged per key with body
    values winning." Settings resolved from a template: image, args, disk, ports, env, registry,
    persistent mount, startSsh, startJupyter, allowedCudaVersions.
    => ports: ["22/tcp"] REPLACES the ComfyUI template's ["8188/http","8080/http","8888/http","22/tcp"].
       Port 8188 is therefore not publicly exposed BY DESIGN. Task 5 Step 7 still tests it, but the
       expected answer is now known rather than hoped for.
    => body mounts.network overrides the template's persistent mount, so the "at-most-one of
       persistent/network, else 400" rule is not tripped.
  - "Sending either CUDA field (gpu.allowedCudaVersions or gpu.minCudaVersion) replaces the
    template's CUDA constraint entirely." => create_pod must NOT send either, so the template's own
    constraint (12.8/12.9 vs 13.x) drives host selection and the CUDA line cannot be mismatched.
    Current create_pod sends neither. Correct as-is; carry this into Task 5's dispatch as a do-not.
  - "unknown or inaccessible ID -> 404" confirms 2lv7ev3wfp is genuinely gone, not merely private:
    the two live IDs resolve fine for this same key.
  - Template is a one-time source of settings; created pod keeps no link (`template` stays null).

*** MEASURED: 124.4 MB/s (995 Mbps) on pod 5xnsizikxzjrpi, EU-RO-1, cpu5c, HF -> Runpod. ***
  Method: du -sb on the models dir, two samples 60s apart: 7,515,332,608 -> 14,978,181,024 bytes
  = 7.46 GB in 60s. Plain `curl`, no hf-xet, no tuning, single stream per file, sequential.

  R12 (storage conclusion inverted — my recommendation was built on a bad median).
  I argued for the volume from a ~15 MB/s median drawn from RunPod's support Discord, a
  self-selected complaints channel. This account gets 124 MB/s — 8x that, and near the top of the
  whole observed 0.3-120 MB/s range. Consequences:
    - 48.7 GB downloads in ~7 min, not the ~54 min I told the user.
    - A per-session re-download costs ~7 min x $0.72/hr = ~$0.08 of GPU.
    - The 75 GB volume costs $5.25/month. Break-even is therefore ~65 sessions/month, not the
      ~5 I stated. For 2-3 sessions/week the volume LOSES on money.
  Ruling: keep the volume for the remainder of this build (Tasks 5/6/8 reuse it, and re-downloading
  three times would cost more wall-clock than it saves), but the Task 10 recommendation flips from
  "keep it, it pays for itself" to "delete it — volume-less is cheaper at your usage". This also
  vindicates the user's original instinct, which I talked them out of on bad data. Must be stated
  plainly to the user, not buried. create_pod already supports volume_id=None (R4), so the
  volume-less path is already built and tested.
  Cost if wrong: if a future host draws 1-3 MB/s instead, a session starts with a 4-14 hour
  download. Mitigation: `comfy provision` is re-runnable, so the volume can be recreated any time.

  Caveat that keeps this honest: throughput is a documented per-host lottery. One excellent draw
  does not make 124 MB/s a planning figure. What it does establish is that the low tail is not
  universal on this account/datacenter.
Task 4 LIVE RESULT: SUCCESS. Volume g3h3d0v2ee (comfy-models, 75 GB, EU-RO-1) created and populated
  with all 7 files, 48.8 GB verified by SSH. CPU pod 5xnsizikxzjrpi terminated; GET /v2/pods empty,
  pod ID 404s. Account clean apart from the volume. Billing still reports $0 (lags).

  Process failure worth recording: the Task 4 implementer completed the live work but then entered a
  no-op wait loop, returning "waiting for events" ~6 times without finishing, burning ~193k tokens
  and never committing. I stopped it with TaskStop, verified the world state myself over SSH and the
  API, terminated nothing (its finally block had already done so), and dispatched a fresh implementer
  to commit and report. Lesson for later live tasks: give the implementer an explicit completion
  condition and forbid open-ended waiting.

  **R13 (out-of-scope change ENDORSED, not creep).** The implementer added
  `User-Agent: comfy-runpod/0.1.0` to `_http` plus a direct test. Runpod's Cloudflare edge returns
  403 "error code: 1010" against urllib's default identity, so every API call fails without it.
  My own pre-flight curl checks never caught this because curl always sends a User-Agent; urllib
  sends Python-urllib/3.12, which is blocked. Discoverable only by running live — exactly what the
  live step is for. Ruling: keep. Cost if wrong: none; an explicit UA is strictly correct.
Task 4: implementer (finishing agent) DONE, commit 91972a8, 46 tests, comfy.yaml gitignored+untracked.
Task 4: review — Spec COMPLIANT (1 undisclosed omission). Quality APPROVED, 0 Critical, 5 Important.
  Reviewer confirmed by reading the rendered script (not just the test) that a short file really does
  `exit 1`, and that the bare `finally` survives KeyboardInterrupt. Cleanup logic is sound.

  **R14 (spec omission ruled in the implementer's favour).** The brief named
  `scripts/download-models.sh` as a deliverable; it was never created — the script is rendered in
  memory by `render_download_script()` and piped over SSH. Ruling: the omission is CORRECT and the
  plan was wrong. A static copy on disk would duplicate the renderer and drift out of sync, and the
  byte thresholds are baked per-render. Removed from the plan's file list rather than creating a
  redundant file. Cost if wrong: none; the renderer is the single source of truth.

  **R15 (duplicate-volume money risk — fixing).** If create_volume's POST lands but the response is
  lost, the volume exists, no id was printed, and a retry cannot tell -> a silent second $5.25/month
  volume. Ruling: fix now rather than defer, since it is a money leak and the guard is ~5 lines.
  `Client.list_volumes()` (which Task 10 needed anyway) plus a name lookup for `comfy-models` before
  creating. Cost if wrong: one extra GET per provision.
Task 4: minor (deferred): cmd_provision catches ConfigError/RunpodError locally rather than in main.
  Carry into Task 5 — with a second handler appearing there, centralising in `main` becomes correct.
Task 4: minor (deferred): test_script_mentions_every_model_file is a render round-trip against its
  own input — real but weak.
Task 4: fix round 1 dispatched (resumed finishing implementer, sonnet): provision() control-flow
  tests, list_volumes guard, real size-check assertion, honest test names.
Task 4: fix round 1/5 (4 addressed per implementer, re-review pending; commits 91972a8..ef55fee, 54 tests)
Task 5 pre-flight: EU-RO-1 stock has FLIPPED since the spec was written.
  RTX PRO 4500 Blackwell 32GB $0.72/hr -> HIGH (was MEDIUM)
  RTX GeForce 4090      24GB $0.74/hr -> MEDIUM (was HIGH)
  The primary path is now the better-stocked one. Also empirical proof that the ordered GPU list in
  config was the right call rather than over-engineering: stock genuinely moves within hours.
Task 4: fix round 2/5 (2 addressed, 0 open; commits ef55fee..3102add, 55 tests)
Task 4: complete (commits 4ff9249..3102add, review clean after 2 fix rounds)

=== USER DIRECTIVE (2026-09-29, supersedes per-task money gates) ===
  "Run all the steps which do not involve my interaction. Ensure no billing stands in the end, and
   report what's left to do to another agent in ./docs/HANDOFF.md."
  Consequences:
  - Tasks 5, 8, 9 run without further approval. Volume DELETION at Task 10 is now explicitly
    authorised ("ensure no billing stands in the end") — this is the destructive act I had reserved.
  - **R16 (Task 6 becomes headless).** Task 6 as planned needs a human in the browser for
    File -> Export (API). Ruling: author the five API-format graphs directly instead. API format is
    just {id: {class_type, inputs, _meta}}, all five modes are core-node-only, and Task 8 validates
    them by actually running them — a graph that is wrong will fail loudly rather than silently.
    Cost if wrong: a graph needs hand-repair, and the handoff says which.
  - **R17 (one pod session, not three).** Tasks 5, 6 and 8 each wanted their own pod. Ruling: boot
    once and do all three in that session. Saves ~2 boots of pod time and ~8 min of first-boot wait.
    Cost if wrong: a failure mid-session means re-booting to continue.
  - **R18 (centralise error handling in main).** Reviewer flagged that cmd_provision catches
    ConfigError/RunpodError locally; Task 5 adds three more handlers. Ruling: move the try/except
    into `main` now and drop the local one, since Task 5 already edits cli.py. Cost if wrong: a
    handler that wants custom handling has to opt out.
Task 5: implementer DONE (offline only, no pod), commit 29fa987, 86 tests (55 + 31 new).
  Three disclosed deviations: (1) tunnel detachment uses start_new_session=True and stderr=DEVNULL
  instead of the brief's PIPE, reasoning a PIPE's read end closing on parent exit would SIGPIPE the
  detached ssh; (2) cmd_up calls pick_gpu twice rather than invent a schema field; (3) an early
  partial write_state right after pod creation so a tunnel/readiness failure cannot leak an
  untracked, still-billing pod. (3) is a genuine safety improvement over the brief.
Task 5: review dispatched (sonnet), with an explicit ask for a live-check list since tunnel/ssh
  behaviour is not verifiable from a diff.
Task 5: review — Spec COMPLIANT. Quality: 1 CRITICAL, 3 Important, 2 Minor.
  **R19 (widened catch tuple ENDORSED).** main catches (ConfigError, RunpodError, TunnelError,
  ComfyUIError) vs the instructed two. Serves the refactor's actual purpose — a TunnelError reaching
  the user as a traceback is the bug the refactor existed to kill. Cost if wrong: none.
  **CRITICAL (pod leak, pre-live).** _bring_up's 900s ssh_target wait sits between create_pod and
  cmd_up's write_state, so a timeout loses the pod id entirely and `comfy down` reports "nothing
  running" while a $0.72/hr pod bills on. Deviation #3 aimed at exactly this and missed by one call
  frame. Reviewer confirmed neither existing test covers it. Must be fixed BEFORE the live run,
  because the next boot is the first ever of CUDA 13 template wgd3p4n4o6 — the most likely timeout.
  Also fixing: double pick_gpu can display one GPU while provisioning another (stock moved twice
  today, so this is live not theoretical); stderr=DEVNULL makes every ssh failure indistinguishable
  (ssh exits 255 for all of them) -> redirect to a file instead; _kill_tunnel signals a bare pid
  with no identity check -> verify /proc/<pid>/cmdline before killing.
Task 5: fix round 1 dispatched (resumed implementer, sonnet). Reviewer also left a 5-item live-check
  list for the pod session: real detach survival across terminal close, /system_stats key names on
  this template, whether billing reports spend for a RUNNING pod, whether the ssh-timeout path is
  reachable in practice, and pid-recycling risk.
Task 5: fix round 1/5 (4 addressed, 0 open; commits 29fa987..66209ad, 95 tests). Re-reviewer traced
  the Critical fix statement-by-statement: write_state now sits inside _bring_up immediately after
  pod_id is assigned, with only a str() call between, so the leak window is closed not moved.
Task 5: fix round 2 dispatched — _pid_cmdline conflates "pid gone" with "no /proc", so _kill_tunnel
  would silently stop killing tunnels on any non-Linux platform. No impact on this NixOS box.
Task 5: minor (deferred): config.write_state is a bare write_text, not atomic.
Task 5: minor (deferred): _bring_up's state write clobbers a prior un-downed pod's record.
Task 5: fix round 2 (1 addressed; commit b44e58a, 96 tests).
  **R20 (deliberate deviation from the loop).** I verified this round by reading the 4-line diff and
  its test directly rather than dispatching a formal scoped re-review. Reason: the change is small
  enough to hold entirely in view, it only alters behaviour on platforms without /proc (this machine
  has it, so live risk is nil), and the review budget is better spent on the live session. The diff
  does exactly what was asked: _proc_available() gate, warn-and-kill when absent, silent when the
  pid is merely gone, warn-and-skip on cmdline mismatch. Cost if wrong: a non-Linux user sees one
  extra warning line.
Task 5: complete (commits 3102add..b44e58a, review clean after 2 fix rounds)
=== LIVE SESSION BEGINS: one pod for Tasks 5-verify + 6 + 8, then teardown incl. volume. ===

=== LIVE SESSION — Task 5 verification results (pod 6x4tgl99bzh6kt, RTX PRO 4500 Blackwell) ===
  1. FIRST EVER live boot of CUDA 13 template wgd3p4n4o6: SUCCESS.
     ComfyUI 0.30.0, torch 2.10.0+cu130, python 3.12.3, 31 GB VRAM reported.
  2. SSH tunnel over direct TCP: works. http://127.0.0.1:8188 serves /system_stats.
  3. *** SECURITY REQUIREMENT VERIFIED: https://<pod>-8188.proxy.runpod.net returns 404. ***
     ports:["22/tcp"] genuinely overrode the template's ["8188/http","8080/http","8888/http","22/tcp"],
     confirming the schema's "explicit body fields override the template's" in practice.
  4. All 7 models on the volume are visible to ComfyUI via /object_info. Volume mount works.
  5. 1071 node types available; zero custom nodes installed.

  **R21 (GPU stock is far more volatile than the spec assumed).** Within ~90 minutes EU-RO-1 went:
  Blackwell MEDIUM->HIGH->LOW, 4090 HIGH->MEDIUM->NONE. The first `up` attempt failed outright with
  "no configured GPU has capacity"; a retry minutes later succeeded on the same Blackwell. Ruling:
  added RTX 5090 (32 GB, $0.99/hr, also CUDA 13 -> template wgd3p4n4o6) as a third entry in
  comfy.yaml's ordered list. The ordered-list design is vindicated — this is not over-engineering.
  Cost if wrong: an extra catalog entry that is never used.

  **Usability gap found (for HANDOFF):** `comfy` is NOT on PATH inside `nix develop` — the devShell
  exposes the package via PYTHONPATH only, so the console script exists solely in the built package
  (`nix run .#`). Every live command in the plan says `comfy ...` but must be
  `python -m comfy_runpod.cli ...` inside the dev shell. Worth fixing in the flake.

Task 6: dispatched (sonnet) against the live tunnel — author + validate 5 API-format graphs.
Task 6: COMPLETE, commit 95b6cbf. All 5 modes authored headlessly AND validated end-to-end against
  the live pod with real outputs: t2i_00001_.png, i2i_00002_.png, t2v/i2v/v2v .mp4 (512x288, 3.06s).
  R16 (headless authoring instead of browser export) is vindicated — no human was needed.
  Three findings that would have broken Task 7's planned code, caught because Task 7 was not yet
  written:
    F1: text input is named `text` on CLIPTextEncode but `prompt` on TextEncodeZImageOmni.
    F2: LoadVideo's file input is `file`, not `image` (that assumption holds only for LoadImage).
    F3: CLIPLoader.type has no "z_image" literal; "qwen_image" is correct for qwen_3_4b, confirmed
        empirically rather than from docs.
  **R22.** Task 7 must patch *the key already present* in the node's inputs (candidates
  ("text","prompt") and ("image","file","video")) rather than use a class->field lookup table.
  A table goes stale when a node is swapped; probing the existing key does not. Raise WorkflowError
  when no candidate is present, because a prompt that silently fails to reach the graph yields a
  plausible image of the wrong thing — the worst failure mode available.
  i2i caveat from the agent: the edit prompt barely registered even at denoise 0.75. Graph is sound;
  this is generic img2img hue-resistance. A better "omni" conditioning path is noted for later.
Task 7: dispatched (sonnet) with F1/F2 as explicit corrections to its brief. Pod deliberately kept
  UP during this because EU-RO-1 stock is LOW and a teardown might not get a GPU back.
Task 7: implementer DONE, commit 7a63097, 129 tests. **My error: I never ran task-brief for Task 7**,
  so no brief file existed. The agent proceeded from the inlined corrections plus this ledger's
  F1/F2/R22 entries and validated against the five real graphs — a good recovery, and it flagged the
  missing brief rather than inventing requirements.
  **R23 (two interface changes kept, one reverted).** The agent changed three signatures:
    load(path) instead of load(mode, directory)        -> KEPT, cleaner.
    find_by_title -> node dict instead of node id str  -> KEPT, genuinely better.
    apply_run(workflow, run) dropping seed/uploaded_name -> REVERTED, it is a functional regression.
  Why the revert: apply_run set the sampler seed from run.seed and only when non-None. run.seed None
  means "random" and is the default, so seeds_for()'s per-image seeds were never applied and a
  16-image batch would return 16 IDENTICAL images — the single most common thing this tool does.
  Second break: it patched run.input.name (the LOCAL filename) rather than the server-side name
  returned by /upload/image, which can differ by dedup or subfolder.
  Cost if wrong: a slightly wider signature than the agent preferred.
  Caught only because I ran the module against the real graphs myself before letting Task 8 depend
  on it — the unit tests all passed, because they encoded the new shape.
Task 7: fix round 1 dispatched. Pod still up and billing during this (~$0.12) — cheaper than a
  teardown that might not get a GPU back at LOW stock.
Task 7: fix round 1 (2 Critical addressed; commit 9cf05eb, 130 tests). Sentinel check across all five
  real graphs: t2i/i2i hit 'prompt', t2v/i2v/v2v hit 'text', seed_ok True everywhere. Exactly the
  per-node-class divergence F1 predicted, now handled by probing rather than a lookup table.
Task 7: complete (commits b44e58a..9cf05eb, verified against real graphs; formal review deferred to
  the final whole-branch review since the pod was billing).
Task 8: dispatched (sonnet), live, with --keep mandated so it cannot terminate my session pod.
Task 8: COMPLETE, commit 3ef658b, 153 tests. All 3 live validations PASSED:
  (1) bad mode fails before any pod work, state byte-identical; (2) t2i count=3 -> 3 images with
  DIFFERENT sha256 (proves the R23 seed fix), visually a real fox not noise; (3) i2v -> real h264
  512x288 49 frames 3.06s via upload -> server-name round-trip -> apply_run.
  The implementer caught two more plan bugs before they cost pod time: the pseudocode always opened
  a fresh tunnel (would have failed, 8188 already bound) and queued every job before downloading any
  (violating "job 7 fails, jobs 1-6 already on disk"). Rewrote sequentially.
  My process error, third time: I never ran task-brief for Tasks 6, 7 or 8. All three recovered by
  working from the dispatch prompt + plan + this ledger, and all three flagged it. No harm done but
  it made each dispatch longer than it needed to be.

=== TEARDOWN COMPLETE (user directive: "ensure no billing stands in the end") ===
  `comfy down` live-tested: terminated pod 6x4tgl99bzh6kt, reported real spend $0.21 from
  /v2/billing/pods, cleared state, killed the tunnel.
  Volume g3h3d0v2ee DELETED (DELETE /v2/network-volumes/<id> -> 204). 48.8 GB of models gone,
  irreversibly, as instructed.
  FINAL: pods [] volumes [] endpoints []. Nothing bills. Total spend for the whole build ~$0.29.
Task 10: dispatched (sonnet) — `comfy teardown` + the flake fix putting `comfy` on the devShell PATH.
docs/HANDOFF.md written for a contextless agent. CLAUDE.md and README.md written.
