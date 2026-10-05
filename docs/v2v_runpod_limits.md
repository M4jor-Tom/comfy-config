# v2v limits on Runpod

**Measured on 2026-10-05.** Applies to the current `workflows/v2v.json` and
ComfyUI 0.30.0. The short preview succeeded on a real GPU, as confirmed by the
user. Larger input limits below are calculations, not GPU benchmark results.

## Actual configuration

| Property | Observed value or current setting |
| --- | --- |
| GPU inspected | NVIDIA RTX PRO 4500 Blackwell |
| GPU VRAM | 32 GB |
| Pod system RAM hard limit | 62 GB decimal (57.74 GiB), verified through `memory.max` |
| System RAM occupied during inspection | Approximately 14.9 GB |
| Workflow output dimensions | 512×288 |
| Workflow output frames | 49 |
| Workflow output frame rate | 16 FPS |
| Workflow output duration | 3.0625 seconds |
| Batch size / sampler steps | 1 / 30 |
| ComfyUI upload limit | Approximately 100 MiB, including multipart overhead; default `--max-upload-size=100`, with no startup override |
| Harness upload timeout | 300 seconds (5 minutes) |
| Harness generation timeout | 1,800 seconds (30 minutes) per job |

The timeouts measure upload/render wall-clock time, **not video playback duration**.
The separate `terminate_after: 3h` infrastructure setting does not change them.

`comfy.yaml` also permits RTX 5090 (32 GB VRAM) and RTX 4090 (24 GB VRAM)
fallbacks. **System RAM is allocated by the selected host and can differ between
pods.** The harness currently requests only GPU type/count, not a minimum RAM
allocation. The inspected pod's 62 GB is not a guarantee for the next run.

In Runpod REST v2, a pod's `gpu.memory` means allocated **system RAM**, whereas
the GPU catalog's `memory` means **VRAM**. Inside a pod, inspect cgroup memory
limits rather than assuming `/proc/meminfo` describes the container's allowance.

## Why the original input failed

The original input was 1024×1024, approximately 30 FPS, 7,107 frames, and
236.89853 seconds long. Its compressed file size was only 92.9 MB.

The workflow sends `LoadVideo` directly into `GetVideoComponents`, which decodes
**the entire source video** into an in-memory frame batch. The Wan node's
`length: 49` truncates that batch **after decoding**, so it does not cap input
memory consumption.

For this RGB float32 decoding path:

```text
one decoded batch bytes ≈ width × height × frame_count × 12
peak decoding bytes    ≈ width × height × frame_count × 24
```

The second allocation comes from stacking the individual frame tensors. Audio,
model weights, caches, and workspaces consume additional memory; other pixel
formats or alpha channels can change the estimate.

The original input therefore needed approximately **89.4 GB (83.3 GiB)** for one
batch, or **178.9 GB (166.6 GiB)** at decoding peak. This supports RAM exhaustion
as the cause of the disconnect; the terminated failing pod's kernel/server logs
were unavailable, so an OOM kill was not directly confirmed.

## Estimated input ceilings — not verified safe maxima

There is no independent maximum duration, resolution, or FPS: their product
sets the frame-memory requirement. For constant-frame-rate video:

```text
frame_count ≈ duration_seconds × FPS

decoder_budget = RAM_limit − RAM_in_use − reserve
estimated_max_frames = floor(decoder_budget / (24 × width × height))
```

Using the inspected pod's snapshot and reserving another **20% of total RAM**:

```text
62.0 GB − 14.9 GB − 12.4 GB ≈ 34.7 GB decoder budget
```

| Input dimensions | Estimated frame ceiling | Duration at 16 FPS | Duration at 30 FPS |
| --- | ---: | ---: | ---: |
| 512×288 | 9,805 | 612.8 s | 326.8 s |
| 1024×1024 | 1,378 | 86.1 s | 45.9 s |
| 1280×720 | 1,568 | 98.0 s | 52.3 s |
| 1920×1080 | 697 | 43.6 s | 23.2 s |
| 3840×2160 | 174 | 10.9 s | 5.8 s |

**Do not treat these values as guaranteed safe boundaries.** They estimate only
decoder RAM for one observed host and memory snapshot. The reserve is a planning
margin, not a measured maximum. Upload size, additional allocations, different
host RAM, and render time can impose lower limits. Larger output dimensions or
frame counts also increase GPU memory demand, which this table does not model.

For variable-frame-rate recordings, use actual frame counts rather than
`r_frame_rate`; the original recording reported a misleading `100000/1` there.
Compression and bitrate affect upload size, not decoded frame-memory size.

## Verified working input and recommendation

For the unchanged workflow, prepare inputs matching the successful preview:

| Property | Recommended value |
| --- | --- |
| Dimensions | 512×288 |
| Frames | 49 |
| Frame rate | 16 FPS |
| Duration | 3.0625 seconds |
| Encoding | MP4, H.264, `yuv420p` |
| Audio | None |
| File size | Below 95 MiB, leaving upload-overhead margin |

The verified preview was approximately 55 KB; the 95 MiB recommendation is an
upload margin, not a tested file-size maximum. Letterbox rather than crop when
preserving the source camera framing matters.

**Longer inputs currently add decoding cost without producing longer output:**
only the first 49 source frames reach generation. Source FPS is not automatically
resampled: 49 frames at 30 FPS represent about 1.63 seconds of source motion but
are saved as 3.06 seconds at the workflow's 16 FPS.

The YAML `size:` field **does not resize this v2v workflow or its source video**.
The harness patches size only on a node titled `latent`; this graph has none.
Its dimensions and frame count remain those set on `Wan22FunControlToVideo`.

Longer generated clips require workflow changes. Whole-movie conversion requires
chunking. Core `Video Slice` (displayed as **Trim Video**) can bound decoding when
placed before `GetVideoComponents`; slicing an already-decoded image batch is too
late. Time trimming alone does not resample source FPS.

## Verification and sources

Inspect a local input without creating a paid resource:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,avg_frame_rate,nb_frames,duration,pix_fmt:format=size \
  -of json INPUT.mp4
```

If `nb_frames` is unavailable, add `-count_frames` and request `nb_read_frames`;
this scans the video and can take longer.

An empirical safe operating envelope requires controlled tests recording cgroup
`memory.max`, `memory.current`, `memory.peak`, and `memory.events`, plus GPU usage
and render time. Obtain explicit approval before renting hardware or running paid
benchmarks. Do not infer a larger generation limit from the decoder-only table.

Sources:

- [Current workflow](../workflows/v2v.json): Wan dimensions/length, sampler, and output FPS.
- [Workflow patching](../src/comfy_runpod/workflow.py): `apply_run` patches size only on `latent`.
- [ComfyUI client](../src/comfy_runpod/comfyui.py): upload and per-job timeouts.
- [Runpod client](../src/comfy_runpod/runpod_api.py): GPU creation request.
- [Runpod REST v2 live schema](https://api.runpod.io/v2/openapi.json): `GpuConfig.memory` and `CreateGpuConfig.minRamPerGpu`.
- [ComfyUI 0.30.0 video nodes](https://github.com/Comfy-Org/ComfyUI/blob/v0.30.0/comfy_extras/nodes_video.py): `GetVideoComponents` and `VideoSlice`.
- [ComfyUI 0.30.0 decoder](https://github.com/Comfy-Org/ComfyUI/blob/v0.30.0/comfy_api/latest/_input_impl/video_types.py): `VideoFromFile.get_components_internal`.
- [ComfyUI 0.30.0 CLI arguments](https://github.com/Comfy-Org/ComfyUI/blob/v0.30.0/comfy/cli_args.py): default upload limit.
- Live inspection: cgroup RAM limit `61,999,996,928` bytes and usage
  `14,897,369,088` bytes; ComfyUI `/system_stats` and startup arguments.
