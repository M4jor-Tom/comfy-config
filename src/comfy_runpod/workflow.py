"""Patch a committed API-format ComfyUI graph with a run's parameters.

A workflow is the raw `{node_id: {class_type, inputs, _meta}}` mapping
ComfyUI's `/prompt` endpoint expects. Nodes are located by `_meta.title`
(authored once in `workflows/*.json`), never by id or class_type, so the
graphs can be edited in the ComfyUI UI without breaking this module.

Two input fields vary by node class rather than by title, so they cannot be
set by a fixed key: the prompt text (`text` on CLIPTextEncode, `prompt` on
TextEncodeZImageOmni) and the uploaded input file (`image` on LoadImage,
`file` on LoadVideo). Both are patched by probing which candidate key is
already present on that node -- never by a class_type lookup table, which
would go stale the moment a node is swapped. A title whose node has none of
the candidates raises rather than silently doing nothing: a prompt that
never reaches the graph produces a plausible-looking image of the wrong
thing.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

from .config import Run

_TEXT_FIELD_CANDIDATES = ("text", "prompt")
_INPUT_FIELD_CANDIDATES = ("image", "file", "video")
_SEED_MAX = 2**32 - 1


class WorkflowError(Exception):
    """A workflow graph is malformed, or a run's parameters don't fit it."""


def load(path: Path) -> dict:
    """Parse an API-format ComfyUI graph from disk."""
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError as e:
        raise WorkflowError(f"{path} does not exist") from e
    except json.JSONDecodeError as e:
        raise WorkflowError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise WorkflowError(f"{path} must contain a mapping of node id -> node")
    return raw


def find_by_title(workflow: dict, title: str) -> dict | None:
    """The single node whose `_meta.title` equals `title`, or None.

    Raises WorkflowError if more than one node shares the title: a run
    parameter would otherwise patch an arbitrary one of them.
    """
    matches = [
        (node_id, node)
        for node_id, node in workflow.items()
        if isinstance(node, dict) and node.get("_meta", {}).get("title") == title
    ]
    if len(matches) > 1:
        ids = ", ".join(node_id for node_id, _ in matches)
        raise WorkflowError(f"multiple nodes titled {title!r}: {ids}")
    return matches[0][1] if matches else None


def _require(workflow: dict, title: str) -> dict:
    node = find_by_title(workflow, title)
    if node is None:
        raise WorkflowError(f"workflow has no node titled {title!r}")
    return node


def _set_candidate(node: dict, candidates: tuple[str, ...], value: object) -> None:
    """Set the first of `candidates` already present as a key in the node's
    `inputs`. Raises WorkflowError naming the node (by its own title), its
    class_type, and the keys it actually has if none of them is -- a table
    of class_type -> field would go stale the moment a node is swapped;
    this probes the graph itself instead.
    """
    inputs = node.get("inputs", {})
    for field in candidates:
        if field in inputs:
            inputs[field] = value
            return
    title = node.get("_meta", {}).get("title")
    raise WorkflowError(
        f"node {title!r} (class_type={node.get('class_type')!r}) has none of "
        f"{candidates} in its inputs; found {sorted(inputs)}"
    )


def apply_run(workflow: dict, run: Run) -> dict:
    """A patched copy of `workflow` with `run`'s parameters applied.

    Never mutates `workflow`. `positive`/`negative`/`sampler` are required;
    `latent` is patched for size only when present (i2i and i2v have none —
    their size is either fixed by the input image or already baked into a
    non-`latent`-titled node); `input` is patched only when `run.input` is
    given, and its absence from the graph then is an error rather than a
    silently ignored file.
    """
    out = copy.deepcopy(workflow)

    positive = _require(out, "positive")
    _set_candidate(positive, _TEXT_FIELD_CANDIDATES, run.prompt)

    negative = _require(out, "negative")
    _set_candidate(negative, _TEXT_FIELD_CANDIDATES, run.negative)

    sampler = _require(out, "sampler")
    if run.seed is not None:
        sampler["inputs"]["seed"] = run.seed

    latent = find_by_title(out, "latent")
    if latent is not None:
        latent["inputs"]["width"], latent["inputs"]["height"] = run.size

    if run.input is not None:
        input_node = find_by_title(out, "input")
        if input_node is None:
            raise WorkflowError(
                f"run supplies an input file ({run.input.name!r}) but the "
                "workflow has no node titled 'input'"
            )
        _set_candidate(input_node, _INPUT_FIELD_CANDIDATES, run.input.name)

    return out


def apply_overrides(workflow: dict, overrides: dict[str, object]) -> dict:
    """A patched copy of `workflow` with `{"title.field": value}` overrides
    applied verbatim -- the caller names the exact field, so no candidate
    probing is needed here. Never mutates `workflow`.
    """
    out = copy.deepcopy(workflow)
    for key, value in overrides.items():
        title, sep, field = key.partition(".")
        if not sep:
            raise WorkflowError(f"override key {key!r} must be 'title.field'")
        node = find_by_title(out, title)
        if node is None:
            raise WorkflowError(f"override {key!r}: no node titled {title!r}")
        node.setdefault("inputs", {})[field] = value
    return out


def seeds_for(run: Run) -> list[int]:
    """`run.count` concrete seeds.

    A fixed seed increments deterministically so a batch is reproducible
    without every image being identical; 'random' (`run.seed is None`)
    draws that many distinct seeds.
    """
    if run.seed is not None:
        return [run.seed + i for i in range(run.count)]
    return random.sample(range(_SEED_MAX + 1), run.count)
