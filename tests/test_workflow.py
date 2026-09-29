import copy
import json
from pathlib import Path

import pytest

from comfy_runpod.config import Run
from comfy_runpod.workflow import (
    WorkflowError,
    apply_overrides,
    apply_run,
    find_by_title,
    load,
    seeds_for,
)

WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / "workflows"


def _run(**kw):
    """A Run with sane defaults; a test only passes what it varies. Mirrors
    T2's Run fields -- seed/input/overrides are left at their dataclass
    defaults (None/None/{}) unless a test cares about them."""
    defaults = dict(mode="t2i", prompt="a fox in a misty forest", count=1, size=(512, 512))
    defaults.update(kw)
    return Run(**defaults)


def _node(class_type, title, inputs):
    return {"class_type": class_type, "inputs": dict(inputs), "_meta": {"title": title}}


def _full_workflow(**node_overrides):
    """positive/negative/sampler/latent, in the shape the pre-corrected brief
    assumed (CLIPTextEncode, `text`). Individual tests override one node to
    exercise a different class/field shape."""
    nodes = {
        "positive": _node("CLIPTextEncode", "positive", {"text": "old positive"}),
        "negative": _node("CLIPTextEncode", "negative", {"text": "old negative"}),
        "sampler": _node("KSampler", "sampler", {"seed": 0, "steps": 30}),
        "latent": _node("EmptySD3LatentImage", "latent", {"width": 512, "height": 512}),
    }
    nodes.update(node_overrides)
    return {str(i): node for i, node in enumerate(nodes.values(), start=1)}


# --- load ---------------------------------------------------------------


def test_load_parses_valid_workflow(tmp_path):
    p = tmp_path / "wf.json"
    p.write_text(json.dumps({"1": {"class_type": "X", "inputs": {}, "_meta": {"title": "a"}}}))
    wf = load(p)
    assert wf["1"]["class_type"] == "X"


def test_load_raises_on_missing_file(tmp_path):
    with pytest.raises(WorkflowError, match="does not exist"):
        load(tmp_path / "nope.json")


def test_load_raises_on_invalid_json(tmp_path):
    p = tmp_path / "wf.json"
    p.write_text("{not valid json")
    with pytest.raises(WorkflowError, match="not valid JSON"):
        load(p)


def test_load_raises_on_non_mapping_top_level(tmp_path):
    p = tmp_path / "wf.json"
    p.write_text("[1, 2, 3]")
    with pytest.raises(WorkflowError, match="mapping"):
        load(p)


# --- find_by_title --------------------------------------------------------


def test_find_by_title_returns_matching_node():
    wf = _full_workflow()
    node = find_by_title(wf, "sampler")
    assert node is not None
    assert node["class_type"] == "KSampler"


def test_find_by_title_returns_none_when_absent():
    wf = _full_workflow()
    assert find_by_title(wf, "nonexistent") is None


def test_find_by_title_raises_on_duplicate_titles():
    wf = {
        "1": _node("CLIPTextEncode", "positive", {"text": "a"}),
        "2": _node("CLIPTextEncode", "positive", {"text": "b"}),
    }
    with pytest.raises(WorkflowError, match="multiple nodes titled 'positive'"):
        find_by_title(wf, "positive")


# --- apply_run: prompt / negative / seed / latent size ------------------


def test_apply_run_sets_positive_and_negative_text():
    wf = _full_workflow()
    out = apply_run(wf, _run(prompt="new positive", negative="new negative"))
    assert find_by_title(out, "positive")["inputs"]["text"] == "new positive"
    assert find_by_title(out, "negative")["inputs"]["text"] == "new negative"


def test_apply_run_sets_latent_size_when_present():
    wf = _full_workflow()
    out = apply_run(wf, _run(size=(768, 1024)))
    node = find_by_title(out, "latent")
    assert (node["inputs"]["width"], node["inputs"]["height"]) == (768, 1024)


def test_apply_run_tolerates_missing_latent_node():
    """i2i has no 'latent' node -- its size comes from the input image."""
    wf = {
        "1": _node("TextEncodeZImageOmni", "positive", {"prompt": "p"}),
        "2": _node("TextEncodeZImageOmni", "negative", {"prompt": "n"}),
        "3": _node("KSampler", "sampler", {"seed": 0}),
    }
    out = apply_run(wf, _run())  # must not raise
    assert find_by_title(out, "latent") is None


def test_apply_run_sets_sampler_seed_when_given():
    wf = _full_workflow()
    out = apply_run(wf, _run(seed=12345))
    assert find_by_title(out, "sampler")["inputs"]["seed"] == 12345


def test_apply_run_leaves_seed_untouched_when_none():
    wf = _full_workflow()
    out = apply_run(wf, _run(seed=None))
    assert find_by_title(out, "sampler")["inputs"]["seed"] == 0  # authored value, left alone


def test_apply_run_does_not_mutate_the_input_workflow():
    wf = _full_workflow()
    original = copy.deepcopy(wf)
    apply_run(wf, _run(prompt="different", seed=99, size=(999, 999)))
    assert wf == original


@pytest.mark.parametrize("missing_title", ["positive", "negative", "sampler"])
def test_apply_run_raises_when_a_required_title_is_missing(missing_title):
    nodes = {
        "positive": _node("CLIPTextEncode", "positive", {"text": "p"}),
        "negative": _node("CLIPTextEncode", "negative", {"text": "n"}),
        "sampler": _node("KSampler", "sampler", {"seed": 0}),
    }
    del nodes[missing_title]
    wf = {str(i): node for i, node in enumerate(nodes.values(), start=1)}
    with pytest.raises(WorkflowError, match=f"no node titled '{missing_title}'"):
        apply_run(wf, _run())


# --- apply_run: text field varies by node class (correction 1) ----------


def test_apply_run_zimageomni_positive_sets_prompt_not_text():
    """TextEncodeZImageOmni (t2i/i2i) keeps its text in `prompt`; patching
    it must not add a spurious `text` key."""
    wf = _full_workflow(positive=_node("TextEncodeZImageOmni", "positive", {"prompt": "old", "clip": ["2", 0]}))
    out = apply_run(wf, _run(prompt="new prompt"))
    node = find_by_title(out, "positive")
    assert node["inputs"]["prompt"] == "new prompt"
    assert "text" not in node["inputs"]


def test_apply_run_cliptextencode_positive_sets_text():
    """CLIPTextEncode (t2v/i2v/v2v) keeps its text in `text`."""
    wf = _full_workflow(positive=_node("CLIPTextEncode", "positive", {"text": "old", "clip": ["2", 0]}))
    out = apply_run(wf, _run(prompt="new prompt"))
    node = find_by_title(out, "positive")
    assert node["inputs"]["text"] == "new prompt"
    assert "prompt" not in node["inputs"]


def test_apply_run_raises_when_positive_has_neither_candidate_field():
    """No silent no-op: a node that has neither `text` nor `prompt` must
    fail loudly, naming its class_type, rather than leave the prompt
    unset."""
    wf = {"1": _node("MysteryNode", "positive", {"unrelated_field": "value"})}
    with pytest.raises(WorkflowError, match="MysteryNode"):
        apply_run(wf, _run())


# --- apply_run: input file field varies by node class (correction 2) ----


def test_apply_run_sets_loadimage_input_field():
    wf = _full_workflow(input=_node("LoadImage", "input", {"image": "placeholder.png"}))
    out = apply_run(wf, _run(mode="i2i", input=Path("/local/dir/cat.png")))
    assert find_by_title(out, "input")["inputs"]["image"] == "cat.png"


def test_apply_run_sets_loadvideo_input_field():
    """LoadVideo's upload-target field is `file`, not `image`."""
    wf = _full_workflow(input=_node("LoadVideo", "input", {"file": "placeholder.mp4"}))
    out = apply_run(wf, _run(mode="v2v", input=Path("/local/dir/clip.mp4")))
    assert find_by_title(out, "input")["inputs"]["file"] == "clip.mp4"


def test_apply_run_raises_when_input_given_but_no_input_node():
    wf = _full_workflow()  # t2i-shaped: no 'input' node
    with pytest.raises(WorkflowError, match="no node titled 'input'"):
        apply_run(wf, _run(input=Path("/local/dir/cat.png")))


# --- apply_overrides ------------------------------------------------------


def test_apply_overrides_sets_named_field():
    wf = _full_workflow()
    out = apply_overrides(wf, {"sampler.cfg": 7.5})
    assert find_by_title(out, "sampler")["inputs"]["cfg"] == 7.5


def test_apply_overrides_does_not_mutate_the_input_workflow():
    wf = _full_workflow()
    original = copy.deepcopy(wf)
    apply_overrides(wf, {"sampler.cfg": 7.5})
    assert wf == original


def test_apply_overrides_raises_on_unknown_title():
    wf = _full_workflow()
    with pytest.raises(WorkflowError, match="no node titled 'nope'"):
        apply_overrides(wf, {"nope.cfg": 1})


def test_apply_overrides_raises_on_key_without_a_dot():
    wf = _full_workflow()
    with pytest.raises(WorkflowError, match="title.field"):
        apply_overrides(wf, {"cfg": 1})


# --- seeds_for --------------------------------------------------------


def test_seeds_for_fixed_seed_increments_deterministically():
    assert seeds_for(_run(seed=100, count=3)) == [100, 101, 102]


def test_seeds_for_random_seeds_are_distinct_and_in_range():
    seeds = seeds_for(_run(seed=None, count=25))
    assert len(seeds) == 25
    assert len(set(seeds)) == 25  # no duplicate seed within one batch
    assert all(0 <= s <= 2**32 - 1 for s in seeds)


# --- validation against the real committed graphs ------------------------

_REQUIRED_TITLES = {
    "t2i": ("positive", "negative", "sampler", "output"),
    "i2i": ("positive", "negative", "sampler", "output", "input"),
    "t2v": ("positive", "negative", "sampler", "output"),
    "i2v": ("positive", "negative", "sampler", "output", "input"),
    "v2v": ("positive", "negative", "sampler", "output", "input"),
}


@pytest.mark.parametrize("mode", sorted(_REQUIRED_TITLES))
def test_real_committed_workflow_parses_and_applies(mode):
    """Loads the actual workflows/<mode>.json shipped by Task 6 -- not a
    fixture copy -- and proves this module's assumptions hold against it."""
    wf = load(WORKFLOWS_DIR / f"{mode}.json")

    for title in _REQUIRED_TITLES[mode]:
        node = find_by_title(wf, title)  # raises on a duplicate title
        assert node is not None, f"{mode}.json has no node titled {title!r}"

    run_kwargs = dict(
        mode=mode,
        prompt="an unmistakably distinct test prompt for validation",
        count=1,
        size=(512, 512),
    )
    if mode in ("i2i", "i2v"):
        run_kwargs["input"] = Path("uploaded.png")
    elif mode == "v2v":
        run_kwargs["input"] = Path("uploaded.mp4")
    run = Run(**run_kwargs)

    out = apply_run(wf, run)

    positive_out = find_by_title(out, "positive")
    field = "text" if "text" in positive_out["inputs"] else "prompt"
    assert positive_out["inputs"][field] == run.prompt

    # the loaded graph itself must be untouched
    positive_original = find_by_title(wf, "positive")
    assert positive_original["inputs"][field] != run.prompt
