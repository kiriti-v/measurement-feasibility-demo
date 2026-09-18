import hashlib
import json
from pathlib import Path

import pytest
from loreta import public_demo as demo


def test_no_research_presets_loaded():
    assert demo.engine.MUSIC.epochs.empty
    assert demo.engine.OPENNEURO.epochs.empty
    assert all(b.epochs.subject.str.startswith("sim-").all() for b in demo.BUNDLES.values())


def test_endpoints():
    client = demo.server.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/healthz").json["data"] == "synthetic-only"
    layout = client.get("/_dash-layout").get_data(as_text=True)
    assert '"Upload"' not in layout
    assert "ICU" not in layout and "INMO" not in layout
    for path in ["/assets/../results/markers/region_epochs.csv", "/upload"]:
        assert client.post(path, json={}).status_code == 404
    assert client.post("/_dash-update-component", json={}).status_code == 400
    assert client.post("/_dash-update-component", data="x" * 9000,
                       content_type="application/json").status_code == 413


@pytest.mark.parametrize("noise", ["lower", "higher"])
@pytest.mark.parametrize("marker", ["adr", "theta_rel"])
@pytest.mark.parametrize("condition", ["Condition A", "Condition B"])
def test_analysis(noise, marker, condition):
    result = demo.update(noise, marker, condition, 1)
    assert len(result) == 6
    assert all(len(fig.data) for fig in result[1:])
    assert demo.engine.measurement(demo.BUNDLES[noise], marker, "whole_scalp",
                                   condition, "Reference")["effect"]["n"] == 40


def callback_payload():
    key = next(iter(demo.app.callback_map))
    return {"output": key, "outputs": [{"id": i, "property": p} for i, p in [
        ("audit", "children"), ("overview", "figure"), ("effects", "figure"),
        ("floor", "figure"), ("reliability", "figure"), ("planning", "figure")]],
        "inputs": [{"id": k, "property": "value", "value": v} for k, v in
                   [("noise", "lower"), ("marker", "adr"),
                    ("condition", "Condition A"), ("target", 1)]],
        "state": [], "changedPropIds": ["noise.value"]}


def test_actual_dash_callback_and_rejection():
    client = demo.server.test_client()
    payload = callback_payload()
    response = client.post("/_dash-update-component", json=payload)
    assert response.status_code == 200
    assert len(response.json["response"]) == 6
    payload["inputs"][0]["value"] = "music"
    assert client.post("/_dash-update-component", json=payload).status_code == 400
    payload = callback_payload()
    payload["state"] = [{"id": "uploaded-data", "property": "data", "value": "private"}]
    assert client.post("/_dash-update-component", json=payload).status_code == 400


def test_release_allowlist():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "release-manifest.json").read_text())
    assert set(manifest) == {
        "loreta/__init__.py", "loreta/public_demo.py", "loreta/general_feasibility_explorer.py",
        "loreta/reactivity.py", "loreta/staging_recipes.py", "requirements.txt", "render.yaml",
        "README.md", ".github/workflows/ci.yml", "tests/test_public_demo.py", ".gitignore"}
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
              and not any(x in p.parts for x in [".git", "__pycache__", ".pytest_cache", ".venv"])}
    assert actual == set(manifest) | {"release-manifest.json"}
    for name, digest in manifest.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest


def test_log_scale_and_variability_scenario():
    measurements = [demo.engine.measurement(demo.BUNDLES[key], "adr", "whole_scalp",
                    "Condition A", "Reference") for key in ["lower", "higher"]]
    assert demo.engine.uses_log("adr", demo.BUNDLES["lower"])
    assert measurements[0]["effect"]["mean"] == pytest.approx(.164, abs=.001)
    assert measurements[1]["floor_matched"] > measurements[0]["floor_matched"] * 2


@pytest.mark.parametrize("target", [float("nan"), float("inf"), -1, 100, "1", True])
def test_bounded_target(target):
    with pytest.raises(ValueError):
        demo.validate_selection("lower", "adr", "Condition A", target)
