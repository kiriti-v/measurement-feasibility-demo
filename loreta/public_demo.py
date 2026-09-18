"""Bounded synthetic demonstration; no upload or research-data routes."""
from __future__ import annotations

import math
import os

os.environ["FEASIBILITY_PUBLIC_DEMO"] = "1"

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, dcc, html
from flask import abort, jsonify, request
from loreta import general_feasibility_explorer as engine

APP_ID = "measurement-feasibility-demo"
RELEASE = os.environ.get("RENDER_GIT_COMMIT", "local-unreleased")


def synthetic_bundle(noise: float) -> engine.DatasetBundle:
    rng = np.random.default_rng(20260917)
    rows = []
    for subject in range(40):
        baseline = rng.normal(0, .2)
        response = rng.normal(.15, .12)
        for condition, shift in [("Reference", 0), ("Condition A", response),
                                 ("Condition B", response * .4)]:
            for epoch in range(30):
                value = baseline + shift + rng.normal(0, noise) + epoch * .001
                rows.append((f"sim-{subject:03}", condition, epoch, "whole_scalp",
                             10 ** value, 1 / (1 + np.exp(-value))))
    frame = pd.DataFrame(rows, columns=["subject", "condition", "epoch", "region",
                                        "adr", "theta_rel"])
    return engine.DatasetBundle("synthetic", "Synthetic repeated measurements", frame,
                                marker_scales={"adr": "log10", "theta_rel": "relative"})


BUNDLES = {"lower": synthetic_bundle(.2), "higher": synthetic_bundle(.65)}
app = Dash(__name__, title="Measurement feasibility", include_assets_files=False,
           assets_folder="_no_public_assets")
app.index_string = engine.app.index_string
server = app.server
server.config["MAX_CONTENT_LENGTH"] = 8192
CONTROLS = {"noise": {"lower", "higher"}, "marker": {"adr", "theta_rel"},
            "condition": {"Condition A", "Condition B"}}
app.layout = html.Main(className="shell", children=[
    html.H2("Measurement feasibility"),
    html.P("Synthetic demonstration · 40 simulated subjects · no patient data. Uploads are disabled."),
    html.Section(className="source", children=[
        engine._control("Within-condition variability", dcc.Dropdown(
            id="noise", options=[{"label": "Lower", "value": "lower"},
                                  {"label": "Higher", "value": "higher"}], value="lower", clearable=False)),
        engine._control("Marker", dcc.Dropdown(id="marker", options=engine._options(
            ["adr", "theta_rel"], engine.label), value="adr", clearable=False)),
        engine._control("Condition vs Reference", dcc.Dropdown(id="condition",
            options=engine._options(["Condition A", "Condition B"]), value="Condition A", clearable=False)),
        engine._control("Target / observed mean magnitude", dcc.Slider(
            id="target", min=.25, max=2, step=.25, value=1)),
    ]),
    html.P("Empirical p95 is a within-recording variability benchmark, not an individual detection threshold. "
           "Time projections assume independent, stationary epochs; sample-size estimates are illustrative, not guarantees."),
    html.Section(id="audit", className="audit"),
    dcc.Graph(id="overview"),
    html.Div(className="grid", children=[dcc.Graph(id=name) for name in
             ["effects", "floor", "reliability", "planning"]]),
    html.Small(f"{APP_ID} · release {RELEASE[:12]}"),
])


@server.before_request
def request_boundary():
    if request.method == "POST":
        if request.path != "/_dash-update-component":
            abort(404)
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or body.get("output") not in app.callback_map:
            abort(400)
        inputs = body.get("inputs")
        if not isinstance(inputs, list) or len(inputs) != 4 or body.get("state") not in (None, []):
            abort(400)
        values = {}
        for item in inputs:
            if not isinstance(item, dict) or item.get("property") != "value":
                abort(400)
            key = item.get("id")
            if key not in {*CONTROLS, "target"} or key in values:
                abort(400)
            values[key] = item.get("value")
        try:
            validate_selection(**values)
        except (TypeError, ValueError):
            abort(400)


def validate_selection(noise, marker, condition, target):
    for key, value in [("noise", noise), ("marker", marker), ("condition", condition)]:
        if not isinstance(value, str) or value not in CONTROLS[key]:
            raise ValueError("Invalid selection")
    if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target) or not .25 <= target <= 2:
        raise ValueError("Invalid target")


@server.get("/healthz")
def health():
    return jsonify(status="ok", app=APP_ID, release=RELEASE, data="synthetic-only")


@app.callback(Output("audit", "children"), Output("overview", "figure"),
              Output("effects", "figure"), Output("floor", "figure"),
              Output("reliability", "figure"), Output("planning", "figure"),
              Input("noise", "value"), Input("marker", "value"),
              Input("condition", "value"), Input("target", "value"))
def update(noise, marker, condition, target):
    validate_selection(noise, marker, condition, target)
    bundle = BUNDLES[noise]
    args = (bundle, marker, "whole_scalp", condition, "Reference")
    measured = engine.measurement(*args, target)
    audit = [html.Div(className="audit-cell", children=[html.Div(key), html.Strong(value)])
             for key, value in engine.audit_readout(*args, target, measured)
             if key != "Clear floor individually"]
    planning = engine.fig_detectability(*args, target, measured)
    planning.update_layout(title="Illustrative recording-time projection",
                           yaxis_title=f"Projected variability benchmark ({engine.units(marker, bundle)})")
    planning.data[0].hovertemplate = "%{x:.1f} usable min/unit<br>projected benchmark %{y:.3f}<extra></extra>"
    return (audit, engine.fig_condition_overview(bundle, marker, "whole_scalp", "Reference"),
            engine.fig_effects(*args), engine.fig_floor(*args, measured),
            engine.fig_reliability(*args, measured), planning)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8085")), debug=False)
