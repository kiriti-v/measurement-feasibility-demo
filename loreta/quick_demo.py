"""Small, self-contained planning page using the shared synthetic measurements."""
import json
from pathlib import Path


def summaries(bundles, engine):
    output = {}
    for noise, bundle in bundles.items():
        for marker in ("adr", "theta_rel"):
            for condition in ("Condition A", "Condition B"):
                measured = engine.measurement(bundle, marker, "whole_scalp", condition, "Reference")
                output[f"{noise}|{marker}|{condition}"] = {
                    "mean": measured["effect"]["mean"], "sd": measured["effect"]["sd"],
                    "floor": measured["floor_matched"], "n": measured["effect"]["n"],
                    "minutes": measured["n_ref"] * bundle.epoch_seconds / 60,
                    "sb": measured["reliability"]["spearman_brown"],
                }
    return output


def page(bundles, engine, release):
    template = Path(__file__).with_name("quick_demo.html").read_text()
    return template.replace("__DATA__", json.dumps(summaries(bundles, engine), allow_nan=False)).replace(
        "__RELEASE__", release[:12])
