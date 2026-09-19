# Measurement feasibility demo

A synthetic demonstration of repeated-measures marker contrasts, within-recording variability,
split-half consistency, and illustrative recording-time and sample-size planning.

The root page is a small mobile-friendly, self-contained planning demonstration. Its summaries
come from the same synthetic Python analysis; target changes calculate locally without network
requests. `/classic` retains all analytical charts. The existing Render Free service can still
sleep before either page is delivered: this frontend improvement does not remove host cold starts.

This package contains no participant data. Its two presets simulate different amounts of
within-condition variability using a fixed random seed. The numbers are not study results.
The analytical functions are shared with the local research tool; this hosted interface is
deliberately read-only and does not provide its table upload/staging workflow.

## Run

Use Python 3.13.7. Create a virtual environment, then:

```sh
pip install -r requirements.txt
python -m loreta.public_demo
```

Open http://127.0.0.1:8085. For a hosted process:

```sh
gunicorn loreta.public_demo:server --bind 0.0.0.0:8085 --workers 1 --threads 2 --timeout 60
```

## Interpretation

Pooled empirical p95 is a descriptive within-recording variability benchmark, not a calibrated
individual detection threshold. Internal split-half consistency is not independent test-retest
validation. Time projections assume stationary independent epochs; planning N uses a normal
approximation for a two-sided paired mean test at alpha .05 and 80% power. The selected observed
effect is only an illustrative target. Real planning needs independently justified targets and
state-, protocol-, and duration-matched variance estimates. These are separate projections,
not a jointly validated protocol optimizer or a clinical device.

## Deployment and cost boundary

`render.yaml` requests one Free Python web service with no database or disk. Connect the sanitized
repository in a non-PMG Render account and use **After CI Checks Pass**, not On Commit.
Do not add a payment method, upgrade compute, or create paid resources. If Render requires
payment-card verification, stop and choose another deployment route with the owner.

Render Free sleeps after 15 idle minutes; startup may take about a minute. It offers no uptime
guarantee and may suspend service on resource limits. It is a conference demo, not clinical
infrastructure. See https://render.com/docs/free and https://render.com/docs/deploys.

`/healthz` identifies the app, synthetic data boundary, and `RENDER_GIT_COMMIT` release. Verify that
value against the tested commit after every deployment. The page shows its short release ID.
There are no upload callbacks or persistent user-data writes. Host access logs may retain normal
request metadata; this is not a confidential data service.

## Test and rollback

```sh
pip install pytest==9.1.1
python -m pytest -q
```

CI checks all packaged file hashes, data boundary, bounded inputs, HTTP routes, actual Dash callback,
and every preset/marker/condition combination. Only successful CI commits should deploy.
The first service creation also needs a green CI result before use; creation may deploy before
automatic-deploy gates apply. Do not print a QR until live release and browser checks pass.

To roll back, select a previously verified deploy in Render (Free retains two previous rollbacks),
disable automatic deployment during diagnosis, and verify `/healthz` and interaction again. Revert
the faulty commit in GitHub and wait for green CI before re-enabling automatic deployment. Rebuild
the publication manifest whenever package files change; never copy the research tree wholesale.
