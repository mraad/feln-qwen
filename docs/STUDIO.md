# FELN Studio on Orin

The existing `feln-studio` application is deployed unchanged at revision
`03123e49fca9f80f42b3ee1b0f0dde39ca9bd2ef`. It uses the existing fine-tuned
Qwen3.5-9B Q4_K_M model through `http://127.0.0.1:18092` on Orin.
The model and application API services were not restarted or replaced.

## Open from the Mac

Services and tunnels were subsequently stopped at the user’s request. See the
[README](../README.md) to resume the existing deployment when intended. Once Studio
is running, open **http://localhost:8766/** through this tunnel:

```sh
ssh -fN -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 -L 127.0.0.1:8766:127.0.0.1:8766 orin
```

Studio binds only to Orin's loopback interface, preserving its Host and Origin checks.
Keep the same local and remote port. No model inference runs on the Mac.
Ports 18091 and 18092 can be forwarded separately for the API endpoints.

Choose a recorded question or type your own, then click **Generate**. The only
registered backend is **Qwen3.5-9B - FELN LoRA - Q4_K_M**. The picker contains
the 534 held-out v6 questions, judged against their normalized expected FELN.
Custom questions have no gold verdict. Editing the prompt changes the behavior
from the default prompt used in the published model benchmarks.

This is Studio's existing generation/inspection UI. Map execution is not implemented
in Studio; the separate Qwen app's `/query` endpoint still provides spatial GeoJSON.

## Screenshot

![Deployed FELN Studio with Qwen3.5-9B, a pipelines-to-discoveries query, generated JSON, and an exact-match verdict](images/feln-studio-orin.png)

[Full-resolution PNG](images/feln-studio-orin.png), captured on
**2026-09-17 at 14:14 UTC** from `http://localhost:8766/` through the Mac's SSH tunnel
to Orin. The recorded question is:

> Pipelines with a dimension over 20.0 that are no more than 2 miles from a condensate discovery.

The browser selected this question and clicked **Generate FELN** using the default
trained prompt. The real response was valid and exactly matched the recorded FELN,
with 54 generated tokens and 3.216 seconds of server time for this individual request.
This is a screenshot example, not an additional aggregate benchmark.

Capture used headless Chrome 153.0.8010.48 with Playwright 1.63.0, a 1440×1100 viewport,
light appearance, and a 1440×1977 full-page PNG. No application content or model responses
were substituted. The browser reported no JavaScript page errors, and the captured
image was visually inspected. [Capture metadata, response, and image hash](evidence/studio-screenshot.json).

The unchanged Studio UI contains the legacy helper text “Greedy decoding on this
Mac. Nothing leaves it.” For this deployment, the browser is on the Mac and all
model computation is on Orin; requests travel through the SSH tunnel.

## Deployment and dependencies

- Host: private SSH alias `orin` (address redacted).
- Unit: `feln-studio.service`, deployment account redacted; stopped, enabled at boot.
- Workspace: `/data/feln-studio-20260917`.
- Application source: `app/`; isolated runtime: `venv/`.
- Python: measured **3.13.15**, installed privately using **uv 0.12.15**.
- Model bundle: `/data/feln-qwen-20260917/bundle`.
- Catalog and gold: `/data/feln-qwen-20260917/data-20260917-v6/{Layers.json,test.json}`.
- Logs: `/data/feln-studio-20260917/logs/studio.log`.

Only the existing GGUF backend is enabled (`--backends lora`). Its imports need no
RAG, embedding, MLX, or PyTorch packages. Runtime pins are in
[`studio-requirements.txt`](../serving/studio-requirements.txt); the full installed
package list is in [deployment evidence](evidence/studio-deployment.txt).
The pinned FELN source is the same revision used by the model evaluation,
`b42696ee1c84d114c446fab0879008e305844e22`. The source is run directly with `python -m`;
the project's full multi-backend dependency set is not installed.
The existing model environment, system Python, JetPack, and CUDA are unchanged.
The local `.env` and credentials were excluded from the Git archive.

## Reproduce in a fresh directory

Set `LAYERS_JSON_REPOSITORY` in the install shell and configure the service-account
drop-in as described in [Local configuration](CONFIGURATION.md). The checked-in
unit uses a neutral example account, not the historical deployment identity.

These commands describe the completed deployment. They must not be rerun over
existing release files without approval. First archive the clean source on the Mac:

```sh
git -C ../feln-studio archive --format=tar \
  --output="$PWD/runs/feln-studio-03123e4.tar" \
  03123e49fca9f80f42b3ee1b0f0dde39ca9bd2ef
```

Create the new directory on Orin, copy the source archive, the pinned
`runs/reference/feln.tar`, and the two `serving/` files named below. On Orin:

```sh
cd /data/feln-studio-20260917
mkdir app feln logs
tar -xf feln-studio-03123e4.tar -C app
tar -xf feln.tar -C feln
python3 -m venv bootstrap
bootstrap/bin/python -m pip install uv==0.12.15
export UV_PYTHON_INSTALL_DIR=$PWD/python
export UV_CACHE_DIR=$PWD/cache
bootstrap/bin/uv python install 3.13.15 --no-bin
bootstrap/bin/uv venv --python 3.13.15 --managed-python venv
bootstrap/bin/uv pip install --python venv/bin/python -r studio-requirements.txt ./feln
systemd-analyze verify feln-studio.service
# Only if the destination unit does not exist:
sudo cp -n feln-studio.service /etc/systemd/system/feln-studio.service
sudo systemctl daemon-reload
# Set [Service] User= to the existing deployment account in the local drop-in.
sudo systemctl edit feln-studio.service
sudo systemctl enable --now feln-studio.service
```

The complete launch flags and hardening settings are in
[`feln-studio.service`](../serving/feln-studio.service). It deliberately connects to
the existing model; `--start` is not used and no second model is loaded.

## Verification and operation

Measured checks:

- Existing Studio suite: **7 tests passed**; Ruff check and format check passed.
  Pyright passed with the explicit existing environment:
  `.venv/bin/pyright --pythonpath .venv/bin/python`.
- `/`, `/app.js`, and `/style.css` returned HTTP 200 through the Mac tunnel.
- `/api/config` reported the Qwen backend healthy and all 534 recorded questions.
- Nine real requests, one per primary-layer/layer-count stratum, returned valid,
  exact FELN through Studio. Full responses and timings are in
  [integration evidence](evidence/studio-integration.json).
- Restart changed Studio PID 136335→136753. The existing model PID 123809 and app
  PID 123810 stayed unchanged. All three services remained active and enabled.
  [Restart and environment evidence](evidence/studio-deployment.txt).

At initial deployment, CUA browser automation was unavailable
(`CUA_REPL_ENABLED_SURFACES` was unset). The subsequent screenshot check used an
isolated headless Chrome session to select a recorded question, click Generate,
and verify the **EXACT MATCH** verdict. The screenshot and response are linked
above. A physical device reboot was not performed.

```sh
ssh orin 'systemctl status feln-studio.service --no-pager'
ssh orin 'tail -50 /data/feln-studio-20260917/logs/studio.log'
curl http://localhost:8766/api/config
curl http://localhost:8766/api/generate -H 'Content-Type: application/json' \
  -d '{"backend":"lora","query":"Show all wells"}'
```

Restart only when no Studio request is active:
`ssh orin 'sudo systemctl restart feln-studio.service'`.
For an approved rollback, stop/disable only `feln-studio.service` and preserve its
files. The model and Qwen API services can continue running independently.
