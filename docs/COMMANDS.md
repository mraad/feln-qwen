# Exact commands and service operations

Host identities and private paths are sanitized. Load the host-specific variables
and configure the SSH aliases described in [Local configuration](CONFIGURATION.md)
before using these commands. The service account in checked-in units is a placeholder;
configure its local drop-in after copying the units and before starting them.

Commands below run from this repository unless a host is stated. The validated input
directory is `runs/data-20260917-v6`; choose a new output path for every subsequent run.
Artifacts and source snapshots are ignored by Git, not discarded.

## Prepare and check (Mac)

```sh
export PYTHONPATH="${FELN_APP_REPO}:$PWD"
"${FELN_APP_REPO}/.venv/bin/python" training/prepare.py \
  --source "${NORTHSEA_SOURCE}" \
  --output runs/data-20260917-v6
"${FELN_APP_REPO}/.venv/bin/python" training/prepare.py \
  --output runs/data-20260917-v6 --verify
"${FELN_APP_REPO}/.venv/bin/python" training/test_contract.py
"${FELN_APP_REPO}/.venv/bin/ruff" check training serving
```

The first command above will deliberately fail if the output already exists. The
verification commands are read-only and safe to repeat. The source originals are
never updated. Archived reference sources and exact revisions are in PIPELINE.md.

## Reference sources and staging

The two reference archives were made with `git archive` at the pinned commits below;
uncommitted edits in the sibling projects are excluded and remain untouched. These
commands are for a **fresh** workspace: do not rerun an archive, copy, or extraction
over existing files without approval.

```sh
mkdir -p runs/reference
git -C ../feln-lora archive --format=tar \
  6d5ad2650ea5b86cd0a92e75075dd2b04555a787 > runs/reference/feln-lora.tar
git -C ../feln archive --format=tar \
  b42696ee1c84d114c446fab0879008e305844e22 > runs/reference/feln.tar
```

RTX uses `/opt/dlami/nvme/feln-qwen-20260917/{app,feln,training,reference}`;
Orin uses `/data/feln-qwen-20260917/{app,feln,serving,logs}`. Extract the application
archive into `app/` and the FELN archive into `feln/` on each host. Copy this repo's
`training/*.py` to RTX `training/`; the Orin evaluator is copied as `evaluate.py`.
Copy the entire validated `runs/data-20260917-v6/` to each workspace without changing
its contents. Run `prepare.py --verify` against the staged copy before training.
The measured dependency snapshot is `evidence/training-packages.json`; use the
existing RTX environment read-only, as below.

The database copy specifically authorized by the user was:

```sh
scp "${NORTHSEA_SOURCE}/NorthSea.ddb" \
  orin:/data/feln-qwen-20260917/NorthSea.ddb
ssh orin 'sha256sum /data/feln-qwen-20260917/NorthSea.ddb'
```

Expected SHA-256: `bf176393f6745628d084fdd0564be8399507222215a44dc6a66bfe77cb94d6c8`.
The app opens this copy read-only; it never reads the workstation at inference time.

## Train (RTX)

```sh
ssh rtx-training
cd /opt/dlami/nvme/feln-qwen-20260917
export PYTHONPATH=$PWD/app:$PWD/feln
nvidia-smi
tmux ls
# Only use verified free GPUs; do not interrupt someone else's job.
"${TRAIN_VENV}/bin/torchrun" --standalone --nproc_per_node=2 \
  training/train.py --base base --data data-20260917-v6 --output smoke-v6 --smoke \
  > smoke-v6.log 2>&1
# Run only after smoke success and stock-model inference on Orin:
"${TRAIN_VENV}/bin/torchrun" --standalone --nproc_per_node=2 \
  training/train.py --base base --data data-20260917-v6 --output full-v6 \
  > full-v6.log 2>&1
```

`base` is a complete HF snapshot of Qwen/Qwen3.5-9B at
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Model and tokenizer identifiers,
training arguments, wall time, peak allocator memory, and chosen checkpoint are in
each run's `report.json`. Trainer history and optimizer checkpoints remain alongside
the adapter. Do not run `uv sync` or install into the shared AutoModel environment.

The exact download command was:

```sh
HF_HOME=$PWD/hf "${TRAIN_VENV}/bin/python" -c \
  'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3.5-9B", revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a", local_dir="base")'
```

## Score and export (RTX)

```sh
# Build the identical stock prompt/schema bundle in a fresh directory:
"${TRAIN_VENV}/bin/python" training/export.py \
  --base base --schema data-20260917-v6/Layers.json --bundle-only --output stock-bundle
# Same data, exact prompt, greedy generation and strict comparator for both:
CUDA_VISIBLE_DEVICES=0 "${TRAIN_VENV}/bin/python" training/evaluate.py \
  --model base --bundle stock-bundle --records data-20260917-v6/test.json \
  --output baseline-hf-test-v6
CUDA_VISIBLE_DEVICES=1 "${TRAIN_VENV}/bin/python" training/evaluate-v2.py \
  --model base --adapter full-v6/adapter --bundle stock-bundle \
  --records data-20260917-v6/test.json --output tuned-hf-test-v6-bf16
"${TRAIN_VENV}/bin/python" training/export-v2.py \
  --base base --adapter full-v6/adapter --schema data-20260917-v6/Layers.json --output merged-v6-mtp0
"${LLAMA_CPP_DIR}/.venv-convert/bin/python" \
  "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" merged-v6-mtp0 --outfile tuned-v6-text-f16.gguf --outtype f16 --no-mtp
"${LLAMA_CPP_DIR}/build/bin/llama-quantize" \
  tuned-v6-text-f16.gguf tuned-v6-text-q4_k_m.gguf Q4_K_M
sha256sum tuned-v6-text-q4_k_m.gguf
```

The remote `export-v2.py` is the repository's `training/export.py`. Both zero-MTP
metadata and `--no-mtp` are required for this text-only checkpoint: the converter
otherwise assumes the optional predictor is present and emits an invalid 33rd block.

These are the implemented commands; consult the evidence/results for which have
completed. Checkpoint selection uses validation, never test. A GGUF validation score
must be recorded before final selection of quantization. Evaluate Q8_0 if Q4_K_M
degrades validation accuracy; do not select quantization on the test result.

## Orin installation and runtime

The final artifact was copied through the Mac, without copying either SSH key to
a remote machine. For a fresh destination only:

```sh
scp -3 -o ServerAliveInterval=30 \
  rtx-training:/opt/dlami/nvme/feln-qwen-20260917/tuned-v6-text-q4_k_m.gguf \
  orin:/data/feln-qwen-20260917/tuned-v6-text-q4_k_m.gguf
ssh orin 'sha256sum /data/feln-qwen-20260917/tuned-v6-text-q4_k_m.gguf'
```

Expected SHA-256: `7401eb1746a82814db8e5179242c824f9c89cc9f3eca419a15b3444ec309af69`.
The app needs `inference_config.json` and `Layers.json` from the export in its local
`bundle/`; these have the same hashes as the already-staged stock bundle because
fine-tuning did not change the tokenizer, prompt, or schema. Verify parity against
`evidence/prompt-schema-parity.txt` rather than overwriting an existing bundle.

The task uses its own venv, unpacked application sources, database copy, bundle, model,
and logs below `/data/feln-qwen-20260917`. Python packages are installed into that venv;
no OS packages or NVIDIA libraries are replaced. The DuckDB spatial extension is
installed into the serving user's extension cache once, then loaded offline.

```sh
cd /data/feln-qwen-20260917
python3 -m venv venv
venv/bin/pip install -r serving/requirements.txt ./feln
venv/bin/python -c 'import duckdb; duckdb.connect().execute("INSTALL spatial")'
```

The actual execution check exposed DuckDB's timezone dependency `pytz`; it is now
explicitly pinned, rather than relying on an unrelated package to provide it.

The units in `serving/` are installed and enabled. Their running and restart state
is reported separately in the final evidence. Confirm checksums before installing.
Refuse to overwrite pre-existing unit names or an active release without approval.

```sh
sudo systemd-analyze verify /data/feln-qwen-20260917/serving/feln-qwen-*.service
# For a fresh install only; first check these paths do not already exist:
ln -s tuned-v6-text-q4_k_m.gguf model.gguf
sudo cp -n /data/feln-qwen-20260917/serving/feln-qwen-*.service /etc/systemd/system/
sudo systemctl daemon-reload
# Set [Service] User= to the existing deployment account in each local drop-in.
sudo systemctl edit feln-qwen-model.service
sudo systemctl edit feln-qwen-app.service
sudo systemctl enable --now feln-qwen-model.service feln-qwen-app.service
systemctl status feln-qwen-model.service feln-qwen-app.service --no-pager
curl http://127.0.0.1:18091/health
curl http://127.0.0.1:18091/query -H 'Content-Type: application/json' \
  -d '{"text":"Show gas wells in Norway","limit":10}'
```

FELN-only endpoint: POST `/feln` with the same body. `/query` additionally executes the
compiled query through the original application's parameterized read-only DuckDB
planner. It returns FELN, raw model output, model timing, and bounded GeoJSON results.
Output schema validity does not guarantee that a model understood the request.

Both listeners bind only to loopback. From the Mac:

```sh
ssh -L 18091:127.0.0.1:18091 orin
curl http://127.0.0.1:18091/health
```

Model endpoint: Orin `127.0.0.1:18092`, llama.cpp `/completion`.
App endpoint: Orin `127.0.0.1:18091`. No workstation tunnel is used.
Studio additionally uses Orin `127.0.0.1:8766`. See the
[combined Mac tunnel command](../README.md#application-access) to forward all three
services, and the [Studio runbook](STUDIO.md) for its independent service management.

## Actual-artifact evaluation and end-to-end benchmark (Orin)

```sh
cd /data/feln-qwen-20260917
export PYTHONPATH=$PWD/app
venv/bin/python evaluate.py --url http://127.0.0.1:18092 --bundle bundle \
  --records data-20260917-v6/test.json --output tuned-orin-test-repeat
venv/bin/python serving/benchmark-v2.py --records data-20260917-v6/test.json \
  --output e2e-benchmark-repeat.json
systemctl show feln-qwen-model feln-qwen-app -p MemoryCurrent -p MemoryPeak
```

The initial matched Orin run uses the isolated probe on port 18096 and saves
`tuned-orin-test-v6`; the command above repeats it against the permanent service
in a fresh output directory. The matched RTX command was:

```sh
cd /opt/dlami/nvme/feln-qwen-20260917
CUDA_VISIBLE_DEVICES=0 "${LLAMA_CPP_DIR}/build/bin/llama-server" \
  -m tuned-v6-text-q4_k_m.gguf -c 4096 -np 1 -ngl 99 \
  --host 127.0.0.1 --port 18096 --no-webui
# In another RTX shell:
PYTHONPATH=$PWD/app "${TRAIN_VENV}/bin/python" training/evaluate-v2.py \
  --url http://127.0.0.1:18096 --bundle merged-v6-mtp0 \
  --records data-20260917-v6/test.json --output tuned-rtx-q4-test-v6
```

`evaluate-v2.py` is the repository's `training/evaluate.py`; the original remote
version is retained. Both hosts use the same GGUF SHA-256 and test-record hashes.
The RTX inference command uses one GPU; the separate training command uses two.

The deployed `benchmark-v2.py` is the repository's `serving/benchmark.py`; the
initial probe script is retained separately. The benchmark takes the first two test
questions per primary-layer/layer-count stratum (18 cases), records HTTP failures,
and reports raw and compiled FELN separately.
The completed deployment benchmark is `e2e-benchmark-v6.json` on Orin and
`evidence/orin-e2e-benchmark-v6.json` locally. The repeat commands deliberately use
new output names; all evaluators refuse to overwrite completed results.

Memory sampling reads `/proc/meminfo` at 50 ms intervals. Report system shared-RAM
pressure and sampling limits, not a fictitious dedicated VRAM reading from
`nvidia-smi` (unsupported on this Orin). Warm latency excludes the first request.
HF batched scoring timings are not comparable to single-request HTTP latency.

## Operations, troubleshooting, rollback

Service management on Orin (restart only when the task's services have no active
users). The explicit restart below was verified during deployment:

```sh
systemctl status feln-qwen-model.service feln-qwen-app.service --no-pager
sudo systemctl restart feln-qwen-model.service feln-qwen-app.service
curl --fail --retry 20 --retry-delay 1 --retry-connrefused http://127.0.0.1:18091/health
```

For an approved rollback, disable only these new units and preserve their files:

```sh
sudo systemctl disable --now feln-qwen-app.service feln-qwen-model.service
```

- Logs: `/data/feln-qwen-20260917/logs/{model,app}.log`; training logs remain on RTX.
- `systemctl is-enabled` establishes boot activation; `systemctl restart` verifies
  service restart. Do not claim a device reboot was tested unless it was performed.
- HTTP 429: one active request; retry after it completes. This is deliberately a
  single-user service. HTTP 422: invalid input or model output. HTTP 503: model/runtime
  or execution unavailable. `/health` reports model readiness.
- Missing `spatial`: run `venv/bin/python -c 'import duckdb; duckdb.connect().execute("INSTALL spatial")'`
  as the deployment service account (name redacted), then retry. Installation needs network; routine serving does not.
- OOM or thermal throttling: inspect `/proc/meminfo`, `tegrastats`, and logs. Do not
  change JetPack/power/boot configuration. A smaller model requires a fresh validated
  run, not relabeling this model's results.
- Missing database: copying the full database requires the explicit transfer approval
  recorded in PIPELINE.md. Do not fetch it implicitly from the workstation at runtime.
- To roll back this new deployment, stop/disable only `feln-qwen-app` and
  `feln-qwen-model` after confirming no users are relying on them. No previous service
  was replaced. Preserve all artifacts. To restore a previous project-owned release,
  point its model/bundle paths at retained, checksum-verified artifacts and restart
  these two units with the user's approval if they are in use.

Remaining limitations are documented with the measured results, including synthetic
language, static catalog, regional planar distance approximation (EPSG:32632), and
single-request concurrency.
