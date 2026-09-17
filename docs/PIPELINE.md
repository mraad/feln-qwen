# FELN Qwen pipeline

Status: full training, actual-artifact evaluation, persistent Orin deployment,
service restart, and end-to-end benchmarks are complete. All results below are explicitly
measured or estimated. See [README](../README.md) for the comparison table and
[lessons](LESSONS.md) for observed failures and fixes.

For a sequential explanation of how the data, two-GPU training, checkpoint
selection, export, transfer, and Orin services fit together, start with
[Training and deployment](TRAINING_AND_DEPLOYMENT.md).

## Scope and task

Implementation lives in `feln-qwen`, as confirmed by the user. `feln-lora` is a
read-only application reference, pinned at `6d5ad2650ea5b86cd0a92e75075dd2b04555a787`.
The FELN dependency is pinned at `b42696ee1c84d114c446fab0879008e305844e22`.
Source archives are in `runs/reference/` and copied into the isolated remote workspaces.
Existing Nemotron training artifacts were inspected as reference, never resumed.
`RTX-Qwen` is undefined in the inspected application source.

The real task is natural language → `{"layers":[],"where":[],"relations":[]}`.
`where[i]` filters `layers[i]`; `relations[i]` connects the primary layer to
`layers[i+1]`. The original application's `Schema`, `make_prompt`, `complete`,
`SpatialQuery`, and the FELN package's strict comparator are reused. The new HTTP
adapter omits the old application's remote GPU machine-control flow.

The user explicitly prioritizes accurate expected FELN JSON over latency. Assumption:
one simultaneous inference request, 4,096-token context, at most 512 output tokens,
with full model offload. Concurrent requests to the app receive HTTP 429.
Validation selects checkpoints; the held-out test is never a training input.
Prompt configuration and schema SHA-256 values match between the RTX bundle,
controlled baseline, and Orin application (`evidence/prompt-schema-parity.txt`).

## Hardware and software

Measured by SSH, not inferred from device names. Raw evidence is in `evidence/`.

| Item | Workstation (measured) | Orin (measured) |
|---|---|---|
| Host (sanitized aliases) | `ssh rtx-training` | `ssh orin` |
| GPU | 2 × RTX PRO 6000 Blackwell Server Edition | Jetson AGX Orin Developer Kit |
| GPU memory | 97,887 MiB each, initially idle | shared system memory |
| RAM | 536,562,638,848 bytes | 65,932,554,240 bytes |
| Initial available RAM | 525,277,069,312 bytes | 62,904,112 KiB |
| Disk location | /opt/dlami/nvme, 3,542,793,912,320 bytes available | /data, roughly 3.5 TiB available |
| Driver | 595.58.03 | 595.78 |
| CUDA compiler | 13.2.51 | 13.2.86 |
| Framework | torch 2.10.0+cu130, Transformers 5.15.1, PEFT 0.20.0 | llama.cpp CUDA, no PyTorch needed for serving |
| Platform | Ubuntu 24.04, x86_64 | Ubuntu 24.04, aarch64, L4T R39.2.1 |

`nvidia-jetpack` metapackage was not reported installed; report the measured L4T/CUDA
versions rather than assigning an inferred JetPack label. No system packages, power
mode, boot settings, or existing services were changed. The existing Orin build is
`434ddbbc0e30522e897670681e503b797c12b7c1`; the RTX conversion build is
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`.

New workspace: `/opt/dlami/nvme/feln-qwen-20260917` on RTX;
new deployment directory: `/data/feln-qwen-20260917` on Orin.
The existing AutoModel environment is used read-only; no packages are installed into
it, and no other project's checkpoints or services are changed.

## Model selection and memory budget

Current official releases were checked on 2026-09-17, including
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), and
[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).
Exact revisions are in `evidence/model-revisions.json`.
The selected training candidate is `Qwen/Qwen3.5-9B`, revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, with the identical tokenizer revision.
Only its text language model is trained/exported; no vision encoder is served.
The existing converter explicitly supports `Qwen3_5ForCausalLM`.

Selected deployment format: GGUF Q4_K_M, FP16 cache, one slot. The corrected
artifact is 5,629,108,640 bytes; its identity is in `evidence/tuned-text-export-v6.json`.
The stock 9B Q4 file already on Orin is 6,169,341,984 bytes (repository metadata);
its SHA-256 is checked against the published file. An independently converted stock
artifact from the pinned HF revision is also produced for the controlled comparison.

Estimated budget before the final app benchmark: 6 GiB weights/quantization metadata,
0.25 GiB attention KV allowance (theoretical tensors are 0.125 GiB: 8 full-attention
layers × 4 KV heads × 256 dimensions × 2 K/V × 2 bytes × 4,096 tokens),
0.25 GiB recurrent state allowance, 2 GiB runtime
buffers/overhead, 3 GiB app/Python/DuckDB allowance, 4 GiB safety margin: 15.5 GiB.
These are estimates, not a substitute for the final measured peak. A stock-model plus
running-app probe measured 56,445,296,640 bytes still available and a 141,418,496-byte
app cgroup peak before the successful end-to-end query. The "Show all wells" request
returned the expected FELN and one local database row in 3.6719 seconds (measured,
preliminary stock model, while a validation client also existed). See
`evidence/orin-app-with-database-memory.txt` and `evidence/orin-app-stock-e2e-v2.json`.
Do not interpret the model's small cgroup reading as total GPU memory: CUDA shared
allocations are also reflected in system RAM pressure. Shared RAM means
GPU and CPU allocations are counted against the same physical capacity.

Larger 27B and 35B candidates can fit at 4-bit on this Orin; they are **not** ruled out
by RAM. They require roughly 3× and 4× the resident weights, respectively, and the
27B dense model has materially more active computation. Selection of 9B is conditional
on its held-out accuracy, rather than a claim that the newest/largest model cannot run.
122B at roughly 4.8 effective bits per weight needs about 68 GiB for weights alone
(estimate), exceeding measured available RAM before the app/cache/overhead. Larger
Qwen3.8 MoE releases similarly exceed the full-residency budget. 4B and 2B are fallback
candidates if the 9B artifact cannot meet the measured resource requirements.

A larger stock candidate was measured explicitly: `ggml-org/Qwen3.8-27B-GGUF`,
revision `0669b98607d47046c7c2b3f801011d54a08cfccf`, file
`Qwen3.8-27B-Q4_K_M.gguf`, 18,973,870,432 bytes, SHA-256
`31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.
It loaded successfully on the existing Orin runtime (measured). On the same first
663-token validation prompt, 57 generated tokens ran at 7.49 tokens/s for 27B
versus 23.11 tokens/s for 9B (measured runtime log,
`evidence/orin-candidate-cost-comparison.txt`; not an aggregate benchmark). Its exact prompt is
generated from the official Qwen3.8-27B tokenizer revision in the revision manifest.
The stock 9B controlled artifact scored 18/90 raw exact, 19/90 compiled exact,
and 48/90 schema-valid on this selection set (measured;
`evidence/baseline9-selection.json`). Its p50 was 4.3146 seconds; timing is
preliminary because the 27B model was loaded during part of this run.

The deterministic selection set takes the first ten distinct families per
primary-layer/layer-count stratum from v6 validation (90 questions total), preserving
the same order and selection for both models. No test questions select model size.
The larger stock model scored 19/90 raw exact, 26/90 compiled exact, and
64/90 schema-valid, with warm p50 14.3466 seconds and p95 21.3253 seconds
(measured; `evidence/candidate27-selection.json`). It improves compiled accuracy
over stock 9B but remains poor on this application contract. The fine-tuned 9B Q4 artifact subsequently scored 673/673 on full validation,
including 90/90 on this same selection set, substantially exceeding the larger
stock alternative while using about 30% of its file size. This supports selecting
9B for this task; a fine-tuned 27B or 35B accuracy advantage has not been measured.
This is a stock larger alternative comparison, not a claim about a fine-tuned 27B.

The pre-training stock runtime probe generated 99 tokens at 22.6509 tokens/s,
with 487.131 ms prompt processing and 4,326.534 ms generation (measured).
Its intentionally minimal prompt did not yield valid application FELN; it proves
runtime inference only. Application accuracy uses the real prompt and schema grammar.
See `evidence/orin-stock9-inference.txt`.

## Dataset transformations and provenance

`training/prepare.py` snapshots FELN.json, Layers.json and all four OKF Markdown files
into a new output directory, refusing to overwrite an existing dataset.
`runs/data-20260917-v6` is the validated build; previous numbered paths preserve failed
validation attempts. All original source hashes still matched after preparation.

1. Read and validate all 1,000 FELN records and their `source_text` paraphrases.
2. Parse the actual OKF index/schema/domain/hint tables, checking supported fields,
   types, sampled values, and enum domains against Layers.json.
3. Seven Wells fields exist only in OKF (`symbol`, `casing_lot`, `composite_log`,
   `core_photo`, `cores`, `cuttings`, `dst`). Do not generate unsupported outputs.
   Record these fields and every alias/hint difference in the manifest. Layers.json
   governs the application's SQL types and matching policy. OKF aliases and verified
   domain/sample values contribute additional natural-language examples.
4. Generate 3,600 spatial examples with the reference application's typed renderer,
   using seed 20260917. Generate single-layer field/domain examples from verified OKF
   content. These are synthetic questions, not human-validated observations.
5. Apply the application's `Schema.compile` to labels. Preserve operators, NULL
   semantics, primary layer, and relation direction. Record each before/after label
   and its source provenance in `transformations.json`.
6. Normalize question whitespace/case for deduplication, fail on conflicting labels,
   and merge provenance lists for duplicates (149 removed, measured).
7. Group paraphrases and literal substitutions by SQL structure and spatial relation,
   canonicalizing secondary order while preserving the primary. Stratify groups by
   primary layer and number of layers, then split approximately 80/10/10.
   The shared catalog is intentionally available in every split; query families are
   disjoint. This tests generalization to held-out combinations, not unknown catalogs.
   v6 also groups **all distance-unit variants** and `inside`/`within` aliases using
   the FELN library's canonical relation kinds. Independent canonical-target checks
   reject equivalent FELNs crossing splits. The earlier v5 grouping retained units;
   its early training attempt was stopped, and its exploratory validation scores are
   not final evaluation results. v6 retrains from the stock base.
8. Validate all labels, checksums, text uniqueness, and group disjointness. Keep zero-row
   queries: result counts are not an accuracy signal.

Measured split counts: 4,790 train / 673 validation / 534 test; 3,393 query families.
All 534 test reference FELNs executed through `SpatialQuery` against the approved
Orin database copy with zero errors in 57.1397 seconds (measured;
`evidence/reference-spatial-test-v6.json`). SHA-256 before and after matched. This
checks label executability, not model accuracy, and does not remove zero-row queries.

All nine primary-layer/layer-count strata appear in each split. Each record preserves
source file plus record index or Markdown line, and generated examples preserve seed
and generator index. Manifest hashes cover inputs and all generated artifacts.

## Training and evaluation design

BF16 LoRA, rather than QLoRA: each measured GPU has about 96 GiB VRAM, so 9B frozen
BF16 weights plus trainable adapters and activations fit comfortably. NF4 adds a
dependency and quantization error without solving a measured capacity problem.
DDP uses two complete replicas, one per GPU. Rank 32, alpha 64, all text linear layers,
dropout 0, assistant-only loss, maximum sequence length 2,048; overlength labels fail
instead of truncating. Batch 2 per GPU × accumulation 4 × two GPUs = 16 effective.
AdamW fused, LR 2e-4, cosine decay, 30 warmup steps, two epochs, seed 20260917.
Validation loss selects the best saved epoch; held-out generation reports the strict
FELN comparator both before and after the app's schema compiler. Neither compiler nor
judge is changed to improve a score. Checkpoints are retained, never overwritten.

Smoke run: four optimizer steps on 32 train/32 validation records, explicit `--smoke`.
An initial run exposed a Transformers API change (`warmup_ratio` removed); this was
fixed to `warmup_steps`. A retry accidentally omitted `--smoke`; it was interrupted
early, retained as `smoke-v2`, and is not counted as a completed run. `smoke-v3` is the
explicit smoke test for the earlier split. After correcting the distance-unit grouping,
`smoke-v6` reran the four-step smoke on the final dataset before `full-v6` began.

`smoke-v3` passed: 33.7069 seconds, 4 optimizer steps, train loss 0.484356,
25,066,128,384-byte rank-0 peak allocated and 38,541,459,456-byte peak reserved
(all measured; `evidence/smoke-report.json`). The final `smoke-v6` passed in 30.4099
seconds with train loss 0.626291, rank-0 peak allocated 24,598,767,616 bytes and peak
reserved 27,887,927,296 bytes (measured; `evidence/smoke-v6-report.json`).
`full-v6` started only after this successful smoke and the stock-model Orin app query.
`full-v6` completed all 600 steps and two epochs. Measured training-and-save time
was 3,500.0459 seconds, train loss 0.0142053, rank-0 peak allocated
25,594,151,424 bytes and peak reserved 50,342,133,760 bytes. Maximum observed
sequence lengths were 924 train and 869 validation. Checkpoint 600 was selected:
validation loss 0.0009413157 versus 0.0027208291 at checkpoint 300. The adapter is
346,294,736 bytes, SHA-256
`e60a7c4abe38383c0e6d3e3f6525779242fef8dd4d0a6e1a6842741bfdd7a76b`.
See `evidence/full-v6-report.json` and `evidence/full-v6-trainer-state.json`.

Export merges adapter deltas in FP32 and stores FP16 before llama.cpp conversion and
Q4_K_M quantization. The pinned tokenizer produces the app's exact prefix/suffix,
including disabled thinking. Evaluation saves every output with expected FELN,
provenance, validity, strict raw/compiled result, and timing.

## Source-data transfer approval

Automatic approval review blocked copying the full NorthSea.ddb database to Orin,
requiring explicit authorization for that payload/destination.
The user explicitly approved the exact copy in the conversation. It is
23,080,960 bytes, SHA-256
`bf176393f6745628d084fdd0564be8399507222215a44dc6a66bfe77cb94d6c8`, to
`/data/feln-qwen-20260917/NorthSea.ddb`. The originals remain read-only. FELN generation
and model evaluation do not depend on this database transfer; spatial execution does.

## Export and generated-output results

First-epoch checkpoint 300 measured validation loss 0.0027208291 over 673
validation examples in 40.3774 seconds. This is teacher-forced token loss, not
generated FELN accuracy (`evidence/checkpoint-300-state.json`).

The first tuned HF generation attempt failed in an FP32 adapter matrix multiply.
An isolated 128×128 CUDA matrix multiply reproduced `CUBLAS_STATUS_NOT_INITIALIZED`
for FP32, while BF16 passed in the existing environment. Generation now uses BF16
autocast, matching training; the shared environment and NVIDIA libraries are untouched.
The failed `tuned-hf-val-v6` output is retained; the retry uses
`tuned-hf-val-v6-bf16` and `tuned-hf-test-v6-bf16`, with `training/evaluate-v2.py`
on RTX corresponding to the repository evaluator. This is an environment limitation,
not evidence of a failed training run.

The first GGUF advertised an optional 33rd MTP block absent from the text-only
HF class; loading failed with `blk.32.attn_norm.weight not found`. Export now sets
`mtp_num_hidden_layers=0`, and conversion uses its supported `--no-mtp` option.
The corrected `tuned-v6-text-q4_k_m.gguf` successfully loads on the RTX runtime.
The earlier GGUF and interrupted copies are retained as failed artifacts, never
selected for deployment. The zero-MTP configuration alone triggered a converter
assertion; the explicit converter flag is required and documented.

Tuned HF validation measured 672/673 raw and compiled exact, with 672 schema-valid
outputs (`evidence/tuned-hf-val-v6-bf16.json`). The remaining failure represented
`multilateral` as numeric 1 instead of the catalog string YES. These are generated
outputs, distinct from the teacher-forced loss reported above.

The full 534-case HF test measured 69/534 raw exact and 100/534 compiled exact
for the untuned base; the tuned adapter measured 533/534 for both, with all 534
outputs schema-valid. Source slices: Layers-generated 347/347, OKF-derived 29/29,
and FELN-source 157/158 for the tuned model. Its sole test error dropped a closing
parenthesis inside the literal `COMPLETED (SHUT IN)`. Reports are
`evidence/baseline-hf-test-v6.json` and `evidence/tuned-hf-test-v6-bf16.json`.
HF uses unconstrained greedy generation; GGUF/app decoding adds the original
JSON grammar, so matched Orin stock/tuned results are reported separately.

The corrected Q4 artifact measured 673/673 raw exact, compiled exact, and
schema-valid on the full validation set (`evidence/tuned-text-q4-val-v6.json`).
The combined export/runtime/grammar path therefore showed no validation accuracy
loss relative to the HF adapter. This does not isolate quantization from grammar
or runtime effects. Q4_K_M was selected using validation; no test result selects
quantization. Q8 was unnecessary because Q4 passed this gate. The temporary RTX
llama-server was stopped after validation.

The matched untuned Orin Q4 artifact completed all 534 test questions: 96/534
raw exact, 129/534 compiled exact, and 269/534 schema-valid; warm median 4.1229
seconds and p95 8.0294 seconds (measured; `evidence/baseline-orin-test-v6.json`).
This is the same grammar, prompt, device, and test record order used for the final
artifact comparison, unlike the unconstrained HF baseline.

The matched tuned Q4 RTX test measured 530/534 raw and compiled exact (99.25%),
532/534 schema-valid, warm median 0.4015705 seconds, warm p95 0.6435170 seconds,
and 2.42622 requests/s. All 534 questions completed in 220.0951 seconds using one
RTX GPU. Two outputs lost a parenthesis inside a status literal; two truncated
`field_current_activity_status`. The report and errors are
`evidence/tuned-rtx-q4-test-v6.json` and `evidence/tuned-rtx-q4-errors-v6.json`.
The completed task-owned server was stopped; `nvidia-smi` then reported zero MiB
used on both GPUs. Final Orin inference does not depend on that process.

## Final Orin results and deployment

The final GGUF SHA-256 matches the RTX artifact. Orin measured **530/534 raw and
compiled exact (99.25%)**, versus the controlled untuned Orin baseline's 96/534
(17.98%) raw and 129/534 (24.16%) compiled. This is an 81.27 percentage-point
strict-accuracy improvement. Both RTX and Orin produced byte-identical raw strings
for all 534 test prompts, including the four errors. Both produced 532 schema-valid
outputs. The combined export/quantized-runtime path scored three fewer exact test
responses than the HF adapter; validation was 673/673 and selected Q4 before testing.
No test result was used to change quantization or training.

The matched workload uses the same artifact, prompt, JSON grammar, test order,
4,096-token context, one slot, and sequential localhost HTTP. RTX uses one GPU;
the runtime's default CPU thread counts are 24 RTX / 12 Orin. Runtime revisions
differ as recorded above. Measured results:

| Metric | RTX | Orin |
|---|---:|---:|
| Exact FELN | 530/534 | 530/534 |
| First request | 0.378707 s | 3.079219 s |
| Warm median | 0.401570 s | 3.221432 s |
| Warm p95 | 0.643517 s | 5.167540 s |
| Requests/s | 2.426222 | 0.302663 |
| Median generated tokens/s | 184.4962 | 23.0200 |
| Elapsed for 534 requests | 220.0951 s | 1,764.3366 s |

RTX's median response is 8.02× faster for this workload. Sources:
`evidence/tuned-rtx-q4-test-v6.json`, `evidence/tuned-orin-test-v6.json`, and
`evidence/{rtx,orin}-tuned-runtime-metrics.json`. RTX's loaded, idle allocation
after testing was 5,751 MiB (`nvidia-smi` observation, not peak memory).

Permanent services were deployed on **orin** (private SSH alias):
`feln-qwen-app.service` at `127.0.0.1:18091` and
`feln-qwen-model.service` at `127.0.0.1:18092`.
Both are system units using an unprivileged deployment account (name redacted), enabled for boot. Explicit restart
changed model PID 123578→123809 and app PID 123579→123810; both returned active/running
and health reported local spatial execution ready. No physical device reboot was
performed. See `evidence/orin-service-restart.txt` and
`evidence/orin-final-verification.txt`. The temporary probes are stopped; the only
running llama-server on Orin is this task's permanent model service.
The workstation has no llama-server and both GPUs report zero MiB in
`evidence/rtx-final-idle.txt`. Orin's model, prompt bundle, app, and database are local;
the app's model URL is loopback, and there is no workstation inference connection.

After restart, the app benchmark measured 18/18 raw and compiled exact, zero HTTP
errors, first request 3.318501 s, warm median 3.974140 s, warm p95 4.931088 s,
66.832476 s total, and 0.269330 requests/s. It covers the first two test questions
per primary-layer/layer-count stratum. Its 17 warm samples make p95 the maximum;
use the full 534-case model test for a broader latency distribution. Because the
sample differs, the two medians cannot be subtracted to isolate database cost.
All request details are in `evidence/orin-e2e-benchmark-v6.json`.

At 50 ms intervals over 1,320 samples, **peak system RAM used was 10,807,091,200
bytes (10.06 GiB)** and **minimum available RAM was 55,125,463,040 bytes (51.34 GiB)**
alongside the running app and model. These are measured system-wide shared-memory
pressure, including the OS, not a model-only CUDA allocation or an unsampled absolute
peak. Swap use remained at the pre-existing 1,572,864 bytes. The app cgroup peak was
175,169,536 bytes; the model cgroup peak was 635,809,792 bytes after the extra query,
which excludes much of CUDA shared allocation and must not replace the system reading.
The measured free headroom supports the estimated 15.5 GiB planning budget; there
is no resource-driven need to fall back to 4B. Larger models remain capacity-feasible,
but their fine-tuned accuracy advantage was not measured.

A separate permanent-service `Show all wells` request returned exactly
`{"layers":["Wells"],"where":[""],"relations":[]}` and one bounded GeoJSON feature
in 1.790707 s of server time, including 0.082801 s spatial execution
(`evidence/orin-final-request.json`). Database SHA-256 still matches the approved
read-only copy. Original NorthSea files were never modified.

Limitations: four known held-out errors, synthetic/source language and a fixed
catalog, one concurrent request, multi-second full responses, an HTTP adapter using
the original application core rather than a new UI, the reference planner's regional
EPSG:32632 distance approximation, and untested physical reboot recovery. New catalogs,
real-user language, and fine-tuned larger alternatives require separate evaluation.

## Verification notes

The subsequent [FELN Studio deployment](STUDIO.md) adds the existing browser
generation UI on Orin port 8766, using this same model service and prompt bundle.
It passed nine representative live requests without changing or restarting the
model or original API. The benchmark results above remain the original pipeline
measurements; Studio's integration measurements are recorded separately.

The local contract checks and Ruff passed. CodeRabbit 0.7.6 was verified against
the official release archive, but automatic approval review blocked sending private
source diffs to its external API. No CodeRabbit review ran; source inspection is local.
The Orin benchmark update preserves the initial file as `benchmark.py` and installs
the revised version as `benchmark-v2.py`, avoiding an unapproved overwrite.
