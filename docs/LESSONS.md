# Lessons from the FELN Qwen pipeline

These lessons describe observed results from this run, not hypothetical issues.

## Measure hardware before choosing precision or model size

The workstation has two **RTX PRO 6000 Blackwell Server Edition** GPUs with
97,887 MiB each, rather than a capacity inferred from the phrase “RTX 6000”.
That made BF16 LoRA practical without QLoRA. The Orin has 65,932,554,240 bytes of
shared RAM. Its existing CUDA llama.cpp build already ran Qwen, so replacing
JetPack, CUDA, or system packages was unnecessary. See the inventory and runtime
outputs in [evidence](evidence).

Larger models can fit: the 27B Q4 model loaded on Orin. Its 19/90 strict validation
matches and 14.35 s median latency did not justify it against tuned 9B Q4's 90/90
on those same selection questions. This does **not** establish that a fine-tuned
27B or 35B would be less accurate. See [model selection](PIPELINE.md).

## Learn the application's contract before generating examples

FELN is aligned layer/filter arrays with spatial relations anchored to the primary
layer. Generic chat examples would miss enum codes, typed filters, case-sensitive
literals, negative distance predicates, and relation direction. Reuse the pinned
application's renderer, schema compiler, prompt, client, and spatial planner.
`RTX-Qwen` was not a defined model in the application.

OKF exposes seven Wells fields absent from the app catalog. Record that disagreement
and exclude unsupported outputs; do not silently expand the app's schema. Preserve
every source location and before/after label transformation.

## Leakage grouping needs semantic normalization

Text deduplication alone is insufficient. Group paraphrases, literal substitutions,
secondary-filter permutations, distance-unit variants, and `inside`/`within` aliases.
An earlier split retained distance units in its family key; its early training run
was stopped and retained. Final v6 data was rebuilt and retrained from the stock base.

The final checks independently reject overlapping normalized texts, query families,
and canonical FELN targets. They passed locally and on both staged copies. Keep these
checks before the full training run; do not select splits using model test results.
See [contract check](evidence/contract-check-v6.txt) and
[staged verification](evidence/staged-data-verification.txt).

## A stock runtime probe does not prove the custom export path

Stock Qwen inference and an app/database request worked on Orin before full training.
The fine-tuned text-only export nevertheless exposed a separate MTP issue:
`Qwen3_5ForCausalLM` did not export the optional predictor, while inherited metadata
advertised a 33rd block. The runtime rejected the missing `blk.32.attn_norm.weight`.

The fix is **both** `mtp_num_hidden_layers=0` in the exported configuration and the
converter's supported **`--no-mtp`** option. Zero metadata alone triggered a converter
assertion. Keep failed files intact and use new output paths. The corrected Q4 file
loaded and scored 673/673 on validation. For subsequent runs, take the smoke adapter
through merge, conversion, load, and a short app check before full training.
See [failure log](evidence/gguf-mtp-load-failure.log),
[corrected config](evidence/merged-text-config.json), and
[Q4 validation](evidence/tuned-text-q4-val-v6.json).

## Match evaluation precision to training

The existing workstation environment failed an isolated FP32 CUDA matrix multiply
with `CUBLAS_STATUS_NOT_INITIALIZED`; BF16 passed. PEFT stores adapter weights in
FP32, so loading a BF16 base alone did not make every evaluation operation BF16.
Wrapping generation in BF16 autocast matched the successful training configuration
and completed evaluation. No shared packages or drivers were changed.
See the [failed attempt](evidence/hf-fp32-adapter-failure.log) and
[successful validation](evidence/tuned-hf-val-v6-bf16.json).

## Report semantic failures even when JSON is valid

The tuned HF model scored 533/534 on test. Its missed status literal lost a closing
parenthesis, producing valid JSON and a valid filter with the wrong meaning.
Its separate validation miss emitted numeric 1 instead of the string YES for
`multilateral`. Neither is a success. Report raw exactness and post-compiler exactness
separately, and preserve every prediction.

The exported Q4 model's RTX test scored 530/534 despite 673/673 validation. Two
responses lost that status parenthesis; two truncated the field name
`field_current_activity_status`. The latter failed schema validation. Perfect
validation is not perfect held-out accuracy, and the HF result cannot stand in for
the deployment artifact. See the [four saved failures](evidence/tuned-rtx-q4-errors-v6.json).

All 534 reference FELNs executed locally with zero errors and an unchanged database
hash. Only 276 returned a nonempty bounded result. Retain zero-result queries: row
count is not evidence of language-model correctness. See
[reference execution](evidence/reference-spatial-test-v6.json).

## Compare like-for-like performance

Use the same GGUF hash, question order, prompt, grammar, context, and concurrency
for RTX/Orin comparisons. Report runtime revisions. HF batches of eight are not
single-request HTTP timings. Model-only HTTP latency is not app-plus-database latency.
Record first-request and warm latency separately, and label every value measured or
estimated. On Orin, sample shared system RAM; a process cgroup or CUDA allocator
counter alone does not capture all device memory pressure.

The completed matched run produced identical raw strings for all 534 questions on
both hosts and the same 530/534 exact score. RTX's 0.402 s median versus Orin's
3.221 s is an approximately 8× speed difference for this workload, with no observed
output difference. That result is specific to the pinned builds and settings.
The app/database path separately passed 18/18, with 3.974 s median and 10.81 GB
sampled system RAM usage at peak. See [the measured tables](../README.md).

## Verify persistence and preserve existing work

The initial user-systemd probe disappeared after logout because the serving user had
no lingering user manager. Use task-owned system services running as the unprivileged deployment account (name redacted);
verify enabled state and explicit service restarts. Do not claim a physical device
reboot unless tested. DuckDB also required `pytz` on the real spatial execution path;
it is pinned in the isolated serving environment.

At verification time, both permanent units were enabled and active. They were
later stopped at the user's request; the units remain enabled for boot.
Explicit restart during verification changed both
process IDs; health and the subsequent spatial benchmark passed. Database hashes
still match. This verifies service restart and boot configuration, while a physical
device reboot remains untested. See [restart evidence](evidence/orin-service-restart.txt)
and [final verification](evidence/orin-final-verification.txt).

A long-idle SSH pipeline stalled at an empty destination. SCP through the Mac with
keepalives transferred data successfully; the exact underlying stall cause was not
established. Check final size and SHA-256 before loading a model. Preserve partial
copies and use fresh filenames when overwrite approval is absent.

The full database copy required explicit authorization and was approved by the user.
It is opened read-only. Automatic approval review blocked an optional CodeRabbit
upload of private diffs; no external review ran. Do not describe a skipped or blocked
review as a pass. Keep training logs, failed exports, and stopped probes' artifacts
for diagnosis; stop only this task's completed services after checking their clients.
