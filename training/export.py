"""Merge in FP32, save FP16 text-only model and the application's exact prompt bundle."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from src.edge_client import output_schema
from src.feln_data import Schema
from src.prompt import make_prompt
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

p = argparse.ArgumentParser()
p.add_argument("--base", type=Path, required=True)
p.add_argument("--adapter", type=Path)
p.add_argument("--schema", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--bundle-only", action="store_true")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
tokenizer = AutoTokenizer.from_pretrained(a.base)
schema = Schema(a.schema)
if not a.bundle_only:
    model = Qwen3_5ForCausalLM.from_pretrained(a.base, dtype=torch.float32)
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload(
            safe_merge=True
        )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    # The text-only HF class does not load/export the optional MTP predictor.
    # Its inherited metadata must not make the GGUF loader require a 33rd block.
    model.config.mtp_num_hidden_layers = 0
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.to(torch.float16).save_pretrained(a.output)
tokenizer.save_pretrained(a.output)
sentinel = "FELN_QUERY_PLACEHOLDER_9b41"
prefix, suffix = make_prompt(sentinel, tokenizer, schema.context()).split(sentinel)
with (a.output / "inference_config.json").open("x") as f:
    json.dump(
        {
            "prompt_prefix": prefix,
            "prompt_suffix": suffix,
            "max_new_tokens": 512,
            "json_schema": output_schema(list(schema.layers)),
        },
        f,
        indent=2,
    )
shutil.copy2(a.schema, a.output / "Layers.json")
