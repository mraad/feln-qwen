"""BF16 LoRA on two GPUs; immutable outputs and assistant-only loss."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from src.feln_data import Schema
from src.prompt import build_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Qwen3_5ForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    if rank == 0:
        a.output.mkdir(parents=True, exist_ok=False)
    set_seed(20260917)
    tokenizer = AutoTokenizer.from_pretrained(a.base)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    context = Schema(a.data / "Layers.json").context()
    rows = {s: json.loads((a.data / f"{s}.json").read_text()) for s in ("train", "val")}
    if a.smoke:
        rows = {s: v[:32] for s, v in rows.items()}
    data = {s: build_dataset(v, tokenizer, 2048, context) for s, v in rows.items()}
    model = Qwen3_5ForCausalLM.from_pretrained(
        a.base, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    model.enable_input_require_grads()
    config = TrainingArguments(
        output_dir=str(a.output),
        num_train_epochs=2,
        max_steps=4 if a.smoke else -1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=1 if a.smoke else 30,
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        optim="adamw_torch_fused",
        logging_steps=1 if a.smoke else 10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=None,
        report_to="none",
        seed=20260917,
        data_seed=20260917,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=config,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, pad_to_multiple_of=8
        ),
    )
    started = time.monotonic()
    result = trainer.train()
    trainer.save_model(str(a.output / "adapter"))
    trainer.save_state()
    if rank == 0:
        tokenizer.save_pretrained(a.output / "adapter")
        report = {
            "metrics": result.metrics,
            "seconds": time.monotonic() - started,
            "peak_allocated_bytes_rank0": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes_rank0": torch.cuda.max_memory_reserved(),
            "max_tokens": {
                s: max(len(r["input_ids"]) for r in v) for s, v in data.items()
            },
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "model_id": "Qwen/Qwen3.5-9B",
            "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            "tokenizer_id": "Qwen/Qwen3.5-9B",
            "tokenizer_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "training_args": config.to_dict(),
        }
        with (a.output / "report.json").open("x") as f:
            json.dump(report, f, indent=2)
        print(
            json.dumps({k: v for k, v in report.items() if k != "training_args"}),
            flush=True,
        )


if __name__ == "__main__":
    main()
