#!/usr/bin/env python3
"""Script d'entraînement LoRA (Supervised Fine-Tuning) pour le Fixer Freestyle.

Adapte Qwen2.5-0.5B-Instruct sur le corpus Night City pour lui insuffler le cynisme,
l'argot de rue et l'adaptation aux 4 statuts tout en conservant sa capacité d'improvisation libre.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints" / "fixer_freestyle_lora"
DATASET_PATH = DATA_DIR / "freestyle_cyberpunk_dialogues.jsonl"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

SEED = 2077
torch.manual_seed(SEED)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Dispositif d'entraînement : {device}")


class SFTCyberpunkDataset(Dataset):
    """Dataset avec masquage de la perte (Loss Masking) sur le prompt."""

    def __init__(self, data_path: Path, tokenizer, max_length: int = 256) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[dict[str, torch.Tensor]] = []

        with open(data_path, "r", encoding="utf-8") as f:
            raw_dialogues = [json.loads(line) for line in f]

        for item in raw_dialogues:
            messages = item["messages"]
            # Messages système + utilisateur (sans la réponse)
            prompt_messages = messages[:2]

            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            full_tokens = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )["input_ids"][0]

            prompt_tokens = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )["input_ids"][0]

            # Masquage : les tokens du prompt sont ignorés (-100) dans le calcul de cross-entropy
            labels = full_tokens.clone()
            prompt_len = min(len(prompt_tokens), len(full_tokens))
            labels[:prompt_len] = -100

            self.samples.append(
                {
                    "input_ids": full_tokens,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


def collate_fn(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].size(0) for x in batch)
    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        inp = item["input_ids"]
        lbl = item["labels"]
        pad_len = max_len - inp.size(0)

        padded_inp = torch.cat([inp, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        padded_lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
        mask = torch.cat([torch.ones_like(inp), torch.zeros(pad_len, dtype=torch.long)])

        input_ids.append(padded_inp)
        labels.append(padded_lbl)
        attention_mask.append(mask)

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def train_lora(
    steps: int = 150,
    batch_size: int = 4,
    grad_accum_steps: int = 2,
    lr: float = 3e-4,
) -> None:
    print(f"\n1. Chargement du modèle de base : {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)

    print("2. Configuration de l'adaptateur LoRA...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, peft_config)
    model.to(device)
    model.print_trainable_parameters()

    print(f"\n3. Préparation du dataset SFT depuis {DATASET_PATH}...")
    dataset = SFTCyberpunkDataset(DATASET_PATH, tokenizer, max_length=256)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id or tokenizer.eos_token_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f"\n4. Démarrage de l'entraînement LoRA ({steps} pas effectifs)...")
    model.train()
    step = 0
    accum_loss = 0.0
    start_time = time.perf_counter()

    data_iter = iter(loader)
    while step < steps:
        optimizer.zero_grad()
        for _ in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        step += 1

        if step % 25 == 0 or step == steps:
            avg_loss = accum_loss / 25 if step % 25 == 0 else accum_loss
            accum_loss = 0.0
            print(f"Pas {step:>4}/{steps} | Loss SFT: {avg_loss:.4f}")

    elapsed = time.perf_counter() - start_time
    print(f"\n✓ Entraînement terminé en {elapsed:.1f}s ({elapsed / steps * 1000:.1f} ms/pas)")

    # Sauvegarde des poids LoRA
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)
    print(f"✓ Poids LoRA sauvegardés avec succès dans : {CHECKPOINT_DIR}")


if __name__ == "__main__":
    train_lora(steps=150, batch_size=4, grad_accum_steps=2, lr=3e-4)
