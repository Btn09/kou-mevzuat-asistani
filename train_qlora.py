#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-8B'yi raft_train.jsonl ile QLoRA (4-bit + LoRA) yöntemiyle eğitir.

Sadece LoRA adaptörü eğitilir (~40M parametre); 8B ana model 4-bit olarak donuk kalır.
Loss sadece asistan cevabı üzerinden hesaplanır: model soruyu/bağlamı değil,
bağlamı kullanarak cevap vermeyi öğrenir.

Kurulum (ayrı sanal ortam; global PyTorch nightly'ye dokunmaz):
    python -m venv --system-site-packages .venv-train
    .venv-train\\Scripts\\python -m pip install peft trl bitsandbytes

Kullanım:
    .venv-train\\Scripts\\python train_qlora.py            # eğit
    .venv-train\\Scripts\\python train_qlora.py --kontrol  # sadece veri/token kontrolü, eğitim yok

İlk çalıştırmada Qwen/Qwen3-8B (~16 GB) Hugging Face'ten indirilir.
Çıktı: qwen3-8b-kou-lora/ (LoRA adaptörü). Sonraki adım: export_ollama.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

# ------------------------- AYARLAR -------------------------
BASE_MODEL = "Qwen/Qwen3-8B"
TRAIN_JSONL = Path("raft_train.jsonl")
TEST_JSONL = Path("raft_test.jsonl")
OUTPUT_DIR = Path("qwen3-8b-kou-lora")

MAX_SEQ_TOKENS = 4096  # Bundan uzun örnekler atlanır (4 madde bağlamı genelde ~1500-3000 token).
EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM = 8  # Efektif batch = 8
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SEED = 42
# Windows'ta VRAM dolunca sürücü fazlasını sessizce sistem RAM'ine taşır (sysmem fallback) ve
# eğitim 2-3 kat yavaşlar. PyTorch bunu bilmediği için önbelleği boşaltmaz; bu sınır onu
# önbelleği yeniden kullanmaya zorlar. Ölçülen gerçek tepe kullanım ~10.6 GB.
MAX_VRAM_FRACTION = 0.8

# ---- v2: çok belgeli sistem + Trendyol-LLM-8B-T1 (python train_qlora.py --v2) ----
V2 = "--v2" in sys.argv[1:]
MAX_TRAIN_EXAMPLES = None  # v1: hepsi
MAX_EXAMPLES_PER_UNIT = None
MAX_EVAL_EXAMPLES = None
STEP_EVAL = None  # v1: epoch sonunda değerlendirme
if V2:
    _EGITIM = Path(__file__).resolve().parent.parent / "veri" / "egitim"
    BASE_MODEL = "Trendyol/Trendyol-LLM-8B-T1"  # Qwen3-8B tabanlı, aynı sohbet şablonu
    TRAIN_JSONL = _EGITIM / "raft_train.jsonl"
    TEST_JSONL = _EGITIM / "raft_test.jsonl"
    OUTPUT_DIR = Path("trendyol-8b-kou-lora")
    # ~10 bin örneğin tamamıyla 3 epoch ~20 saat sürer. Model bilgi değil davranış öğreniyor
    # (kaynak gösterme, doğru maddeyi seçme, yönlendirme); dengeli ~4000 örnek ve 1 epoch yeterli.
    EPOCHS = 1
    MAX_TRAIN_EXAMPLES = 4000
    MAX_EXAMPLES_PER_UNIT = 4  # aynı maddeden en fazla bu kadar cevaplı örnek (madde çeşitliliği)
    MAX_EVAL_EXAMPLES = 200  # her değerlendirme ~4-5 dk sürsün
    # Bu PyTorch/Blackwell kurulumunda dikkat belleği uzunluğun karesiyle büyüyor (verimli çekirdek yok):
    # 2683 token 10.6 GB, 4086 token 15.7 GB ölçüldü. 3072 token ilk adımda OOM verdi (parçalanma payı
    # yetmedi); 2816 token ~11 GB. Daha uzun örneklerde dikkat dağıtıcı maddeler çıkarılır (fit_to_length).
    MAX_SEQ_TOKENS = 2816
    MAX_VRAM_FRACTION = 0.85  # eğitim sırasında GPU'yu başka uygulama kullanmıyor (~0.3 GB)
    STEP_EVAL = 50  # PC kapanırsa (Windows Update) en fazla ~25 dk kaybolsun diye sık ara kayıt

# ---- v3: v2 adaptöründen devam eğitimi (python train_qlora.py --v2 --v3) ----
# Yeni maddeler + devam sorusu örnekleri + v2 verisinden tekrar örneklemi (generate_raft_data_v3.py).
# Sıfırdan eğitim (~5 saat) yerine v2'nin öğrendiği davranış korunup yalnızca yeni beceriler eklenir.
V3 = V2 and "--v3" in sys.argv[1:]
INIT_ADAPTER: Path | None = None
if V3:
    _EGITIM_V3 = Path(__file__).resolve().parent.parent / "veri" / "egitim_v3"
    TRAIN_JSONL = _EGITIM_V3 / "raft_train_v3.jsonl"
    TEST_JSONL = _EGITIM_V3 / "raft_eval_v3.jsonl"
    OUTPUT_DIR = Path("trendyol-8b-kou-lora-v3")
    INIT_ADAPTER = Path("trendyol-8b-kou-lora")
    MAX_TRAIN_EXAMPLES = None  # örnekleme üreticide yapıldı
    MAX_EXAMPLES_PER_UNIT = None
    LEARNING_RATE = 1e-4  # devam eğitimi: v2'nin yarısı (öğrenileni bozmadan ince ayar)

# ---- v4: tam veri + ayrıntılı cevaplarla SIFIRDAN (python train_qlora.py --v2 --v4) ----
# v3'ün devam eğitimi %78 -> %72 düşürdü; v4 Trendyol-LLM-8B-T1 tabanından başlar (generate_raft_data_v4.py).
V4 = V2 and "--v4" in sys.argv[1:]
if V4:
    _EGITIM_V4 = Path(__file__).resolve().parent.parent / "veri" / "egitim_v4"
    TRAIN_JSONL = _EGITIM_V4 / "raft_train_v4.jsonl"
    TEST_JSONL = _EGITIM_V4 / "raft_eval_v4.jsonl"
    OUTPUT_DIR = Path("trendyol-8b-kou-lora-v4")
    MAX_TRAIN_EXAMPLES = None  # örnekleme üreticide yapıldı
    MAX_EXAMPLES_PER_UNIT = None

# ---- v5: v4 verisi güncel veri/bağlamlarla + sayı/tarih/grup yoğun yeni örnekler, SIFIRDAN
# (python train_qlora.py --v2 --v5; generate_raft_data_v5.py)
V5 = V2 and "--v5" in sys.argv[1:]
if V5:
    _EGITIM_V5 = Path(__file__).resolve().parent.parent / "veri" / "egitim_v5"
    TRAIN_JSONL = _EGITIM_V5 / "raft_train_v5.jsonl"
    TEST_JSONL = _EGITIM_V5 / "raft_eval_v5.jsonl"
    OUTPUT_DIR = Path("trendyol-8b-kou-lora-v5")
    MAX_TRAIN_EXAMPLES = None
    MAX_EXAMPLES_PER_UNIT = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def tokenize_example(tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, list[int]]:
    """Prompt kısmını -100 ile maskeler; loss sadece asistan cevabında hesaplanır.

    Qwen3'te düşünme modu kapalı (enable_thinking=False): prompt '<think>\\n\\n</think>\\n\\n'
    ile biter. Canlı sistemde de aynı şekilde çağrılmalı.
    """
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    answer_text = messages[-1]["content"].strip() + "<|im_end|>\n"
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    return {
        "input_ids": prompt_ids + answer_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(answer_ids)),
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


def sample_rows(rows: list[dict[str, Any]], limit: int | None, per_unit: int | None) -> list[dict[str, Any]]:
    """Reddetme/kapsam dışı örneklerin hepsi kalır; cevaplı örnekler madde başına sınırlanıp örneklenir."""
    import random

    if limit is None or len(rows) <= limit:
        return rows
    rng = random.Random(SEED)
    rows = rows[:]
    rng.shuffle(rows)
    keep = [r for r in rows if r.get("meta", {}).get("type") != "oracle"]
    per_count: dict[str, int] = {}
    oracle = []
    for row in rows:
        if row.get("meta", {}).get("type") != "oracle":
            continue
        unit = row["meta"].get("unit_id", "")
        if per_unit is None or per_count.get(unit, 0) < per_unit:
            per_count[unit] = per_count.get(unit, 0) + 1
            oracle.append(row)
    selected = keep + oracle[:max(0, limit - len(keep))]
    rng.shuffle(selected)
    return selected


_ARTICLES: dict[str, dict[str, Any]] | None = None


def fit_to_length(tokenizer: Any, row: dict[str, Any]) -> dict[str, list[int]]:
    """Uzun örneği atmak yerine bağlamdaki dikkat dağıtıcı maddeleri (sondan) çıkarıp sığdırır.

    Doğru madde (unit_id) her zaman kalır. Mesaj ortak formatla (raft_format) yeniden kurulur.
    Sadece v2 verisinde (meta.context_ids) uygulanır; en az bir madde kalır.
    """
    global _ARTICLES
    example = tokenize_example(tokenizer, row["messages"])
    meta = row.get("meta", {})
    if len(example["input_ids"]) <= MAX_SEQ_TOKENS or "context_ids" not in meta:
        return example
    from raft_format import SYSTEM_PROMPT_V2, build_messages

    if _ARTICLES is None:
        path = Path(__file__).resolve().parent.parent / "veri" / "islenmis" / "maddeler.jsonl"
        _ARTICLES = {a["id"]: a for a in read_jsonl(path)}
    context = [_ARTICLES[i] for i in meta["context_ids"] if i in _ARTICLES]
    answer = row["messages"][-1]["content"]
    while len(context) > 1:
        removable = [i for i, a in enumerate(context) if a["id"] != meta.get("unit_id")]
        if not removable:
            break
        context.pop(removable[-1])
        messages = build_messages(meta["question"], context, answer, system=SYSTEM_PROMPT_V2)
        example = tokenize_example(tokenizer, messages)
        if len(example["input_ids"]) <= MAX_SEQ_TOKENS:
            break
    return example


def build_dataset(tokenizer: Any, path: Path, limit: int | None = None,
                  per_unit: int | None = None) -> list[dict[str, list[int]]]:
    rows = sample_rows(read_jsonl(path), limit, per_unit)
    examples = [fit_to_length(tokenizer, row) for row in rows]
    kept = [example for example in examples if len(example["input_ids"]) <= MAX_SEQ_TOKENS]
    lengths = sorted(len(example["input_ids"]) for example in examples)
    print(
        f"{path}: {len(examples)} örnek, token min/medyan/max = "
        f"{lengths[0]}/{lengths[len(lengths) // 2]}/{lengths[-1]}, "
        f"{len(examples) - len(kept)} örnek {MAX_SEQ_TOKENS} token sınırını aştığı için atlandı"
    )
    return kept


def answer_only_loss(model: Any, inputs: dict[str, torch.Tensor], num_items_in_batch: Any = None) -> torch.Tensor:
    """Sadece cevap pozisyonlarında lm_head çalıştırıp cross-entropy hesaplar.

    Standart forward tüm ~1700 pozisyon için 152k'lık sözlükte logit üretir (fp32 ~1.6 GB,
    gradyanlarıyla ~4 GB). 16 GB kartta bu VRAM'i taşırıp Windows'u sistem RAM'ine
    düşürüyor ve eğitimi ~6 kat yavaşlatıyordu. Loss zaten sadece cevaptaki ~50 token
    üzerinden hesaplandığı için sonuç aynıdır.
    """
    causal_lm = model.get_base_model()
    hidden = causal_lm.model(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
    ).last_hidden_state
    # t pozisyonundaki gizli durum t+1'deki token'ı tahmin eder.
    targets = inputs["labels"][:, 1:]
    mask = targets != -100
    logits = causal_lm.lm_head(hidden[:, :-1][mask]).float()
    # Gradient biriktirmede Trainer loss'u kendisi bölmüyor (PEFT modeli loss kwargs kabul ediyor);
    # birikimdeki toplam cevap token sayısına bölmezsek loss ve etkili öğrenme oranı 8 kat çıkar.
    if num_items_in_batch is not None:
        loss_sum = torch.nn.functional.cross_entropy(logits, targets[mask], reduction="sum")
        return loss_sum / torch.as_tensor(num_items_in_batch, device=loss_sum.device)
    return torch.nn.functional.cross_entropy(logits, targets[mask])


def main() -> int:
    for path in (TRAIN_JSONL, TEST_JSONL):
        if not path.exists():
            print(f"{path} bulunamadı. Önce: python generate_raft_data.py")
            return 1

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_data = build_dataset(tokenizer, TRAIN_JSONL, MAX_TRAIN_EXAMPLES, MAX_EXAMPLES_PER_UNIT)
    eval_data = build_dataset(tokenizer, TEST_JSONL, MAX_EVAL_EXAMPLES)

    if "--kontrol" in sys.argv[1:]:
        sample = train_data[0]
        print("\n--- Örnek: modelin gördüğü tam metin ---")
        print(tokenizer.decode(sample["input_ids"]))
        print("--- Loss hesaplanan kısım (sadece cevap) ---")
        print(tokenizer.decode([t for t, l in zip(sample["input_ids"], sample["labels"]) if l != -100]))
        return 0

    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    torch.cuda.set_per_process_memory_fraction(MAX_VRAM_FRACTION, 0)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    # prepare_model_for_kbit_training kullanılmıyor: 4-bit olmayan embedding ve lm_head'i fp32'ye
    # yükseltiyor (Qwen3-8B'de ~5 GB) ve 16 GB kartı taşırıyordu. Gereken iki ayar elle yapılıyor.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if INIT_ADAPTER is not None:
        from peft import PeftModel

        print(f"Önceki adaptörden devam: {INIT_ADAPTER}")
        model = PeftModel.from_pretrained(model, str(INIT_ADAPTER), is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=LORA_TARGETS,
                task_type="CAUSAL_LM",
            ),
        )
    model.print_trainable_parameters()

    class AnswerOnlyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = answer_only_loss(model, inputs, num_items_in_batch)
            return (loss, {"loss": loss}) if return_outputs else loss

    total_steps = max(1, len(train_data) * EPOCHS // (BATCH_SIZE * GRAD_ACCUM))
    schedule = ({"eval_strategy": "steps", "eval_steps": STEP_EVAL, "save_strategy": "steps",
                 "save_steps": STEP_EVAL} if STEP_EVAL else {"eval_strategy": "epoch", "save_strategy": "epoch"})
    trainer = AnswerOnlyTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            warmup_steps=max(1, total_steps // 20),  # ~%5
            bf16=True,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=5,
            prediction_loss_only=True,  # Değerlendirmede tüm logitleri toplamaya gerek yok.
            **schedule,
            save_total_limit=2,
            load_best_model_at_end=True,  # Aşırı öğrenme olursa en iyi epoch'u geri yükler.
            metric_for_best_model="eval_loss",
            report_to="none",
            seed=SEED,
            dataloader_num_workers=0,  # Windows'ta çoklu worker sorun çıkarıyor.
        ),
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    checkpoints = list((OUTPUT_DIR / "checkpoints").glob("checkpoint-*"))
    if checkpoints:
        print(f"Önceki ara kayıttan devam ediliyor ({len(checkpoints)} ara kayıt bulundu).")
    trainer.train(resume_from_checkpoint=bool(checkpoints))
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nLoRA adaptörü kaydedildi: {OUTPUT_DIR.resolve()}")
    print("Sonraki adım: python export_ollama.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
