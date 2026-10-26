#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eğitilen LoRA adaptörünü Qwen3-8B ile birleştirir ve Ollama modeli olarak kaydeder.

Adımlar:
  1) Ana model (bf16, CPU/RAM'de) + LoRA -> birleşik model (qwen3-8b-kou-merged/, ~16 GB)
  2) llama.cpp convert_hf_to_gguf.py -> bf16 GGUF (~16 GB)
  3) llama-quantize -> Q4_K_M GGUF (~4.8 GB; ham qwen3:8b ile aynı format)
  4) ollama create kou-asistan  -> sitede çalışacak model

Neden bu yol (Windows, Ollama 0.40):
  - Safetensors'tan doğrudan içe aktarma sadece MLX nicemlemesi (int4/int8...) sunuyor ve
    MLX Windows'ta yok.
  - GGUF içe aktarırken Ollama nicemleme yapmıyor; GGUF önceden nicemlenmiş olmalı.
  - llama-quantize, llama.cpp'nin resmi derlenmiş Windows sürümünden gelir
    (..\\araclar\\llama-bin). Yoksa convert script'inin kendisi Q8_0 üretir (~8.7 GB).

ÖNEMLİ: Ollama, GGUF'un içindeki Qwen3 Jinja şablonunu kullanır. Bu şablon ancak
think=False ile çağrıldığında eğitim formatını birebir üretir. Sitedeki kod Ollama'yı
HER ZAMAN think=False ile çağırmalı (bkz. rag_sorgu.py); aksi halde model düşünme
moduna girip bozuk çıktı verir.

Kurulum (bir kez):
    git clone --depth 1 https://github.com/ggml-org/llama.cpp ..\\araclar\\llama.cpp
    .venv-train\\Scripts\\python -m pip install ..\\araclar\\llama.cpp\\gguf-py
    https://github.com/ggml-org/llama.cpp/releases -> llama-bXXXX-bin-win-cpu-x64.zip
        -> ..\\araclar\\llama-bin\\ klasörüne aç (llama-quantize.exe)

Kullanım:
    .venv-train\\Scripts\\python export_ollama.py

Bittikten sonra qwen3-8b-kou-merged/ ve *.gguf dosyaları silinebilir (Ollama kendi kopyasını tutar).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

# ------------------------- AYARLAR -------------------------
BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_DIR = Path("qwen3-8b-kou-lora")
MERGED_DIR = Path("qwen3-8b-kou-merged")
LLAMA_CPP_DIR = Path("../araclar/llama.cpp")
LLAMA_QUANTIZE = Path("../araclar/llama-bin/llama-quantize.exe")
QUANT_TYPE = "Q4_K_M"  # Sitede ~5 GB; daha yüksek kalite için "Q5_K_M" veya "Q8_0"
MODELFILE = Path("Modelfile")
OLLAMA_MODEL_NAME = "kou-asistan"
GGUF_PREFIX = "kou-asistan"

# --v2 : çok belgeli sistemle eğitilen Trendyol-LLM-8B-T1 -> Ollama "kou-asistan-v2"
# --ham: eğitilmemiş Trendyol-LLM-8B-T1 (karşılaştırma için) -> Ollama "trendyol-8b-ham"
V2 = "--v2" in sys.argv[1:]
HAM = "--ham" in sys.argv[1:]
if V2 or HAM:
    BASE_MODEL = "Trendyol/Trendyol-LLM-8B-T1"
    ADAPTER_DIR = Path("trendyol-8b-kou-lora")
    MERGED_DIR = Path("trendyol-8b-kou-merged")
    GGUF_PREFIX = OLLAMA_MODEL_NAME = "kou-asistan-v2"
    MODELFILE = Path("Modelfile.kou-asistan-v2")
if V2 and "--v3" in sys.argv[1:]:  # v2'den devam eğitimi -> Ollama "kou-asistan-v3"
    ADAPTER_DIR = Path("trendyol-8b-kou-lora-v3")
    MERGED_DIR = Path("trendyol-8b-kou-merged-v3")
    GGUF_PREFIX = OLLAMA_MODEL_NAME = "kou-asistan-v3"
    MODELFILE = Path("Modelfile.kou-asistan-v3")
if V2 and "--v4" in sys.argv[1:]:  # tam veri + ayrıntılı cevap, sıfırdan -> Ollama "kou-asistan-v4"
    ADAPTER_DIR = Path("trendyol-8b-kou-lora-v4")
    MERGED_DIR = Path("trendyol-8b-kou-merged-v4")
    GGUF_PREFIX = OLLAMA_MODEL_NAME = "kou-asistan-v4"
    MODELFILE = Path("Modelfile.kou-asistan-v4")
if V2 and "--v5" in sys.argv[1:]:  # güncel veri + sayı/tarih odaklı örnekler, sıfırdan -> "kou-asistan-v5"
    ADAPTER_DIR = Path("trendyol-8b-kou-lora-v5")
    MERGED_DIR = Path("trendyol-8b-kou-merged-v5")
    GGUF_PREFIX = OLLAMA_MODEL_NAME = "kou-asistan-v5"
    MODELFILE = Path("Modelfile.kou-asistan-v5")
if HAM:
    from huggingface_hub import snapshot_download

    MERGED_DIR = Path(snapshot_download(BASE_MODEL, local_files_only=True))  # LoRA yok, doğrudan HF klasörü
    GGUF_PREFIX = OLLAMA_MODEL_NAME = "trendyol-8b-ham"
    MODELFILE = Path("Modelfile.trendyol-8b-ham")
BF16_GGUF = Path(f"{GGUF_PREFIX}-bf16.gguf")
FALLBACK_GGUF = Path(f"{GGUF_PREFIX}-q8_0.gguf")  # llama-quantize yoksa


def find_ollama() -> str:
    found = shutil.which("ollama")
    if found:
        return found
    default = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    if default.exists():
        return str(default)
    raise FileNotFoundError("ollama bulunamadı; Ollama kurulu mu?")


def merge_adapter() -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Ana model yükleniyor (bf16, RAM): {BASE_MODEL}")
    # 4-bit değil bf16: birleşik ağırlıklar tam hassasiyette olmalı, nicemleme GGUF aşamasında.
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map={"": "cpu"}, low_cpu_mem_usage=True
    )
    print(f"LoRA birleştiriliyor: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR)).merge_and_unload()
    model.save_pretrained(MERGED_DIR, safe_serialization=True)
    AutoTokenizer.from_pretrained(ADAPTER_DIR).save_pretrained(MERGED_DIR)
    print(f"Birleşik model kaydedildi: {MERGED_DIR.resolve()}")


def convert_to_gguf(outtype: str, outfile: Path) -> None:
    if outfile.exists():
        print(f"GGUF zaten var, atlanıyor: {outfile}")
        return
    converter = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not converter.exists():
        raise FileNotFoundError(f"{converter} yok; llama.cpp'yi klonla (bkz. dosya başı).")
    print(f"GGUF'a çevriliyor ({outtype}): {outfile}")
    subprocess.run(
        [sys.executable, str(converter), str(MERGED_DIR), "--outtype", outtype, "--outfile", str(outfile)],
        check=True,
    )


def build_gguf() -> Path:
    """Sitede kullanılacak GGUF'u üretir; llama-quantize yoksa Q8_0'a düşer."""
    if not LLAMA_QUANTIZE.exists():
        print(f"UYARI: {LLAMA_QUANTIZE} yok; {QUANT_TYPE} yerine Q8_0 üretiliyor (~8.7 GB).")
        convert_to_gguf("q8_0", FALLBACK_GGUF)
        return FALLBACK_GGUF
    quantized = Path(f"{GGUF_PREFIX}-{QUANT_TYPE.lower()}.gguf")
    if quantized.exists():
        print(f"GGUF zaten var, atlanıyor: {quantized}")
        return quantized
    convert_to_gguf("bf16", BF16_GGUF)
    print(f"Nicemleniyor ({QUANT_TYPE}): {quantized}")
    subprocess.run([str(LLAMA_QUANTIZE), str(BF16_GGUF), str(quantized), QUANT_TYPE], check=True)
    return quantized


def main() -> int:
    if not HAM and not ADAPTER_DIR.exists():
        print(f"{ADAPTER_DIR} bulunamadı. Önce: train_qlora.py")
        return 1

    if HAM:
        print(f"Eğitilmemiş model doğrudan çevriliyor: {MERGED_DIR}")
    elif not (MERGED_DIR / "config.json").exists():
        merge_adapter()
    else:
        print(f"Birleşik model zaten var, atlanıyor: {MERGED_DIR}")

    gguf_file = build_gguf()

    MODELFILE.write_text(
        f"FROM ./{gguf_file.as_posix()}\n"
        # Yönetmelik cevabı: yaratıcılık değil tutarlılık istiyoruz.
        "PARAMETER temperature 0.2\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER num_ctx 8192\n"
        'PARAMETER stop "<|im_end|>"\n'
        'PARAMETER stop "<|im_start|>"\n',
        encoding="utf-8",
    )
    print(f"Ollama modeli oluşturuluyor: {OLLAMA_MODEL_NAME}")
    result = subprocess.run([find_ollama(), "create", OLLAMA_MODEL_NAME, "-f", str(MODELFILE)], check=False)
    if result.returncode != 0:
        print("ollama create başarısız oldu.")
        return result.returncode
    print(f"\nHazır: {OLLAMA_MODEL_NAME} (Ollama'yı think=False ile çağır!)")
    if HAM:  # MERGED_DIR burada Hugging Face önbelleği: eğitim için gerekli, silinmemeli
        print(f"İstersen {BF16_GGUF} dosyasını silebilirsin (Ollama kendi kopyasını tutar).")
    else:
        print(f"İstersen {MERGED_DIR}/ ve *.gguf dosyalarını silebilirsin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
