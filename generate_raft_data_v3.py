#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 eğitim verisi (PLAN_v3.md Faz 4): v2 adaptöründen devam eğitimi için.

İçerik:
1. Yeni maddeler (fakülte staj/İME, Erasmus ...; qa_raw önbelleğinde olmayanlar): qwen3:14b soru-cevap +
   hakem (v2 üreticisinin fill_cache'i). Bağlam canlı v3 aramasıyla (rag_v3.Retriever3: hibrit + öncelikler).
2. Devam sorusu örnekleri ("takip"): aynı belgeden iki soru; ikincisi qwen3:14b ile kısa devam sorusuna
   çevrilir ("Peki süresi ne kadar?"). Model girdisi Sohbet3 ile aynı biçimde: "<kısa soru> (Önceki soru: ...)".
3. Tekrar (replay): v2 eğitim verisinin bir örneklemi (yürürlük notları temizlenmiş) — unutmayı önler.
Test seti maddeleri (veri/test_v3/test_maddeleri.json) hiçbir örnekte doğru madde olarak kullanılmaz.

Kullanım:  python -X utf8 generate_raft_data_v3.py
Çıktı:     veri/egitim_v3/raft_train_v3.jsonl, raft_eval_v3.jsonl, takip_raw.jsonl (önbellek)
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import ollama

import generate_raft_data as v1
import generate_raft_data_v2 as g2
from raft_format import REFUSAL_ANSWER, SYSTEM_PROMPT_V2, build_messages

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "veri" / "egitim_v3"
TEST_UNITS = ROOT / "veri" / "test_v3" / "test_maddeleri.json"
FOLLOW_CACHE = OUT_DIR / "takip_raw.jsonl"
SEED = 31
REPLAY_ORACLE = 1500  # v2 verisinden tekrar edilecek cevaplı örnek
REPLAY_PER_UNIT = 2
FOLLOW_UPS = 400
EVAL_SIZE = 150
ORACLE_DROP_RATIO = 0.2
GENERAL_UNITS = {"tum", "enstitu"}
FOLLOW_SCHEMA = {"type": "object", "properties": {"devam_sorusu": {"type": "string"}}, "required": ["devam_sorusu"]}
COMPLETE_SCHEMA = {"type": "object", "properties": {"cevap": {"type": "string"}}, "required": ["cevap"]}
COMPLETE_CACHE = OUT_DIR / "eksiksiz_raw.jsonl"
COMPLETE_REWRITES = 350


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as output:
        output.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def make_row(question: str, context: list[dict[str, Any]], answer: str, kind: str, unit_id: str) -> dict[str, Any]:
    return {"messages": build_messages(question, context, answer, system=SYSTEM_PROMPT_V2),
            "meta": {"type": kind, "unit_id": unit_id, "context_ids": [c["id"] for c in context],
                     "question": question}}


def live_context(retriever: Any, search: str, unit: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    """Canlı v3 aramasıyla bağlam. Birim yarı yarıya biliniyor varsayılır (sohbette netleşmiş/soruda geçmiş)."""
    birim = unit["ozellikler"]["birim"]
    known = birim if birim not in GENERAL_UNITS and rng.random() < 0.5 else None
    ranked = retriever.rank_units(retriever.candidates(search), search, known, None)
    return retriever.context(ranked)


def with_oracle(context: list[dict[str, Any]], unit: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    others = [a for a in context if a["id"] != unit["id"]][:3]
    mixed = others + [unit]
    rng.shuffle(mixed)
    return mixed


def follow_up_question(first: str, second: str) -> str | None:
    prompt = (
        "Bir kullanıcı üniversite mevzuat asistanına art arda iki soru soruyor.\n"
        f"1. soru: {first}\n2. soru (tam hali): {second}\n\n"
        "2. soruyu, 1. sorudan hemen sonra yazılmış KISA bir devam sorusuna çevir: konuyu tekrar etme, "
        "'Peki', 'Ya', 'Bunun', 'O zaman' gibi bağlantılarla başlasın, en fazla 10 kelime olsun, anlamı "
        "değişmesin. JSON ver."
    )
    try:
        response = ollama.chat(model=g2.GEN_MODEL, messages=[{"role": "user", "content": prompt}],
                               format=FOLLOW_SCHEMA, think=False, options={"temperature": 0.5, "num_ctx": 4096})
        text = json.loads(response["message"]["content"])["devam_sorusu"].strip()
    except Exception:
        return None
    if not text or len(text.split()) > 12 or text.lower() == second.lower():
        return None
    return text


def complete_answer(unit: dict[str, Any], question: str, answer: str) -> str | None:
    """Kısa cevabı maddeye göre tamamlar; hakem (metne sadık + soruyu cevaplıyor) onaylamazsa None."""
    prompt = (
        f"MADDE METNİ:\n{unit['metin']}\n\nSORU: {question}\nMEVCUT CEVAP: {answer}\n\n"
        "Mevcut cevabı madde metnine göre EKSİKSİZ hale getir: soruyla ilgili koşulları, süreleri, sayıları, "
        "istisnaları ve kararı veren kişi/kurulu ekle. Soruyla ilgisiz bilgi ve metinde olmayan hiçbir bilgi "
        "ekleme. 1-4 doğal, resmi cümle; madde/fıkra numarası veya kaynak yazma. Mevcut cevap zaten eksiksizse "
        "aynen yaz. JSON ver."
    )
    try:
        response = ollama.chat(model=g2.GEN_MODEL, messages=[{"role": "user", "content": prompt}],
                               format=COMPLETE_SCHEMA, think=False, options={"temperature": 0.3, "num_ctx": 8192})
        new = json.loads(response["message"]["content"])["cevap"].strip()
    except Exception:
        return None
    if len(new) < len(answer) or len(new) > 900 or re.search(r"\b(madde|fıkra)\s+\d", new, re.IGNORECASE):
        return None
    candidate = [{"soru": question, "cevap": new}]
    try:
        v1.judge_examples({"text": unit["metin"]}, candidate)
    except Exception:
        return None
    verdict = candidate[0].get("kontrol", {})
    return new if verdict.get("metne_sadik") and verdict.get("soruyu_cevapliyor") else None


def main() -> int:
    from rag_v3 import Retriever3

    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    articles = read_jsonl(g2.ARTICLES)
    by_id = {a["id"]: a for a in articles}
    test_units = set(json.loads(TEST_UNITS.read_text(encoding="utf-8")))
    print(f"Test maddesi (eğitimden hariç): {len(test_units)}", flush=True)

    # ---- 1. Yeni maddeler: v2 eğitim/test verisinde hiç kullanılmamış olanlar (soru-cevap önbelleğe eklenir;
    # "önbellekte yok" ölçütü yeniden çalıştırmada yeni maddeleri kaybettiriyordu).
    v2_units = {r["meta"].get("unit_id") for r in read_jsonl(g2.TRAIN) + read_jsonl(g2.TEST)}
    new_units = [a for a in articles if a["id"] not in v2_units and a["id"] not in test_units
                 and a["ozellikler"]["konu"] != "genel_hukumler" and len(a["metin"]) >= g2.MIN_CHARS
                 and a["ozellikler"]["belge_id"] not in g2.ALWAYS_CAPPED]
    print(f"Yeni madde: {len(new_units)}", flush=True)
    cache = g2.fill_cache(new_units)

    retriever = Retriever3()
    refusal_cache = json.loads(g2.REFUSAL_CACHE.read_text(encoding="utf-8")) if g2.REFUSAL_CACHE.exists() else {}
    new_rows: list[dict[str, Any]] = []
    unsafe = 0
    start = time.time()
    for index, unit in enumerate(new_units, start=1):
        if unit["id"] not in cache:
            continue
        kept, _ = v1.validate_examples({"text": unit["metin"]}, cache[unit["id"]])
        for item in kept:
            context = live_context(retriever, item["soru"], unit, rng)
            if rng.random() < ORACLE_DROP_RATIO:
                others = [a for a in context if a["id"] != unit["id"]][:4]
                if g2.context_answers(item["soru"], others, refusal_cache):
                    unsafe += 1
                    continue
                new_rows.append(make_row(item["soru"], others, REFUSAL_ANSWER, "no_oracle", unit["id"]))
            else:
                answer = g2.answer_with_citation(unit, item["cevap"], item["fikralar"])
                new_rows.append(make_row(item["soru"], with_oracle(context, unit, rng), answer, "oracle_v3", unit["id"]))
        if index % 25 == 0:
            print(f"  bağlam: {index}/{len(new_units)} ({(time.time() - start) / 60:.0f} dk)", flush=True)
    g2.REFUSAL_CACHE.write_text(json.dumps(refusal_cache, ensure_ascii=False), encoding="utf-8")
    print(f"Yeni madde örneği: {len(new_rows)} (bağlamda cevabı olduğu için atılan red: {unsafe})", flush=True)

    # ---- 2. Devam sorusu örnekleri
    full_cache = {row["id"]: row["examples"] for row in read_jsonl(g2.QA_CACHE)}
    by_doc: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for unit_id, examples in full_cache.items():
        unit = by_id.get(unit_id)
        if unit is None or unit_id in test_units:
            continue
        kept, _ = v1.validate_examples({"text": unit["metin"]}, examples)
        for item in kept[:2]:
            by_doc.setdefault(unit["ozellikler"]["belge_id"], []).append((unit, item))
    follow_cache = {row["key"]: row["devam_sorusu"] for row in read_jsonl(FOLLOW_CACHE)} if FOLLOW_CACHE.exists() else {}
    docs = [d for d, items in by_doc.items() if len({u["id"] for u, _ in items}) >= 2]
    follow_rows: list[dict[str, Any]] = []
    attempts = 0
    with FOLLOW_CACHE.open("a", encoding="utf-8") as cache_file:
        while len(follow_rows) < FOLLOW_UPS and attempts < FOLLOW_UPS * 3:
            attempts += 1
            doc = rng.choice(docs)
            (unit_a, item_a), (unit_b, item_b) = rng.sample(by_doc[doc], 2)
            if unit_a["id"] == unit_b["id"]:
                continue
            key = f"{unit_a['id']}|{item_a['soru']}|{unit_b['id']}|{item_b['soru']}"
            if key not in follow_cache:
                follow_cache[key] = follow_up_question(item_a["soru"], item_b["soru"]) or ""
                cache_file.write(json.dumps({"key": key, "devam_sorusu": follow_cache[key]}, ensure_ascii=False) + "\n")
                cache_file.flush()
            short = follow_cache[key]
            if not short:
                continue
            question = f"{short} (Önceki soru: {item_a['soru']})"
            context = live_context(retriever, f"{item_a['soru']} {short}", unit_b, rng)
            answer = g2.answer_with_citation(unit_b, item_b["cevap"], item_b["fikralar"])
            follow_rows.append(make_row(question, with_oracle(context, unit_b, rng), answer, "takip", unit_b["id"]))
            if len(follow_rows) % 50 == 0:
                print(f"  devam sorusu: {len(follow_rows)}/{FOLLOW_UPS}", flush=True)
    print(f"Devam sorusu örneği: {len(follow_rows)}", flush=True)

    # ---- 3. Tekrar örneklemi (v2 verisi; test maddeleri hariç)
    old = [r for r in read_jsonl(g2.TRAIN) + read_jsonl(g2.TEST) if r["meta"].get("unit_id") not in test_units]
    replay = [r for r in old if r["meta"]["type"] != "oracle"]
    oracle = [r for r in old if r["meta"]["type"] == "oracle"]
    rng.shuffle(oracle)
    per_unit: dict[str, int] = {}
    for row in oracle:
        unit_id = row["meta"]["unit_id"]
        if per_unit.get(unit_id, 0) < REPLAY_PER_UNIT and sum(per_unit.values()) < REPLAY_ORACLE:
            per_unit[unit_id] = per_unit.get(unit_id, 0) + 1
            replay.append(row)
    print(f"Tekrar örneği: {len(replay)}", flush=True)

    # ---- 3b. Eksiksiz cevap: tekrar örneklerinin bir kısmında kısa v2 cevabı maddeye göre tamamlanır
    # (v2 testinde hakemin en sık itirazı "cevap eksik"ti). Hakemden geçmeyen yeniden yazım kullanılmaz.
    complete_cache = {row["key"]: row["cevap"] for row in read_jsonl(COMPLETE_CACHE)} if COMPLETE_CACHE.exists() else {}
    targets = [r for r in replay if r["meta"]["type"] == "oracle"][:COMPLETE_REWRITES]
    improved = 0
    with COMPLETE_CACHE.open("a", encoding="utf-8") as cache_file:
        for index, row in enumerate(targets, start=1):
            unit = by_id.get(row["meta"]["unit_id"])
            if unit is None:
                continue
            content = row["messages"][-1]["content"]
            body, sep, citation = content.partition("\n\nKaynak:")
            key = f"{unit['id']}|{row['meta']['question']}"
            if key not in complete_cache:
                complete_cache[key] = complete_answer(unit, row["meta"]["question"], body.strip()) or ""
                cache_file.write(json.dumps({"key": key, "cevap": complete_cache[key]}, ensure_ascii=False) + "\n")
                cache_file.flush()
            if complete_cache[key] and sep:
                row["messages"][-1]["content"] = f"{complete_cache[key]}{sep}{citation}"
                improved += 1
            if index % 100 == 0:
                print(f"  eksiksiz cevap: {index}/{len(targets)} ({improved} iyileşti)", flush=True)
    print(f"Eksiksiz hale getirilen cevap: {improved}/{len(targets)}", flush=True)

    rows = new_rows + follow_rows + replay
    rng.shuffle(rows)
    eval_rows, train_rows = rows[:EVAL_SIZE], rows[EVAL_SIZE:]
    write_jsonl(train_rows, OUT_DIR / "raft_train_v3.jsonl")
    write_jsonl(eval_rows, OUT_DIR / "raft_eval_v3.jsonl")
    counts: dict[str, int] = {}
    for row in train_rows:
        counts[row["meta"]["type"]] = counts.get(row["meta"]["type"], 0) + 1
    print(f"Eğitim: {len(train_rows)} {counts} | değerlendirme: {len(eval_rows)} -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
