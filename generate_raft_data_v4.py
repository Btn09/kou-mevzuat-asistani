#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v4 eğitim verisi: Trendyol-LLM-8B-T1'den SIFIRDAN eğitim (v3'teki devam eğitimi %78 -> %72 düşürmüştü).

İçerik:
1. Tekrar: v2 verisi (cevaplı örneklerden madde başına en fazla 2, tüm red/kapsam dışı örnekleri) — v2'nin
   kaynak gösterme ve reddetme davranışı korunur.
2. Yeni maddeler: tam veriden (mevzuat.gov.tr, KOÜ web sayfaları/dosyaları, harç tabloları, AKTS) konu
   öncelikli ~650 madde. qwen3:14b AYRINTILI soru-cevap üretir (kullanıcılar yüzeysel cevaptan şikâyetçi),
   aynı model hakem olarak metne sadakati denetler; bağlam canlı v3 aramasıyla (rag_v3.Retriever3).
3. Devam sorusu örnekleri: v3 verisindeki "takip" örnekleri.
Test maddeleri (eski test + web testi) hiçbir örnekte kullanılmaz.

Kullanım:  python -X utf8 generate_raft_data_v4.py
Çıktı:     veri/egitim_v4/raft_train_v4.jsonl, raft_eval_v4.jsonl, qa_raw_v4.jsonl (önbellek, kaldığı yerden)
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
import generate_raft_data_v3 as g3
from pdf_to_chromadb import normalize_for_matching
from raft_format import REFUSAL_ANSWER, SYSTEM_PROMPT_V3, build_messages

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "veri" / "egitim_v4"
QA_CACHE = OUT_DIR / "qa_raw_v4.jsonl"
SEED = 41
REPLAY_ORACLE = 2200
REPLAY_PER_UNIT = 2
NEW_PER_UNIT = 4
ORACLE_DROP_RATIO = 0.15
EVAL_SIZE = 200
# (belge kimliği öneki, kaç madde, belge başına en fazla)
QUOTAS = [("mevzuat_", 190, 30), ("kou_", 110, 15), ("webt_", 120, 3), ("webf_", 150, 3), ("webd_", 40, 2),
          ("akts_", 20, 1)]
# Her zaman alınan belgeler (madde başına sınır yok): 2026-2027 harç tablosu ve genel bilgiler
ALWAYS_DOCS = ("webf_20260915_7aa94", "webf_20260903_f1b0d")
ALWAYS_MAX = 45
PRIORITY_RE = re.compile(
    r"harc|katki pay|ogrenim ucret|erasmus|farabi|mevlana|yabanci uyruk|uluslararasi ogrenci|burs|kredi|"
    r"yurt hizmet|maas|aylik|izin|gorevlendirme|ek ders|doner sermaye|tesvik|docent|yukseltme|atama|intorn|"
    r"tip fakultesi|staj|yatay gecis|cift anadal|mazeret|butunleme|mezuniyet|kayit dondur|disiplin|emekli|"
    r"harcirah|yolluk|lisansustu|tez|sinav")


def make_row(question: str, context: list[dict[str, Any]], answer: str, kind: str, unit_id: str) -> dict[str, Any]:
    """Yeni örnekler ayrıntılı-cevap talimatıyla (SYSTEM_PROMPT_V3, sitede kullanılan); v2 tekrar örnekleri kendi
    (kısa) talimatlarıyla kalır: model ayrıntılı üslubu V3 talimatına bağlar."""
    return {"messages": build_messages(question, context, answer, system=SYSTEM_PROMPT_V3),
            "meta": {"type": kind, "unit_id": unit_id, "context_ids": [c["id"] for c in context],
                     "question": question}}


def generate_detailed(article: dict[str, Any], style: str, count: int) -> list[dict[str, Any]]:
    """v2 üreticisinin istemi; cevaplar ayrıntılı (2-5 cümle, gerekirse liste, en fazla ~700 karakter)."""
    fields = article["ozellikler"]
    audience = "bir öğretim elemanının veya idari personelin" if fields["hedef_kitle"] == "personel" else \
        "bir öğrencinin ya da öğretim elemanının"
    prompt = (
        f"Aşağıda Kocaeli Üniversitesi'nde uygulanan '{fields['belge_adi']}' belgesinden bir bölüm var.\n\n"
        f"{article['metin']}\n\n"
        f"Bu metne dayanarak {audience} sorabileceği {count} FARKLI soru ve cevabını üret.\n"
        f"Soru üslubu: {style}\n"
        "Kurallar:\n"
        "- SADECE cevabı metinde açıkça yazan sorular sor; metinde cevabı olmayan durumları sorma.\n"
        "- Soruyu soran metni görmüyor: soruda 'bu madde', 'metne göre', 'fıkra', 'tablo' gibi ifadeler kullanma.\n"
        "- Sorular metnin farklı noktalarını kapsasın, birbirinin tekrarı olmasın.\n"
        "- Cevap YALNIZCA metne dayansın; metinde olmayan bilgi, sayı veya tarih ekleme, yorum yapma.\n"
        "- Metin bir konuyu Senato'ya, bir kurula veya başka bir mevzuata bırakıyorsa cevapta bunu söyle.\n"
        "- Bilgi belirli bir gruba aitse (ör. uluslararası/yabancı uyruklu öğrenciler, ikinci öğretim, normal "
        "süreyi aşanlar, belirli bir fakülte, belirli bir kayıt yılı ya da eğitim-öğretim yılı) bu grubu hem "
        "soruda hem cevapta AÇIKÇA yaz; genel kural gibi sunma. Parantez içindeki tablo adı satırın hangi gruba "
        "ait olduğunu gösterir.\n"
        "- Cevap soruyu DOĞRUDAN yanıtlasın; kaçamak cevap verme.\n"
        "- Cevap AYRINTILI ve EKSİKSİZ olsun: soruyla ilgili tüm koşulları, süreleri, tarihleri, tutarları, "
        "istisnaları ve kararı veren kişi/kurulu aktar. Birden çok durum varsa her birini ayrı satırda '- ' ile "
        "listele. 2-5 cümle, en fazla 700 karakter; soruyla ilgisiz bilgi ekleme.\n"
        "- Cevaplar düzgün Türkçe, nazik ve resmi olsun; 'Siz,' diye başlama.\n"
        "- Cevaba madde/fıkra numarası, köşeli parantez veya kaynak YAZMA. Dayandığın fıkra numaralarını "
        "yalnızca 'fikralar' alanına yaz (fıkra numarası yoksa boş bırak).\n"
        "JSON olarak cevap ver."
    )
    response = ollama.chat(model=g2.GEN_MODEL, messages=[{"role": "user", "content": prompt}], format=v1.QA_SCHEMA,
                           options={"temperature": 0.6, "num_ctx": 8192, "num_predict": 3500}, think=False)
    return json.loads(response["message"]["content"]).get("ornekler", [])


def fill_cache(units: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    cache = {row["id"]: row["examples"] for row in g3.read_jsonl(QA_CACHE)} if QA_CACHE.exists() else {}
    todo = [u for u in units if u["id"] not in cache]
    print(f"Soru üretimi: {len(todo)} madde ({len(units) - len(todo)} önbellekte)", flush=True)
    start = time.time()
    with QA_CACHE.open("a", encoding="utf-8") as cache_file:
        for index, unit in enumerate(todo, start=1):
            styles = g2.STAFF_STYLES if unit["ozellikler"]["hedef_kitle"] == "personel" else g2.STUDENT_STYLES
            count = max(1, min(3, round(len(unit["metin"]) / 400)))
            try:
                examples = [ex for style in styles for ex in generate_detailed(unit, style, count)]
                v1.judge_examples({"text": unit["metin"]}, examples)
            except Exception as exc:
                print(f"  [{index}/{len(todo)}] {unit['id'][:60]}: HATA ({str(exc)[:80]})", flush=True)
                continue
            cache[unit["id"]] = examples
            cache_file.write(json.dumps({"id": unit["id"], "examples": examples}, ensure_ascii=False) + "\n")
            cache_file.flush()
            if index % 10 == 0:
                elapsed = time.time() - start
                print(f"  [{index}/{len(todo)}] {elapsed / 60:.0f} dk, kalan ~{elapsed / index * (len(todo) - index) / 60:.0f} dk",
                      flush=True)
    return cache


def select_units(articles: list[dict[str, Any]], excluded: set[str], rng: random.Random) -> list[dict[str, Any]]:
    from rag_v3 import REPORT_RE

    chosen: list[dict[str, Any]] = []
    always = [a for a in articles if a["ozellikler"]["belge_id"].startswith(ALWAYS_DOCS) and a["id"] not in excluded]
    chosen += always[:ALWAYS_MAX]
    taken = {a["id"] for a in chosen}
    for prefix, total, per_doc in QUOTAS:
        pool = []
        for a in articles:
            fields = a["ozellikler"]
            if (not fields["belge_id"].startswith(prefix) or a["id"] in excluded or a["id"] in taken
                    or len(a["metin"]) < g2.MIN_CHARS or fields.get("konu") == "genel_hukumler"):
                continue
            text = normalize_for_matching(fields["belge_adi"] + " " + a["metin"][:600])
            if prefix.startswith("web") and REPORT_RE.search(normalize_for_matching(fields["belge_adi"] + " " + fields["belge_id"])):
                continue
            weight = 4.0 if PRIORITY_RE.search(text) else 1.0
            pool.append((rng.random() ** (1 / weight), a))  # ağırlıklı rastgele sıra
        pool.sort(key=lambda item: -item[0])
        per: dict[str, int] = {}
        count = 0
        for _, a in pool:
            doc = a["ozellikler"]["belge_id"]
            if per.get(doc, 0) >= per_doc:
                continue
            per[doc] = per.get(doc, 0) + 1
            chosen.append(a)
            taken.add(a["id"])
            count += 1
            if count >= total:
                break
        print(f"  {prefix}: {count} madde ({len(pool)} aday)", flush=True)
    return chosen


def main() -> int:
    from rag_v3 import Retriever3

    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    articles = g3.read_jsonl(g2.ARTICLES)
    excluded = set(json.loads(g3.TEST_UNITS.read_text(encoding="utf-8")))
    for row in g3.read_jsonl(ROOT / "veri" / "test_v3" / "test_seti_web.jsonl"):
        excluded |= {t["beklenen"]["unit_id"] for t in row["turlar"] if t["beklenen"].get("unit_id")}
    v2_rows = g3.read_jsonl(g2.TRAIN) + g3.read_jsonl(g2.TEST)
    excluded_new = excluded | {r["meta"].get("unit_id") for r in v2_rows}
    print(f"Test maddesi (hariç): {len(excluded)}", flush=True)

    # ---- 1. Tekrar (v2)
    replay = [r for r in v2_rows if r["meta"]["type"] != "oracle" and r["meta"].get("unit_id") not in excluded]
    oracle = [r for r in v2_rows if r["meta"]["type"] == "oracle" and r["meta"].get("unit_id") not in excluded]
    rng.shuffle(oracle)
    per_unit: dict[str, int] = {}
    kept_oracle = 0
    for row in oracle:
        unit_id = row["meta"]["unit_id"]
        if per_unit.get(unit_id, 0) < REPLAY_PER_UNIT and kept_oracle < REPLAY_ORACLE:
            per_unit[unit_id] = per_unit.get(unit_id, 0) + 1
            replay.append(row)
            kept_oracle += 1
    print(f"Tekrar örneği (v2): {len(replay)}", flush=True)

    # ---- 2. Yeni maddeler
    units = select_units(articles, excluded_new, rng)
    print(f"Yeni madde: {len(units)}", flush=True)
    cache = fill_cache(units)
    retriever = Retriever3()
    refusal_cache = json.loads(g2.REFUSAL_CACHE.read_text(encoding="utf-8")) if g2.REFUSAL_CACHE.exists() else {}
    new_rows: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    for index, unit in enumerate(units, start=1):
        if unit["id"] not in cache:
            continue
        kept, reasons = v1.validate_examples({"text": unit["metin"]}, cache[unit["id"]])
        for reason, count in reasons.items():
            rejected[reason] = rejected.get(reason, 0) + count
        for item in kept[:NEW_PER_UNIT]:
            context = g3.live_context(retriever, item["soru"], unit, rng)
            if rng.random() < ORACLE_DROP_RATIO:
                others = [a for a in context if a["id"] != unit["id"]][:4]
                if g2.context_answers(item["soru"], others, refusal_cache):
                    continue
                new_rows.append(make_row(item["soru"], others, REFUSAL_ANSWER, "no_oracle", unit["id"]))
            else:
                answer = g2.answer_with_citation(unit, item["cevap"], item["fikralar"])
                new_rows.append(make_row(item["soru"], g3.with_oracle(context, unit, rng), answer, "oracle_v4",
                                            unit["id"]))
        if index % 100 == 0:
            print(f"  bağlam: {index}/{len(units)}", flush=True)
    g2.REFUSAL_CACHE.write_text(json.dumps(refusal_cache, ensure_ascii=False), encoding="utf-8")
    print(f"Yeni örnek: {len(new_rows)} | elenen: {rejected}", flush=True)

    # ---- 3. Devam sorusu (v3 verisinden)
    v3_dir = ROOT / "veri" / "egitim_v3"
    follow = [r for name in ("raft_train_v3.jsonl", "raft_eval_v3.jsonl") for r in g3.read_jsonl(v3_dir / name)
              if r["meta"]["type"] == "takip" and r["meta"].get("unit_id") not in excluded]
    print(f"Devam sorusu örneği: {len(follow)}", flush=True)

    rows = replay + new_rows + follow
    rng.shuffle(rows)
    eval_rows, train_rows = rows[:EVAL_SIZE], rows[EVAL_SIZE:]
    g3.write_jsonl(train_rows, OUT_DIR / "raft_train_v4.jsonl")
    g3.write_jsonl(eval_rows, OUT_DIR / "raft_eval_v4.jsonl")
    counts: dict[str, int] = {}
    for row in train_rows:
        counts[row["meta"]["type"]] = counts.get(row["meta"]["type"], 0) + 1
    print(f"Eğitim: {len(train_rows)} {counts} | değerlendirme: {len(eval_rows)} -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
