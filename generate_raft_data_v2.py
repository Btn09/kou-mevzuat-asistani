#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Çok belgeli sistem için RAFT eğitim verisi üretir (öğretmen model: qwen3:14b).

Her örnek = soru + canlı sistemin getireceği bağlam (rag_v2: kapsam filtresi + reranker) + hedef cevap:
  - oracle      : doğru madde bağlamda; cevap + "Kaynak: <belge>, Madde X, fıkra Y"
  - no_oracle   : doğru madde çıkarılmış (bağlam soruyu gerçekten cevaplamıyorsa); cevap = yönlendirme
  - out_of_scope: mevzuat dışı soru; cevap = yönlendirme
Yürürlükten kalkmış belgelere (KDV 117) dayanan cevaplara bunu belirten not kodla eklenir.

Madde seçimi: öğrenci belgelerinin tamamı, personel belgelerinden belge başına üst sınırlı örneklem
(--tum ile hepsi). Genel hükümler (amaç, kapsam, yürürlük...) ve çok kısa maddeler atlanır.

Kullanım:
    python -X utf8 generate_raft_data_v2.py --limit 3   # deneme
    python -X utf8 generate_raft_data_v2.py             # seçili ~820 madde (kaldığı yerden devam eder)
    python -X utf8 generate_raft_data_v2.py --tum       # 1326 maddenin hepsi

Çıktılar (veri/egitim/): qa_raw.jsonl (önbellek), raft_train.jsonl, raft_test.jsonl, red_kontrol.json
"""

from __future__ import annotations

import datetime as dt
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import ollama

import generate_raft_data as v1  # doğrulama filtreleri, hakem, kapsam dışı sorular, alıntı kontrolü
from rag_v2 import Retriever
from raft_format import REFUSAL_ANSWER, SYSTEM_PROMPT_V2, build_messages, format_citation

VERI = Path(__file__).resolve().parent.parent / "veri"
ARTICLES = VERI / "islenmis" / "maddeler.jsonl"
OUT_DIR = VERI / "egitim"
QA_CACHE = OUT_DIR / "qa_raw.jsonl"
TRAIN = OUT_DIR / "raft_train.jsonl"
TEST = OUT_DIR / "raft_test.jsonl"
REFUSAL_CACHE = OUT_DIR / "red_kontrol.json"

GEN_MODEL = "qwen3:14b"
SEED = 42
CONTEXT_SIZE = 4
ORACLE_DROP_RATIO = 0.2
TEST_RATIO = 0.12  # maddelerin bu kadarı sadece teste (eğitimde görülmez)
MIN_CHARS = 150
# Personel belgelerinde belge başına en fazla bu kadar madde (büyük kanunlar veriyi domine etmesin).
STAFF_CAP = {"kanun_2547_yuksekogretim": 40, "kanun_4734_kamu_ihale": 20, "kanun_6245_harcirah": 20,
             "kdv_genel_tebligi_117_tevkifat": 15, "kou_bap_uygulama_esaslari": 30,
             "kamu_ihale_3f_yuksekogretim_alimlari": 20, "kou_kalite_yonergesi": 15}
STAFF_CAP_DEFAULT = 35
# --tum modunda da sınırlı kalanlar: yürürlükten kalkmış KDV tebliği modeli eskimiş metne alıştırmasın;
# "yürürlükten kalktı" notunu öğrenmesine 15 bölüm yeter (kullanıcıyla kararlaştırıldı).
ALWAYS_CAPPED = {"kdv_genel_tebligi_117_tevkifat"}

STUDENT_STYLES = [
    "Kısa, doğrudan ve resmi dilde sorular.",
    "Öğrencinin kendi durumunu bir cümleyle anlattığı, günlük konuşma dilinde sorular. Bazı SORULAR küçük "
    "harfle, noktalama olmadan veya ufak yazım hatalarıyla yazılmış olsun (bu serbestlik sadece sorular için).",
]
STAFF_STYLES = [
    "Kısa, doğrudan ve resmi dilde sorular.",
    "Bir öğretim elemanının ya da idari personelin kendi durumunu anlattığı, günlük dilde sorular "
    "(ör. bir proje yürütücüsü, ders veren öğretim üyesi, görevlendirme isteyen araştırmacı).",
]
STAFF_OUT_OF_SCOPE = [
    "Maaşım ayın kaçında yatıyor?",
    "Personel otoparkına araç kaydı nasıl yapılır?",
    "Akademik personelin yıllık izin formu nereden indirilir?",
    "Kampüste personel yemekhanesinin menüsü nedir?",
    "Rektörlük binasında toplantı salonu nasıl rezerve edilir?",
    "Üniversitenin e-imza başvurusu hangi birimden yapılır?",
    "Kurum e-posta kotam dolduysa ne yapmalıyım?",
    "Bölüm sekreterinin telefon numarası nedir?",
    "Akademik teşvik başvuru sonuçları ne zaman açıklanacak?",
    "Kreş için personel çocuklarına indirim var mı?",
]
DEPRECATED_NOTE = ("Not: Bu bilgi 1/5/2014'te yürürlükten kalkan {belge}'ne dayanmaktadır; güncel hükümler için "
                   "KDV Genel Uygulama Tebliği'ne bakınız.")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_units(articles: list[dict[str, Any]], take_all: bool, rng: random.Random,
                 cached: set[str] = frozenset()) -> list[dict[str, Any]]:
    candidates = [a for a in articles
                  if a["ozellikler"]["konu"] != "genel_hukumler" and len(a["metin"]) >= MIN_CHARS]
    if take_all:
        selected = [a for a in candidates if a["ozellikler"]["belge_id"] not in ALWAYS_CAPPED]
        for doc in ALWAYS_CAPPED:
            units = [a for a in candidates if a["ozellikler"]["belge_id"] == doc]
            # Önceki turda üretilmiş olanlar tercih edilir (önbellek boşa gitmesin), eksik kalırsa tamamlanır.
            units.sort(key=lambda a: a["id"] not in cached)
            selected += units[:STAFF_CAP.get(doc, STAFF_CAP_DEFAULT)]
        return selected
    selected, by_doc = [], {}
    for article in candidates:
        if article["ozellikler"]["hedef_kitle"] == "personel":
            by_doc.setdefault(article["ozellikler"]["belge_id"], []).append(article)
        else:
            selected.append(article)
    for doc, units in by_doc.items():
        cap = STAFF_CAP.get(doc, STAFF_CAP_DEFAULT)
        selected += units if len(units) <= cap else rng.sample(units, cap)
    return selected


def question_count(article: dict[str, Any]) -> int:
    return max(2, min(8, round(len(article["metin"]) / 250)))


def generate_qa(article: dict[str, Any], style: str, count: int) -> list[dict[str, Any]]:
    fields = article["ozellikler"]
    audience = "bir öğretim elemanının veya idari personelin" if fields["hedef_kitle"] == "personel" else \
        "bir öğrencinin"
    prompt = (
        f"Aşağıda Kocaeli Üniversitesi'nde uygulanan '{fields['belge_adi']}' belgesinden bir bölüm var.\n\n"
        f"{article['metin']}\n\n"
        f"Bu metne dayanarak {audience} sorabileceği {count} FARKLI soru ve cevabını üret.\n"
        f"Soru üslubu: {style}\n"
        "Kurallar:\n"
        "- SADECE cevabı metinde açıkça yazan sorular sor; metinde cevabı olmayan durumları sorma.\n"
        "- Soruyu soran metni görmüyor: soruda 'bu madde', 'metne göre', 'fıkra' gibi ifadeler kullanma.\n"
        "- Sorular metnin farklı noktalarını kapsasın, birbirinin tekrarı olmasın.\n"
        "- Cevap YALNIZCA metne dayansın; metinde olmayan bilgi, sayı veya tarih ekleme, yorum yapma.\n"
        "- Metin bir konuyu Senato'ya, bir kurula veya başka bir mevzuata bırakıyorsa cevapta bunu söyle.\n"
        "- Cevap soruyu DOĞRUDAN yanıtlasın; kaçamak cevap verme.\n"
        "- Cevap EKSİKSİZ olsun: soruyla ilgili koşulları, süreleri, sayıları, istisnaları ve kararı veren "
        "kişi/kurulu atlama (v2 değerlendirmesinde en sık hata eksik cevaptı).\n"
        "- Cevap 1-4 doğal cümle olsun, nazik ve resmi; 'Siz,' diye başlama.\n"
        "- Cevaplar HER ZAMAN düzgün Türkçe yazılsın: Türkçe karakterlerle, büyük harfle başlayan cümlelerle.\n"
        "- Cevaba madde/fıkra numarası, köşeli parantez veya kaynak YAZMA. Dayandığın fıkra numaralarını "
        "yalnızca 'fikralar' alanına yaz (fıkra numarası yoksa boş bırak).\n"
        "JSON olarak cevap ver."
    )
    response = ollama.chat(model=GEN_MODEL, messages=[{"role": "user", "content": prompt}], format=v1.QA_SCHEMA,
                           options={"temperature": 0.7, "num_ctx": 8192, "num_predict": 3000}, think=False)
    return json.loads(response["message"]["content"]).get("ornekler", [])


def fill_cache(units: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    cache = {row["id"]: row["examples"] for row in read_jsonl(QA_CACHE)} if QA_CACHE.exists() else {}
    todo = [u for u in units if u["id"] not in cache]
    print(f"Soru üretimi: {len(todo)} birim ({len(units) - len(todo)} önbellekte), model {GEN_MODEL}", flush=True)
    start = time.time()
    with QA_CACHE.open("a", encoding="utf-8") as cache_file:
        for index, unit in enumerate(todo, start=1):
            styles = STAFF_STYLES if unit["ozellikler"]["hedef_kitle"] == "personel" else STUDENT_STYLES
            legacy = {"text": unit["metin"]}  # v1 hakem/doğrulama fonksiyonları 'text' bekliyor
            try:
                examples = [ex for style in styles for ex in generate_qa(unit, style, question_count(unit))]
                v1.judge_examples(legacy, examples)
            except Exception as exc:
                print(f"  [{index}/{len(todo)}] {unit['id']}: HATA ({exc}); sonra tekrar denenecek", flush=True)
                continue
            cache[unit["id"]] = examples
            cache_file.write(json.dumps({"id": unit["id"], "examples": examples}, ensure_ascii=False) + "\n")
            cache_file.flush()
            elapsed = time.time() - start
            print(f"  [{index}/{len(todo)}] {unit['ozellikler']['belge_adi'][:40]} {unit['ozellikler']['madde_etiketi']}: "
                  f"{len(examples)} örnek | {elapsed / 60:.0f} dk, kalan ~{elapsed / index * (len(todo) - index) / 60:.0f} dk",
                  flush=True)
    return cache


def context_answers(question: str, context: list[dict[str, Any]], cache: dict[str, bool]) -> bool:
    """v1 kontrolü + hata dayanıklılığı: 14B'nin alıntısı yarıda kesilip JSON bozulursa (ilk tam turda
    bu yüzden çöktü) bağlam 'cevaplıyor' sayılır, yani o reddetme örneği güvenli tarafta kalınıp atılır."""
    try:
        return v1.context_answers(question, [{"id": c["id"], "text": c["metin"]} for c in context], cache)
    except Exception:
        return True


def answer_with_citation(unit: dict[str, Any], answer: str, fikralar: list[int]) -> str:
    fields = unit["ozellikler"]
    citation = format_citation(fields["madde_etiketi"], fikralar, fields["belge_adi"], fields["madde_basligi"])
    # Yürürlükten kalkmış belge: model bunu her seferinde söylemeli. Bitiş tarihi geçmiş olmalı; ilk sürümde
    # yalnızca alanın dolu olmasına bakılıyordu ve 2027'de biten Yatay Geçiş Kılavuzu'na da KDV notu ekleniyordu.
    if fields.get("yururluk_bitis") and dt.date.fromisoformat(fields["yururluk_bitis"]) <= dt.date.today():
        answer = f"{answer} {DEPRECATED_NOTE.format(belge=fields['belge_adi'])}"
    return f"{answer}\n\n{citation}"


def main() -> int:
    args = sys.argv[1:]
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    articles = read_jsonl(ARTICLES)
    by_id = {a["id"]: a for a in articles}
    cached_ids = {row["id"] for row in read_jsonl(QA_CACHE)} if QA_CACHE.exists() else set()
    units = select_units(articles, "--tum" in args, rng, cached_ids)
    rng.shuffle(units)
    if limit:
        units = units[:limit]
    print(f"Seçilen birim: {len(units)}")
    cache = fill_cache(units)

    # ---- Doğrulama (v1 filtreleri: uzunluk, metne gönderme, talimat sızıntısı, bozuk Türkçe, hakem, uydurma sayı)
    items, rejected = [], {}
    for unit in units:
        if unit["id"] not in cache:
            continue
        kept, reasons = v1.validate_examples({"text": unit["metin"]}, cache[unit["id"]])
        for reason, count in reasons.items():
            rejected[reason] = rejected.get(reason, 0) + count
        items += [{**example, "unit_id": unit["id"]} for example in kept]
    print(f"\nGeçerli örnek: {len(items)}, elenen: {rejected}")
    if not items:
        return 1

    # ---- Bağlam: canlı sistemle aynı arama (personel maddelerinde arayüzde "akademisyenim" seçili varsayılır)
    print("Bağlamlar hazırlanıyor (rag_v2: kapsam filtresi + reranker)...", flush=True)
    retriever = Retriever()
    refusal_cache = json.loads(REFUSAL_CACHE.read_text(encoding="utf-8")) if REFUSAL_CACHE.exists() else {}
    madde_ids = sorted({by_id[i["unit_id"]]["ozellikler"]["madde_id"] for i in items})
    rng.shuffle(madde_ids)
    test_maddeler = set(madde_ids[:round(len(madde_ids) * TEST_RATIO)])
    train_rows, test_rows, unsafe = [], [], 0

    def row(question, context, answer, kind, unit_id):
        return {"messages": build_messages(question, context, answer, system=SYSTEM_PROMPT_V2),
                "meta": {"type": kind, "unit_id": unit_id, "context_ids": [c["id"] for c in context],
                         "question": question}}

    for item in items:
        unit = by_id[item["unit_id"]]
        kitle = "personel" if unit["ozellikler"]["hedef_kitle"] == "personel" else None
        retrieved = retriever.retrieve(item["soru"], size=CONTEXT_SIZE, kitle=kitle)
        others = [a for a in retrieved if a["id"] != unit["id"]]
        is_test = unit["ozellikler"]["madde_id"] in test_maddeler
        if rng.random() < ORACLE_DROP_RATIO:
            context = others[:CONTEXT_SIZE]
            if context_answers(item["soru"], context, refusal_cache):
                unsafe += 1
                continue
            new = row(item["soru"], context, REFUSAL_ANSWER, "no_oracle", unit["id"])
        else:
            context = others[:CONTEXT_SIZE - 1] + [unit]
            rng.shuffle(context)
            new = row(item["soru"], context, answer_with_citation(unit, item["cevap"], item["fikralar"]),
                      "oracle", unit["id"])
        (test_rows if is_test else train_rows).append(new)

    for questions, kitle in ((v1.OUT_OF_SCOPE_QUESTIONS, None), (STAFF_OUT_OF_SCOPE, "personel")):
        for question in questions:
            context = retriever.retrieve(question, size=CONTEXT_SIZE, kitle=kitle)
            if context_answers(question, context, refusal_cache):
                unsafe += 1
                continue
            new = row(question, context, REFUSAL_ANSWER, "out_of_scope", "")
            (test_rows if rng.random() < TEST_RATIO else train_rows).append(new)

    REFUSAL_CACHE.write_text(json.dumps(refusal_cache, ensure_ascii=False), encoding="utf-8")
    rng.shuffle(train_rows)
    write_jsonl(train_rows, TRAIN)
    write_jsonl(test_rows, TEST)

    def summary(rows):
        counts = {}
        for r in rows:
            counts[r["meta"]["type"]] = counts.get(r["meta"]["type"], 0) + 1
        return f"{len(rows)} {counts}"

    print(f"Bağlamda cevabı olduğu için atılan reddetme örneği: {unsafe}")
    print(f"Train: {summary(train_rows)} -> {TRAIN}\nTest : {summary(test_rows)} -> {TEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
