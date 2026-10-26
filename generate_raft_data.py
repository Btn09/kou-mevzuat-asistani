#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""articles.jsonl'den RAFT (Retrieval-Augmented Fine-Tuning) eğitim verisi üretir.

Her eğitim örneği = soru + RAG'in getireceği gibi 4 madde + hedef cevap:
  - oracle      : doğru madde bağlamda; cevap + "Kaynak: Madde X, fıkra Y"
  - no_oracle   : doğru madde çıkarılmış; cevap = "yönetmelikte yok, birime danışın"
  - out_of_scope: yönetmelik dışı soru (yemekhane, yurt...); cevap = aynı reddetme

Bağlamdaki diğer maddeler (distractor) canlı sistemdekiyle aynı yolla seçilir:
soru bge-m3 ile fıkra chunk'larında aranır, chunk'lar parent maddelere çevrilir.

Kurulum:
    pip install ollama numpy
    ollama pull qwen3:14b
    ollama pull bge-m3

Önce:
    python pdf_to_chromadb.py --sadece-json   (articles.jsonl + prepared_chunks.jsonl)

Kullanım:
    python generate_raft_data.py --limit 3   # önce 3 maddede dene, qa_raw.jsonl'yi incele
    python generate_raft_data.py             # tümü (kaldığı yerden devam eder)

Çıktılar:
    qa_raw.jsonl     -> LLM'in ham soru-cevapları (önbellek; silinirse yeniden üretilir)
    raft_train.jsonl -> eğitim verisi (chat "messages" formatı)
    raft_test.jsonl  -> değerlendirme verisi (eğitimde hiç görülmeyen maddeler)
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import ollama

from raft_format import REFUSAL_ANSWER, article_fikra_numbers, build_messages, format_citation

# ------------------------- AYARLAR -------------------------
ARTICLES_JSONL = Path("articles.jsonl")
CHUNKS_JSONL = Path("prepared_chunks.jsonl")
QA_CACHE_JSONL = Path("qa_raw.jsonl")
REFUSAL_CHECK_CACHE = Path("red_kontrol_cache.json")  # "bağlam soruyu cevaplıyor mu" sonuçları
TRAIN_JSONL = Path("raft_train.jsonl")
TEST_JSONL = Path("raft_test.jsonl")

# Soru-cevap üreten "öğretmen" model; sadece veri üretir, eğitilmez ve sitede çalışmaz.
# qwen3:8b denendi: uydurma cevap ve JSON içinde sonsuz tekrar döngüsü üretti.
GEN_MODEL = "qwen3:14b"
EMBEDDING_MODEL = "bge-m3"  # pdf_to_chromadb.py ile aynı olmalı.
TEMPERATURE = 0.7

CONTEXT_SIZE = 4  # Bağlama konan madde sayısı (canlı sistemde de aynı olmalı).
ORACLE_DROP_RATIO = 0.2  # Örneklerin bu kadarında doğru madde çıkarılır -> reddetme öğrenilir.
TEST_ARTICLE_RATIO = 0.15  # Maddelerin bu kadarı sadece teste ayrılır.
SEED = 42

CHARS_PER_QUESTION = 150  # Madde uzunluğuna göre soru sayısı: len / bu değer
MIN_QUESTIONS = 3
MAX_QUESTIONS = 12

# Her tur farklı soru üslubu üretir; çeşitlilik modelin ezberlemesini azaltır.
# Örnek soru cümlesi verme: küçük modeller örneği kelimesi kelimesine kopyalıyor.
GENERATION_ROUNDS = [
    "Kısa, doğrudan ve resmi dilde sorular.",
    "Öğrencinin kendi durumunu bir cümleyle anlattığı, günlük konuşma dilinde sorular. "
    "Bazı SORULAR küçük harfle, noktalama olmadan veya ufak yazım hatalarıyla yazılmış olsun "
    "(bu serbestlik sadece sorular için; cevaplar her zaman düzgün Türkçe olacak).",
]

# Yönetmelikte cevabı olmayan sorular -> model reddetmeyi öğrenir.
OUT_OF_SCOPE_QUESTIONS = [
    "Yemekhanede öğle yemeği kaç lira?",
    "yurt başvurusu ne zaman başlıyor",
    "KYK bursuna nasıl başvururum?",
    "Kütüphane hafta sonu kaçta kapanıyor?",
    "Kampüse hangi otobüsler gidiyor?",
    "Öğrenci kartımı kaybettim ne yapmam lazım",
    "Kampüste wifi şifresi nedir?",
    "Spor salonunu öğrenciler ücretsiz kullanabiliyor mu?",
    "Yeni bir öğrenci kulübü nasıl kurulur?",
    "Mediko sosyal merkezinde psikolojik danışmanlık var mı?",
    "Otoparka araçla girmek için kart gerekiyor mu?",
    "Mezuniyet töreni hangi tarihte yapılacak?",
    "harç ücreti bu dönem kaç tl",
    "Erasmus hibesi aylık ne kadar?",
    "Bilgisayar mühendisliğinin taban puanı kaç?",
    "Kampüste kaç tane kantin var?",
    "Okulun e-posta hesabıma nasıl giriş yaparım?",
    "Yaz okulunda hangi dersler açılacak?",
    "Öğrenci affı çıktı mı?",
    "Hocamın ofis saatlerini nereden öğrenebilirim?",
    "Kocaeli Üniversitesi rektörü kim?",
    "kayıt dondurma ücreti ne kadar",
    "Umuttepe kampüsünde ATM var mı?",
    "Pedagojik formasyon başvurusu ne zaman?",
    # Reddetme örneklerinin çoğu "doğru madde çıkarıldı ama komşu madde cevaplıyor" kontrolüne
    # takıldığı için reddetme, etiketi kesin olan bu sorularla da öğretilir.
    "Kampüste kaç tane yemekhane var?",
    "Öğrenci yurdunun aylık ücreti ne kadar?",
    "Kampüs içinde ring servisi saat kaçta başlıyor?",
    "Kütüphaneden en fazla kaç kitap ödünç alabilirim?",
    "Okulun Instagram hesabı hangisi?",
    "Bahar şenliği ne zaman yapılacak?",
    "Bilgisayar mühendisliği bölüm başkanı kim?",
    "Kampüste kırtasiye nerede?",
    "Öğrenci işleri telefon numarası nedir?",
    "Laboratuvarlar hafta sonu açık mı?",
    "Kocaeli Üniversitesi kaç yılında kuruldu?",
    "Kampüse bisikletle girebilir miyim?",
    "Erasmus için dil sınavı puanı kaç olmalı?",
    "Okulda kaç tane öğrenci kulübü var?",
    "Yurtta misafir kalabilir mi?",
    "Kampüs kartıma nasıl para yüklerim?",
    "Mezunlar derneğine nasıl üye olurum?",
    "Mühendislik fakültesi hangi kampüste?",
    "Kampüste eczane var mı?",
    "Öğrencilere ücretsiz Microsoft Office veriliyor mu?",
    "ÖSYM sınav sonuçları ne zaman açıklanır?",
    "Kampüsteki kafeterya kaçta kapanıyor?",
    "Askerlik tecil işlemleri için nereye başvurmalıyım?",
    "Okulun posta kodu nedir?",
    "Kampüs içi engelli asansörleri nerede?",
    "Kocaeli'de öğrenci evi kiraları ne kadar?",
]

QA_SCHEMA = {
    "type": "object",
    "properties": {
        "ornekler": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "soru": {"type": "string"},
                    "cevap": {"type": "string"},
                    "fikralar": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["soru", "cevap", "fikralar"],
            },
        }
    },
    "required": ["ornekler"],
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "degerlendirmeler": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "no": {"type": "integer"},
                    "metne_sadik": {"type": "boolean"},
                    "soruyu_cevapliyor": {"type": "boolean"},
                },
                "required": ["no", "metne_sadik", "soruyu_cevapliyor"],
            },
        }
    },
    "required": ["degerlendirmeler"],
}

# Öğrenci metni görmeyecek; metne gönderme yapan sorular işe yaramaz.
TEXT_REFERENCE_RE = re.compile(r"(?i)(bu madde|yukarıdaki|metne göre|metinde|maddeye göre|fıkra)")
# Prompt talimatını cevaba kopyalama ("[1] numaralı fıkraya dayanıyorsa") veya metne gönderme.
ANSWER_LEAK_RE = re.compile(r"(?i)(\[\d+\]|numaralı|fıkra|bu madde|maddeye göre|yönetmelik metninde)")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def question_count(article: dict[str, Any]) -> int:
    return max(MIN_QUESTIONS, min(MAX_QUESTIONS, round(len(article["text"]) / CHARS_PER_QUESTION)))


def generate_qa(article: dict[str, Any], style: str, count: int) -> list[dict[str, Any]]:
    """Bir madde için LLM'den soru-cevap çiftleri ister."""
    prompt = (
        "Aşağıda Kocaeli Üniversitesi Ön Lisans ve Lisans Eğitim-Öğretim Yönetmeliği'nden bir madde var.\n\n"
        f"{article['text']}\n\n"
        f"Bu maddeye dayanarak bir öğrencinin sorabileceği {count} FARKLI soru ve cevabını üret.\n"
        f"Soru üslubu: {style}\n"
        "Kurallar:\n"
        "- SADECE cevabı madde metninde açıkça yazan sorular sor. Metinde cevabı olmayan "
        "durumlar (ör. metinde geçmeyen mezuniyet, kayıt, başka üniversite konuları) hakkında soru sorma.\n"
        "- Öğrenci bu metni görmüyor: soruda 'bu madde', 'metne göre', 'fıkra' gibi ifadeler kullanma.\n"
        "- Sorular maddenin farklı noktalarını kapsasın, birbirinin tekrarı olmasın.\n"
        "- Cevap YALNIZCA madde metnine dayansın; metinde olmayan bilgi, sayı veya tarih ekleme, yorum yapma.\n"
        "- Metin bir konuyu Senato'ya veya başka bir mevzuata bırakıyorsa cevapta bunu söyle.\n"
        "- Cevap soruyu DOĞRUDAN yanıtlasın. 'Bu konu yönetmelik kapsamında düzenlenir' gibi kaçamak "
        "cevaplar verme; metin soruyu tam cevaplamıyorsa o soruyu hiç sorma.\n"
        "- Cevap 1-3 doğal cümle olsun, öğrenciye nazik ve resmi hitap etsin; 'Siz,' diye başlama.\n"
        "- Cevaplar HER ZAMAN düzgün Türkçe yazılsın: Türkçe karakterlerle (ç, ğ, ı, ö, ş, ü), "
        "büyük harfle başlayan ve noktalı cümlelerle.\n"
        "- Cevap metnine madde/fıkra numarası, köşeli parantez veya kaynak YAZMA. "
        "Dayandığın fıkra numaralarını yalnızca 'fikralar' alanına yaz.\n"
        "JSON olarak cevap ver."
    )
    response = ollama.chat(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=QA_SCHEMA,
        # num_predict: model JSON içinde döngüye girerse dakikalarca üretmesin.
        options={"temperature": TEMPERATURE, "num_ctx": 8192, "num_predict": 3000},
        think=False,
    )
    return json.loads(response["message"]["content"]).get("ornekler", [])


def judge_examples(article: dict[str, Any], examples: list[dict[str, Any]]) -> None:
    """Hakem kontrolü: her örneğe 'kontrol' alanı ekler (metne sadık mı, soruyu cevaplıyor mu)."""
    if not examples:
        return
    numbered = "\n\n".join(
        f"#{index}\nSoru: {example.get('soru', '')}\nCevap: {example.get('cevap', '')}"
        for index, example in enumerate(examples, start=1)
    )
    prompt = (
        "Aşağıda bir yönetmelik maddesi ve bu maddeden üretilmiş soru-cevap çiftleri var.\n\n"
        f"MADDE METNİ:\n{article['text']}\n\n"
        f"SORU-CEVAPLAR:\n{numbered}\n\n"
        "Her çifti titizlikle değerlendir:\n"
        "- metne_sadik: Cevaptaki HER bilgi madde metninde açıkça yazıyor mu? Metinde olmayan bir "
        "çıkarım, genelleme veya ekleme varsa false.\n"
        "- soruyu_cevapliyor: Cevap sorulan şeyi doğrudan yanıtlıyor mu? Kaçamak, alakasız veya "
        "yarım cevapsa false.\n"
        "Emin değilsen false ver. JSON olarak cevap ver."
    )
    response = ollama.chat(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=JUDGE_SCHEMA,
        options={"temperature": 0, "num_ctx": 8192, "num_predict": 2000},
        think=False,
    )
    verdicts = json.loads(response["message"]["content"]).get("degerlendirmeler", [])
    for verdict in verdicts:
        number = verdict.get("no")
        if isinstance(number, int) and 1 <= number <= len(examples):
            examples[number - 1]["kontrol"] = {
                "metne_sadik": bool(verdict.get("metne_sadik")),
                "soruyu_cevapliyor": bool(verdict.get("soruyu_cevapliyor")),
            }


ANSWERABLE_SCHEMA = {
    "type": "object",
    "properties": {"destekleyen_alinti": {"type": "string"}},
    "required": ["destekleyen_alinti"],
}
MIN_QUOTE_CHARS = 25


def normalize_quote(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def context_answers(question: str, context: list[dict[str, Any]], cache: dict[str, bool]) -> bool:
    """Reddetme örneği güvenli mi? Doğru madde çıkarılsa bile başka bir madde (ör. Madde 4 tanımları
    ile Madde 28) soruyu cevaplayabilir; o zaman 'reddet' etiketi yanlış olur.

    Evet/hayır sorusu 14B'de fazla hoşgörülü çıktı (rektörün adını bile 'cevaplanabilir' saydı).
    Bunun yerine doğrudan cevabı veren cümleyi birebir alıntılaması istenir; alıntı bağlamda
    gerçekten geçiyorsa soru cevaplanabilir sayılır.
    """
    key = question + "|" + ",".join(article["id"] for article in context)
    if key not in cache:
        articles_text = "\n\n".join(article["text"] for article in context)
        prompt = (
            f"YÖNETMELİK MADDELERİ:\n{articles_text}\n\nÖĞRENCİ SORUSU: {question}\n\n"
            "Bu maddelerde sorunun DOĞRUDAN cevabını veren bir cümle var mı? Varsa o cümleyi "
            "maddeden kelimesi kelimesine kopyala. Konu sadece benzer ise, soru maddelerde geçmeyen "
            "bir bilgi (isim, tarih, ücret, yer vb.) istiyorsa veya emin değilsen destekleyen_alinti "
            "alanını boş bırak. JSON olarak cevap ver."
        )
        response = ollama.chat(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format=ANSWERABLE_SCHEMA,
            options={"temperature": 0, "num_ctx": 8192, "num_predict": 400},
            think=False,
        )
        quote = normalize_quote(json.loads(response["message"]["content"]).get("destekleyen_alinti", ""))
        cache[key] = len(quote) >= MIN_QUOTE_CHARS and quote_found(quote, normalize_quote(articles_text))
    return cache[key]


def quote_found(quote: str, context: str, ngram: int = 5, min_ratio: float = 0.6) -> bool:
    """Alıntı bağlamda geçiyor mu? Model alıntıyı birkaç kelime farkla verebiliyor
    (başlık + fıkra birleştirme vb.); 5'li kelime gruplarının çoğu eşleşirse geçerli sayılır."""
    if quote in context:
        return True
    words = quote.split()
    if len(words) < ngram:
        return False
    grams = [" ".join(words[i:i + ngram]) for i in range(len(words) - ngram + 1)]
    return sum(gram in context for gram in grams) / len(grams) >= min_ratio


def fill_qa_cache(articles: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Önbellekte olmayan maddeler için soru üretir; kesilirse kaldığı yerden devam eder."""
    cache: dict[str, list[dict[str, Any]]] = {}
    if QA_CACHE_JSONL.exists():
        for row in read_jsonl(QA_CACHE_JSONL):
            cache[row["article_id"]] = row["examples"]

    missing = [article for article in articles if article["id"] not in cache]
    if missing:
        print(f"Soru üretimi: {len(missing)} madde, model {GEN_MODEL}")
    with QA_CACHE_JSONL.open("a", encoding="utf-8") as cache_file:
        for index, article in enumerate(missing, start=1):
            label = article["metadata"]["article_label"]
            examples: list[dict[str, Any]] = []
            try:
                for style in GENERATION_ROUNDS:
                    examples.extend(generate_qa(article, style, question_count(article)))
                judge_examples(article, examples)
            except Exception as exc:
                print(f"  [{index}/{len(missing)}] {label}: HATA ({exc}); atlandı, tekrar çalıştırınca denenecek")
                continue
            cache[article["id"]] = examples
            cache_file.write(json.dumps({"article_id": article["id"], "examples": examples}, ensure_ascii=False) + "\n")
            cache_file.flush()
            approved = sum(
                1 for example in examples
                if example.get("kontrol", {}).get("metne_sadik") and example.get("kontrol", {}).get("soruyu_cevapliyor")
            )
            print(f"  [{index}/{len(missing)}] {label}: {len(examples)} örnek, hakem onayı {approved}")
    return cache


def clean_answer(answer: str) -> str:
    """LLM'in kurala rağmen eklediği kaynak/madde ifadelerini temizler; kaynağı kod ekler."""
    answer = re.sub(r"(?im)^\s*kaynak\s*:.*$", "", answer)
    answer = re.sub(r"\s*\((?:geçici\s+)?madde[^)]*\)", "", answer, flags=re.IGNORECASE)
    # 14B, kurala rağmen cevapların çoğuna "Siz, ..." diye başlıyor; geri kalan cümle düzgün.
    answer = re.sub(r"^\s*siz\s*,\s*", "", answer, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", answer).strip()


def tr_upper_first(text: str) -> str:
    """Cevabın ilk harfini Türkçe kurallarıyla büyütür ('ilişik' -> 'İlişik')."""
    if not text:
        return text
    first = {"i": "İ", "ı": "I"}.get(text[0], text[0].upper())
    return first + text[1:]


def validate_examples(article: dict[str, Any], raw_examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Hatalı örnekleri eler. Dönüş: (geçerli örnekler, eleme sebebi sayıları)."""
    valid_fikralar = article_fikra_numbers(article["text"])
    article_numbers = set(NUMBER_RE.findall(article["text"]))
    seen_questions: set[str] = set()
    kept: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for example in raw_examples:
        question = str(example.get("soru", "")).strip()
        answer = clean_answer(str(example.get("cevap", "")))
        if not (10 <= len(question) <= 300) or not (15 <= len(answer) <= 800):
            reject("uzunluk")
            continue
        if TEXT_REFERENCE_RE.search(question):
            reject("metne_gonderme")
            continue
        if ANSWER_LEAK_RE.search(answer):
            reject("cevapta_talimat_sizintisi")
            continue
        # Model cevapları bozuk Türkçe ile öğrenmesin ("danismaninizin onayiyla").
        if len(answer) >= 40 and not TURKISH_CHARS & set(answer):
            reject("bozuk_turkce")
            continue
        verdict = example.get("kontrol", {})
        if not (verdict.get("metne_sadik") and verdict.get("soruyu_cevapliyor")):
            reject("hakem_reddetti")
            continue
        question_key = re.sub(r"\W+", " ", question.casefold()).strip()
        if question_key in seen_questions:
            reject("tekrar")
            continue
        # Cevaptaki her sayı madde metninde geçmeli; geçmiyorsa model uydurmuştur.
        if any(number not in article_numbers for number in NUMBER_RE.findall(answer)):
            reject("uydurma_sayi")
            continue

        fikralar = [n for n in example.get("fikralar", []) if isinstance(n, int) and n in valid_fikralar]
        if not fikralar and len(valid_fikralar) == 1:
            # Tek fıkralı maddede kaynak formatı tutarlı kalsın: "Madde 30, fıkra 1".
            fikralar = list(valid_fikralar)
        answer = tr_upper_first(answer)
        seen_questions.add(question_key)
        kept.append({"soru": question, "cevap": answer, "fikralar": fikralar})
    return kept, rejected


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(ollama.embed(model=EMBEDDING_MODEL, input=texts[start:start + batch_size])["embeddings"])
    matrix = np.asarray(vectors, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


class ArticleRetriever:
    """Canlı RAG ile aynı mantık: fıkra chunk'larında ara, parent maddelere çevir."""

    def __init__(self, chunks: list[dict[str, Any]]):
        self.parent_ids = [chunk["metadata"]["parent_article_id"] for chunk in chunks]
        self.chunk_vectors = embed_texts([chunk["text"] for chunk in chunks])

    def ranked_article_ids(self, question_vector: np.ndarray) -> list[str]:
        order = np.argsort(-(self.chunk_vectors @ question_vector))
        ranked: list[str] = []
        for chunk_index in order:
            parent_id = self.parent_ids[chunk_index]
            if parent_id and parent_id not in ranked:
                ranked.append(parent_id)
        return ranked


def main() -> int:
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    for path in (ARTICLES_JSONL, CHUNKS_JSONL):
        if not path.exists():
            print(f"{path} bulunamadı. Önce: python pdf_to_chromadb.py --sadece-json")
            return 1

    articles = read_jsonl(ARTICLES_JSONL)
    articles_by_id = {article["id"]: article for article in articles}
    rng = random.Random(SEED)

    try:
        cache = fill_qa_cache(articles[:limit] if limit else articles)
    except KeyboardInterrupt:
        print("Durduruldu; üretilenler qa_raw.jsonl'de, tekrar çalıştırınca devam eder.")
        return 130

    # ---- Doğrulama ----
    qa_items: list[dict[str, Any]] = []
    rejected_total: dict[str, int] = {}
    raw_total = 0
    for article_id, raw_examples in cache.items():
        if article_id not in articles_by_id:
            continue
        raw_total += len(raw_examples)
        kept, rejected = validate_examples(articles_by_id[article_id], raw_examples)
        for reason, count in rejected.items():
            rejected_total[reason] = rejected_total.get(reason, 0) + count
        qa_items.extend({**example, "article_id": article_id} for example in kept)
    print(f"\nHam örnek: {raw_total}, geçerli: {len(qa_items)}, elenen: {rejected_total or 'yok'}")
    if not qa_items:
        print("Geçerli örnek yok.")
        return 1

    # ---- Bağlam (distractor) seçimi: canlı sistemle aynı retrieval ----
    print("Embedding hesaplanıyor (bge-m3)...")
    retriever = ArticleRetriever(read_jsonl(CHUNKS_JSONL))
    questions = [item["soru"] for item in qa_items] + OUT_OF_SCOPE_QUESTIONS
    question_vectors = embed_texts(questions)

    # Madde bazlı train/test ayrımı: test maddeleri eğitimde hiç görülmez.
    used_article_ids = sorted({item["article_id"] for item in qa_items})
    rng.shuffle(used_article_ids)
    test_count = max(1, round(len(used_article_ids) * TEST_ARTICLE_RATIO)) if len(used_article_ids) > 1 else 0
    test_article_ids = set(used_article_ids[:test_count])

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    refusal_cache: dict[str, bool] = (
        json.loads(REFUSAL_CHECK_CACHE.read_text(encoding="utf-8")) if REFUSAL_CHECK_CACHE.exists() else {}
    )
    unsafe_refusals = 0
    print("Reddetme örnekleri kontrol ediliyor (bağlam soruyu cevaplıyor mu?)...")

    def make_row(question: str, context_ids: list[str], answer: str, example_type: str, article_id: str, reference: str) -> dict[str, Any]:
        context = [articles_by_id[article_id_] for article_id_ in context_ids]
        return {
            "messages": build_messages(question, context, answer),
            "meta": {
                "type": example_type,
                "article_id": article_id,
                "context_ids": context_ids,
                "question": question,
                "reference_answer": reference,
            },
        }

    for item, question_vector in zip(qa_items, question_vectors[: len(qa_items)]):
        oracle_id = item["article_id"]
        others = [article_id for article_id in retriever.ranked_article_ids(question_vector) if article_id != oracle_id]
        oracle_label = articles_by_id[oracle_id]["metadata"]["article_label"]
        full_answer = f"{item['cevap']}\n\n{format_citation(oracle_label, item['fikralar'])}"

        if rng.random() < ORACLE_DROP_RATIO:
            context_ids = others[:CONTEXT_SIZE]
            if context_answers(item["soru"], [articles_by_id[i] for i in context_ids], refusal_cache):
                unsafe_refusals += 1
                continue
            row = make_row(item["soru"], context_ids, REFUSAL_ANSWER, "no_oracle", oracle_id, full_answer)
        else:
            context_ids = others[: CONTEXT_SIZE - 1] + [oracle_id]
            rng.shuffle(context_ids)  # Doğru madde her zaman aynı sırada olmasın.
            row = make_row(item["soru"], context_ids, full_answer, "oracle", oracle_id, full_answer)
        (test_rows if oracle_id in test_article_ids else train_rows).append(row)

    out_of_scope_vectors = question_vectors[len(qa_items):]
    out_of_scope_order = list(range(len(OUT_OF_SCOPE_QUESTIONS)))
    rng.shuffle(out_of_scope_order)
    out_of_scope_test = set(out_of_scope_order[: max(1, round(len(out_of_scope_order) * TEST_ARTICLE_RATIO))])
    for index, (question, question_vector) in enumerate(zip(OUT_OF_SCOPE_QUESTIONS, out_of_scope_vectors)):
        context_ids = retriever.ranked_article_ids(question_vector)[:CONTEXT_SIZE]
        if context_answers(question, [articles_by_id[i] for i in context_ids], refusal_cache):
            unsafe_refusals += 1
            print(f"  UYARI: kapsam dışı sanılan soru bağlamda cevaplanabiliyor, atlandı: {question}")
            continue
        row = make_row(question, context_ids, REFUSAL_ANSWER, "out_of_scope", "", REFUSAL_ANSWER)
        (test_rows if index in out_of_scope_test else train_rows).append(row)

    REFUSAL_CHECK_CACHE.write_text(json.dumps(refusal_cache, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"Bağlamda cevabı olduğu için atılan reddetme örneği: {unsafe_refusals}")

    rng.shuffle(train_rows)
    write_jsonl(train_rows, TRAIN_JSONL)
    write_jsonl(test_rows, TEST_JSONL)

    def summary(rows: list[dict[str, Any]]) -> str:
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["meta"]["type"]] = counts.get(row["meta"]["type"], 0) + 1
        return f"{len(rows)} örnek {counts}"

    test_labels = sorted(articles_by_id[article_id]["metadata"]["article_label"] for article_id in test_article_ids)
    print(f"\nTrain: {summary(train_rows)} -> {TRAIN_JSONL}")
    print(f"Test : {summary(test_rows)} -> {TEST_JSONL}")
    print(f"Test maddeleri (eğitimde yok): {', '.join(test_labels) or '-'}")
    print("\nqa_raw.jsonl'den rastgele 20-30 örneği gözle kontrol etmeyi unutma.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
