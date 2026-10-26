#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eğitim verisi, fine-tuning ve canlı RAG sorgusunun ORTAK formatı.

Model, eğitimde gördüğü prompt formatına göre davranır. Bu yüzden
generate_raft_data.py, eğitim script'i ve sitedeki RAG sorgusu prompt'u
mutlaka bu dosyadaki fonksiyonlarla kurmalıdır. Burayı değiştirirsen
veriyi yeniden üretip modeli yeniden eğitmen gerekir.
"""

from __future__ import annotations

import re
from typing import Any

SYSTEM_PROMPT = (
    "Sen Kocaeli Üniversitesi öğrencilerine yardımcı olan bir asistansın. "
    "Soruları YALNIZCA verilen yönetmelik maddelerine dayanarak cevapla. "
    "Cevabını kısa ve anlaşılır tut, sonunda dayandığın maddeyi 'Kaynak:' satırıyla belirt. "
    "Verilen maddelerde cevap yoksa bilgi uydurma; bunu açıkça söyle ve öğrenciyi "
    "ilgili birime yönlendir."
)

# Çok belgeli sürüm (veri/ klasöründeki 31 belge). Eski kou-asistan modeli SYSTEM_PROMPT ile eğitildi;
# onu bozmamak için yeni sistem bu prompt'u açıkça seçer (build_messages(..., system=SYSTEM_PROMPT_V2)).
SYSTEM_PROMPT_V2 = (
    "Sen Kocaeli Üniversitesi öğrencilerine ve akademik personeline yardımcı olan bir asistansın. "
    "Soruları YALNIZCA verilen mevzuat maddelerine dayanarak cevapla. Her maddenin hangi belgeden "
    "geldiği başında yazar; belgeler farklı gruplara (ön lisans/lisans, Tıp, Diş Hekimliği, "
    "Konservatuvar, lisansüstü öğrencileri, öğretim elemanları) uygulanır, soruya uygun olanı kullan. "
    "Cevabını kısa ve anlaşılır tut, sonunda dayandığın belgeyi ve maddeyi 'Kaynak:' satırıyla belirt. "
    "Verilen maddelerde cevap yoksa bilgi uydurma; bunu açıkça söyle ve öğrenciyi ilgili birime yönlendir."
)

# v3 hattı: kullanıcılar (çoğunlukla hocalar) yüzeysel değil ayrıntılı cevap istiyor. V2 ile aynı kurallar;
# yalnız "kısa tut" yerine ilgili hükmün tüm koşul, istisna, süre ve tutarlarını aktarması isteniyor.
SYSTEM_PROMPT_V3 = (
    "Sen Kocaeli Üniversitesi öğrencilerine ve akademik personeline yardımcı olan bir asistansın. "
    "Soruları YALNIZCA verilen mevzuat maddelerine dayanarak cevapla. Her maddenin hangi belgeden "
    "geldiği başında yazar; belgeler farklı gruplara (ön lisans/lisans, Tıp, Diş Hekimliği, "
    "Konservatuvar, lisansüstü öğrencileri, öğretim elemanları) uygulanır, soruya uygun olanı kullan. "
    "Cevabını ayrıntılı ver: soruyla ilgili hükümlerdeki tüm koşulları, istisnaları, süreleri, tarihleri ve "
    "tutarları eksiksiz aktar; birden çok durum varsa her birini ayrı madde işaretiyle yaz. "
    "Sonunda dayandığın belgeyi ve maddeyi 'Kaynak:' satırıyla belirt. "
    "Verilen maddelerde cevap yoksa bilgi uydurma; bunu açıkça söyle ve öğrenciyi ilgili birime yönlendir."
)

REFUSAL_ANSWER = (
    "Bu sorunun cevabı elimdeki yönetmelik maddelerinde yer almıyor. "
    "Doğru bilgi için Öğrenci İşleri Daire Başkanlığı'na veya bölüm sekreterliğine danışmanızı öneririm."
)

FIKRA_LINE_RE = re.compile(r"(?m)^\((\d+)\)")


def article_fikra_numbers(article_text: str) -> set[int]:
    """Madde metnindeki fıkra numaraları: '(1) ...', '(2) ...'."""
    return {int(number) for number in FIKRA_LINE_RE.findall(article_text)}


def tr_capitalize(word: str) -> str:
    """Türkçe kurallarıyla 'GEÇİCİ' -> 'Geçici' (str.capitalize 'i̇' gibi bozuk harf üretir)."""
    lowered = word.replace("İ", "i").replace("I", "ı").lower()
    first = {"i": "İ", "ı": "I"}.get(lowered[:1], lowered[:1].upper())
    return first + lowered[1:]


def format_citation(article_label: str, fikralar: list[int], belge: str = "", baslik: str = "") -> str:
    """'MADDE 30', [1] -> 'Kaynak: Madde 30, fıkra 1'.

    belge verilirse (çok belgeli sürüm): 'Kaynak: KOÜ Yaz Öğretimine İlişkin Esaslar, Esas 2, fıkra 1'.
    Kılavuz bölümlerinde (KISIM n) numaralı maddeler fıkra değil madde olarak yazılır:
    'Kaynak: KOÜ ... Kılavuzu – Kurum içi yatay geçiş başvuru koşulları, 10. madde'.
    """
    numbers = ", ".join(str(number) for number in sorted(set(fikralar)))
    if article_label.startswith("KISIM") and belge:
        title = re.sub(r"\s+–\s+fıkra [\d\-]+$", "", baslik) or article_label
        return f"Kaynak: {belge} – {title}" + (f", {numbers}. madde" if numbers else "")
    label = " ".join(tr_capitalize(word) for word in article_label.split())
    prefix = f"Kaynak: {belge}, {label}" if belge else f"Kaynak: {label}"
    return prefix + (f", fıkra {numbers}" if numbers else "")


def format_context(articles: list[dict[str, Any]]) -> str:
    """Getirilen maddeleri numaralı kaynak blokları halinde birleştirir.

    Maddenin öğrenci dilindeki karşılıkları varsa başlığın altına eklenir; yoksa model
    'kayıt dondurma' sorusunu 'izinli sayılma' maddesiyle bağlayamayıp reddediyor.
    """
    blocks = []
    for index, article in enumerate(articles, start=1):
        # Yeni şema: {"metin", "ozellikler"}; eski tek belgelik sistem (kou-asistan): {"text", "metadata"}.
        text = (article.get("metin") or article["text"]).strip()
        metadata = article.get("ozellikler") or article.get("metadata", {})
        aliases = metadata.get("esanlamlilar") or metadata.get("student_aliases")
        extra_lines = []
        if aliases:
            extra_lines.append(f"(Öğrencilerin bu konu için kullandığı ifadeler: {aliases})")
        if metadata.get("student_note"):
            extra_lines.append(f"(Not: {metadata['student_note']})")
        if metadata.get("belge_notu"):  # ör. yürürlükten kalkmış tebliğ uyarısı
            extra_lines.append(f"(UYARI: {metadata['belge_notu']})")
        if extra_lines:
            heading, _, body = text.partition("\n")
            text = "\n".join([heading, *extra_lines, body])
        document = metadata.get("belge_adi")  # çok belgeli sürümde maddenin geldiği belge
        blocks.append(f"[Kaynak {index}] {document}\n{text}" if document else f"[Kaynak {index}]\n{text}")
    return "\n\n".join(blocks)


def build_messages(
    question: str, articles: list[dict[str, Any]], answer: str | None = None, system: str = SYSTEM_PROMPT
) -> list[dict[str, str]]:
    """Chat formatında mesaj listesi. answer verilirse eğitim örneği olur."""
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            # Çok belgeli sürümde bağlamda kanun, yönerge, tebliğ de var; eski model "Yönetmelik" ile eğitildi.
            "content": f"{'Mevzuat' if system in (SYSTEM_PROMPT_V2, SYSTEM_PROMPT_V3) else 'Yönetmelik'} maddeleri:\n\n"
                       f"{format_context(articles)}\n\nSoru: {question.strip()}",
        },
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer.strip()})
    return messages
