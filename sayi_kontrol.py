#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cevaptaki sayıların (tutar, gün/ay/yıl, yüzde, tarih) bağlamdaki maddelerde geçip geçmediğini denetler.

Denetim (19.10): model hatalarının ~1/3'ü yanlış/çelişkili bilgi; en tehlikelisi kaynakta olmayan sayı (uydurma
tutar, yanlış gün sayısı). Bağlamda bulunmayan sayı varsa cevabın altına uyarı eklenir (cevap değiştirilmez).
Yazıyla sayılar ("yirmi gün", "onsekiz ay", "yüzde otuz") rakama çevrilerek karşılaştırılır.
"""

from __future__ import annotations

import re
from typing import Any

UNITS = {"bir": 1, "iki": 2, "üç": 3, "dört": 4, "beş": 5, "altı": 6, "yedi": 7, "sekiz": 8, "dokuz": 9}
TENS = {"on": 10, "yirmi": 20, "otuz": 30, "kırk": 40, "elli": 50, "altmış": 60, "yetmiş": 70, "seksen": 80,
        "doksan": 90}
WORD_RE = re.compile(r"\b(?:(" + "|".join(TENS) + r")\s?(" + "|".join(UNITS) + r")?|(" + "|".join(UNITS) + r"))\b",
                     re.IGNORECASE)
# Rakam dizileri: 1.438,00 / 11.348 / 2026 / 15,5 / %70 ; harf bitişik olanlar (YPZ301, B1) ders/seviye kodu sayılmaz
NUMBER_RE = re.compile(r"(?<![\w.,])(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?(?![\w])")
SKIP_RE = re.compile(r"(?i)\b(madde|fıkra|bent|sayılı|sayı|no|ek)\s*$")


def _value(integer: str, fraction: str | None) -> str:
    number = integer.replace(".", "")
    if fraction and fraction.strip("0"):
        return f"{int(number)},{fraction.rstrip('0')}"
    return str(int(number))


def numbers(text: str) -> set[str]:
    found = set()
    for match in NUMBER_RE.finditer(text):
        found.add(_value(match.group(1), match.group(2)))
    for match in WORD_RE.finditer(text.lower()):
        tens, unit, alone = match.groups()
        found.add(str(TENS.get(tens or "", 0) + UNITS.get(unit or alone or "", 0)))
    return found


def answer_numbers(answer: str) -> list[tuple[str, str]]:
    """(değer, cevaptaki yazılışı); madde/fıkra/kanun numaraları ve liste imleri ("1)", "2.") atlanır."""
    out = []
    for line in answer.split("\n"):
        line = re.sub(r"^\s*(\d{1,2}[.)]|[-•*])\s+", "", line)  # liste imi
        for match in NUMBER_RE.finditer(line):
            if SKIP_RE.search(line[:match.start()]):
                continue
            if re.match(r"\s*sayılı", line[match.end():]):  # "2547 sayılı"
                continue
            out.append((_value(match.group(1), match.group(2)), match.group(0)))
    return out


def unsupported(answer: str, context: list[dict[str, Any]], question: str = "") -> list[str]:
    """Bağlamda (ve soruda) geçmeyen sayılar; 0 ve 1 gibi tek basamaklılar sözcük olarak da sık geçtiğinden
    yalnız bağlamın hiçbir yerinde yoksa sayılır."""
    body = answer.split("\nKaynak:", 1)[0]
    body = re.sub(r"Not: Bu bilgi[^\n]*", "", body)  # sistemin eklediği yürürlük notu (tarih kaynakta değil)
    known = numbers(question)
    for article in context:
        known |= numbers(article["metin"])
        known |= numbers(article["ozellikler"].get("belge_adi", ""))
    missing = []
    for value, written in answer_numbers(body):
        if value not in known and written not in missing:
            missing.append(written)
    return missing


def with_warning(answer: str, missing: list[str]) -> str:
    if not missing:
        return answer
    note = ("⚠️ Not: Cevaptaki şu değer(ler) kaynak metinde bulunamadı, lütfen kaynağı kontrol edin: "
            + ", ".join(missing[:5]))
    body, sep, citation = answer.partition("\n\nKaynak:")
    return f"{body.rstrip()}\n\n{note}{sep}{citation}"
