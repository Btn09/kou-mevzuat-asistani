#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eksik konu raporu: web arayüzünde "bilgi yok" denen sorular (proje/bilgi_yok.jsonl).

Her soru için arama yeniden çalıştırılır ve iki gruba ayrılır:
- VERİDE YOK: en iyi parçanın puanı eşik altında -> eklenmesi gereken belge/konu.
- VERİ VAR, CEVAP YOK: ilgili belge bulunuyor ama model reddetti -> arama/model iyileştirmesi.
Aynı soru birden çok kez sorulduysa bir kez, sayısıyla gösterilir.

Kullanım:  python -X utf8 eksik_konular.py            (son 30 gün)
           python -X utf8 eksik_konular.py --gun 7
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

LOG = Path(__file__).resolve().parent / "bilgi_yok.jsonl"


def main() -> int:
    days = int(sys.argv[sys.argv.index("--gun") + 1]) if "--gun" in sys.argv else 30
    if not LOG.exists():
        print("Henüz 'bilgi yok' cevabı kaydedilmemiş.")
        return 0
    since = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    rows = [json.loads(line) for line in LOG.open(encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r.get("gun", "") >= since]
    from pdf_to_chromadb import normalize_for_matching
    counts = Counter(normalize_for_matching(r["soru"]).strip(" ?.!") for r in rows)
    original = {normalize_for_matching(r["soru"]).strip(" ?.!"): r["soru"] for r in rows}
    print(f"Son {days} gün: {len(rows)} 'bilgi yok' cevabı, {len(counts)} farklı soru\n")
    if not counts:
        return 0

    from rag_v3 import RELEVANCE_MIN, Retriever3
    retriever = Retriever3()
    missing, found = [], []
    for key, count in counts.most_common():
        question = original[key]
        scored = retriever.candidates(question)
        best = max(scored, key=lambda item: item[0], default=(-99.0, None))
        doc = best[1]["ozellikler"]["belge_adi"] if best[1] else "-"
        (missing if best[0] < RELEVANCE_MIN else found).append((count, question, best[0], doc))
    print(f"== VERİDE YOK ({len(missing)}) — eklenecek belge/konu adayı")
    for count, question, score, doc in missing:
        print(f"  [{count}x] {question}   (en yakın: {doc[:60]}, puan {score:.1f})")
    print(f"\n== VERİ VAR, CEVAP YOK ({len(found)}) — ilgili belge bulundu ama cevap verilmedi")
    for count, question, score, doc in found:
        print(f"  [{count}x] {question}   (bulunan: {doc[:60]}, puan {score:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
