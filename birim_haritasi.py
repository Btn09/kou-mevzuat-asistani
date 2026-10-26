#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bölüm / program adı -> birim (fakülte, MYO) haritası: "İnşaat Mühendisliği", "Kimya Bölümü", "FEF" gibi yazımlar
fakülteye eşlenir (23.10: "Fen Edebiyat"ı listede bulamayan kullanıcı "Diğer"i seçince İnşaat duyurusu geliyordu).

Kaynaklar: AKTS program listesi (veri/web/akts/programlar.json; lisans/önlisans) + bölüm sitelerinin adları
(site_bilgisi.json + web_belgeleri.site_birim). Birden çok birime düşen adlar ve günlük sözcüklerle çakışan adlar
alınmaz. Çıktı: veri/web/bolum_birim.json {normalize ad: birim}; rag_v2.UNIT_RULES bunu yükler.

Kullanım:  python -X utf8 birim_haritasi.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from pdf_to_chromadb import normalize_for_matching
from rag_v2 import BIRIMLER
from web_belgeleri import site_birim, tr_title

WEB = Path(__file__).resolve().parent.parent / "veri" / "web"
# Günlük dilde başka anlamı olan ya da tek başına belirsiz adlar
AMBIGUOUS = {"tarih", "kimya", "fizik", "matematik", "biyoloji", "isletme", "iktisat", "hukuk", "felsefe", "sosyoloji",
             "psikoloji", "muzik", "resim", "heykel", "grafik", "tekstil", "turizm", "ebelik", "hemsirelik", "mimarlik",
             "performans", "pazarlama", "elektrik", "metalurji", "fotograf", "rekreasyon", "anestezi", "makine"}
# Kısaltmalar (site adları): kullanıcılar sık yazar
ABBREVIATIONS = {"fef": "fen_edebiyat", "mtf": "mimarlik", "sbf": "sbf", "gsf": "gsf", "kmyo": "kocaeli_myo",
                 "kshmyo": "kshmyo", "foimyo": "ford_otosan_myo", "hakmyo": "hakmyo",
                 "muh fak": "muhendislik", "muhendislik fak": "muhendislik", "tip fak": "tip", "iibf": "isletme"}


def clean_program(name: str) -> str:
    name = re.sub(r"\((iö|io|ii\. ?ogretim|ikinci ogretim|uzaktan|ingilizce|%?\d+ ?ingilizce)\)", " ",
                  normalize_for_matching(name))
    return " ".join(name.replace("-", " ").replace(",", " ").split())


def main() -> int:
    unit_by_name = {normalize_for_matching(v): k for k, v in BIRIMLER.items() if k not in ("genel", "enstitu")}
    # AKTS'deki birim adları BIRIMLER adlarıyla birebir değil ("FEN - EDEBİYAT FAKÜLTESİ")
    def faculty_key(name: str) -> str | None:
        norm = " ".join(normalize_for_matching(name).replace("-", " ").split())
        for full, key in unit_by_name.items():
            if " ".join(full.replace("-", " ").split()) == norm:
                return key
        return None

    candidates: dict[str, set[str]] = defaultdict(set)
    display: dict[str, str] = {}  # normalize ad -> Türkçe görünen ad (bölüm seçme düğmeleri için)
    programs = json.loads((WEB / "akts" / "programlar.json").read_text(encoding="utf-8"))
    for program in programs:
        if program.get("duzey") not in ("Lisans", "Önlisans", "Ön Lisans"):
            continue
        key = faculty_key(program["birim"])
        if key:
            name = clean_program(program["program"])
            candidates[name].add(key)
            shown = re.sub(r"\s*\([^)]*\)", "", program["program"]).strip()
            display.setdefault(name, re.sub(r" (Ve|İle) ", lambda m: f" {m.group(1).lower()} ",
                                             tr_title(shown) if shown.isupper() or shown.istitle() else shown))
    meta = json.loads((WEB / "site_bilgisi.json").read_text(encoding="utf-8"))
    for host, info in meta.items():
        key, _ = site_birim(host, meta)
        title = normalize_for_matching(info.get("baslik") or "")
        if key in ("tum", "enstitu") or not title or "kocaeli universitesi" in title or "fakultesi" in title \
                or "yuksekokul" in title:
            continue
        candidates[" ".join(title.replace("-", " ").replace(",", " ").replace(" bolumu", "").split())].add(key)
    mapping = {}
    for name, keys in candidates.items():
        if (len(keys) == 1 and name not in AMBIGUOUS and len(name) >= 6
                and not re.search(r"[().]", name)):  # program notları / site sloganları alınmaz
            mapping[name] = next(iter(keys))
    mapping.update(ABBREVIATIONS)
    (WEB / "bolum_birim.json").write_text(json.dumps(dict(sorted(mapping.items())), ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    names = {name: display[name] for name in mapping if name in display}
    (WEB / "bolum_adlari.json").write_text(json.dumps(dict(sorted(names.items())), ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    dropped = sorted(n for n, k in candidates.items() if len(k) > 1)
    print(f"Harita: {len(mapping)} ad -> {WEB / 'bolum_birim.json'} | belirsiz (alınmadı): {len(dropped)}")
    print("  belirsiz örnekleri:", dropped[:15])
    return 0


if __name__ == "__main__":
    sys.exit(main())
