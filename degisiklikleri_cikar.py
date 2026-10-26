#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Değişiklik yönetmeliklerindeki yeni metinleri PDF'ten çıkarıp veri/degisiklikler.json'a yazar.

Elimizde değişiklikleri işlenmiş güncel metin olmadığı için değişiklikler ana metne işlem işlem
uygulanır (bkz. belge_isle.py). Yeni metinler elle yazılmaz; değişiklik PDF'indeki tırnak
içindeki (“...”) bloklardan alınır. Hangi bloğun hangi maddeye gittiği aşağıda elle tanımlıdır.

Kullanım:
    python -X utf8 degisiklikleri_cikar.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pymupdf

VERI = Path(__file__).resolve().parent.parent / "veri"
OUT = VERI / "degisiklikler.json"


def quoted_blocks(pdf_path: Path) -> list[str]:
    with pymupdf.open(pdf_path) as pdf:
        text = "\n".join(page.get_text("text", sort=True) for page in pdf)
    return [clean(block) for block in re.findall(r"“(.*?)”", text, flags=re.S)]


def clean(block: str) -> str:
    """Fiziksel satır sonlarını birleştirir; fıkra '(n)' ve bent 'a)' başlarını yeni satırda tutar."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in block.split("\n")]
    out: list[str] = []
    for line in filter(None, lines):
        starts_item = re.match(r"^(\(\d+\)|[a-zçğıöşü]\)|MADDE \d+)", line)
        if out and not starts_item:
            out[-1] += " " + line
        else:
            out.append(line)
    return "\n".join(out)


def split_fikralar(text: str) -> dict[str, str]:
    """'(2) ...\n(3) ...' -> {'2': '...', '3': '...'} (bentler fıkraya dahil)."""
    parts = re.split(r"(?m)^\((\d+)\)\s*", text)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def main() -> int:
    onlisans = quoted_blocks(VERI / "onlisans_lisans/kou_onlisans_lisans_yonetmelik_degisiklik_2023.pdf")
    lisansustu = quoted_blocks(VERI / "lisansustu/kou_lisansustu_yonetmelik_degisiklik_2024.pdf")
    konservatuvar = quoted_blocks(VERI / "fakulteye_ozel/kou_devlet_konservatuvari_yonetmelik_degisiklik_2022.pdf")
    assert len(onlisans) == 3 and len(lisansustu) == 1 and len(konservatuvar) == 1, "beklenmeyen blok sayısı"

    madde_39 = re.sub(r"^MADDE 39-\s*", "", onlisans[2])
    madde_53 = re.sub(r"^MADDE 53-\s*", "", konservatuvar[0])
    changes = [
        {
            "belge": "kou_onlisans_lisans_egitim_ogretim_yonetmeligi_2019",
            "degisiklik_belgesi": "kou_onlisans_lisans_yonetmelik_degisiklik_2023",
            "etiket": "RG 18/9/2023, 32313 ile değişik; Kocaeli 1. İdare Mah. 22/1/2024 kısmi iptal",
            "islemler": [
                {"tur": "bolum_basligi", "bolum": "ALTINCI BÖLÜM", "yeni": onlisans[0]},
                {"tur": "fikra_degistir", "madde": "MADDE 38", "fikralar": split_fikralar(onlisans[1])},
                {"tur": "madde_degistir", "madde": "MADDE 39", "fikralar": split_fikralar(madde_39)},
            ],
        },
        {
            "belge": "kou_lisansustu_egitim_ogretim_yonetmeligi_2023",
            "degisiklik_belgesi": "kou_lisansustu_yonetmelik_degisiklik_2024",
            "etiket": "RG 28/7/2024, 32615 ile değişik",
            "islemler": [
                {"tur": "cumle_degistir", "madde": "MADDE 29", "fikra": "1",
                 "eski": "Jüri üyeleri, biri kurum dışından olmak üzere, danışman dâhil en az üç asıl ve birisi "
                         "Üniversite içerisinden diğeri ise başka bir üniversiteden olmak üzere iki yedek öğretim "
                         "üyesinden oluşur.",
                 "yeni": lisansustu[0]},
            ],
        },
        {
            "belge": "kou_devlet_konservatuvari_lisans_egitim_sinav_yonetmeligi",
            "degisiklik_belgesi": "kou_devlet_konservatuvari_yonetmelik_degisiklik_2022",
            "etiket": "RG 30/10/2022, 31998 ile değişik",
            "islemler": [
                {"tur": "madde_degistir", "madde": "MADDE 53", "fikralar": split_fikralar(madde_53)},
                {"tur": "madde_kaldir", "maddeler": ["MADDE 54", "MADDE 55", "MADDE 56", "MADDE 59"]},
            ],
        },
    ]
    OUT.write_text(json.dumps(changes, ensure_ascii=False, indent=1), encoding="utf-8")
    for change in changes:
        for op in change["islemler"]:
            detail = op.get("madde") or op.get("bolum") or ",".join(op.get("maddeler", []))
            count = len(op.get("fikralar", {})) or ""
            print(f"{change['belge'][:45]:45} {op['tur']:15} {detail} {f'({count} fıkra)' if count else ''}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
