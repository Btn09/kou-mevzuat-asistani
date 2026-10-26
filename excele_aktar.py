#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""İşlenmiş veriyi incelemek için Excel'e aktarır: veri/islenmis/veri_tablosu.xlsx

Sayfalar: Belgeler (katalog), Maddeler (maddeler.jsonl), Parçalar (parcalar.jsonl).
Her sütunda filtre var, başlık satırı sabit.

Kullanım:
    python -X utf8 excele_aktar.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

VERI = Path(__file__).resolve().parent.parent / "veri"
OUT = VERI / "islenmis" / "veri_tablosu.xlsx"

FONT = Font(name="Arial", size=10)
HEADER_FONT = Font(name="Arial", size=10, bold=True, color="FFFFFF")
HEADER_FILL = PatternFill("solid", fgColor="0B6E4F")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def add_sheet(workbook: Workbook, title: str, columns: list[tuple[str, int]], rows: list[list]) -> None:
    """columns: (başlık, sütun genişliği). Geniş metin sütunları satır kaydırmalı."""
    sheet = workbook.create_sheet(title)
    sheet.append([name for name, _ in columns])
    for cell in sheet[1]:
        cell.font, cell.fill = HEADER_FONT, HEADER_FILL
        cell.alignment = Alignment(vertical="center")
    for row in rows:
        sheet.append(["" if value is None else value for value in row])
    for index, (_, width) in enumerate(columns, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
        wrap = width >= 60
        for cell in sheet.iter_cols(min_col=index, max_col=index, min_row=2).__next__():
            cell.font = FONT
            cell.alignment = Alignment(vertical="top", wrap_text=wrap)
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions


def main() -> int:
    catalog = json.loads((VERI / "belgeler.json").read_text(encoding="utf-8"))
    articles = read_jsonl(VERI / "islenmis" / "maddeler.jsonl")
    chunks = read_jsonl(VERI / "islenmis" / "parcalar.jsonl")
    workbook = Workbook()
    workbook.remove(workbook.active)

    add_sheet(workbook, "Belgeler", [
        ("belge_id", 45), ("belge_adi", 50), ("tur", 14), ("kurum", 14), ("hedef_kitle", 12), ("birim", 14),
        ("hiyerarsi", 10), ("yururluk_tarihi", 14), ("yururluk_bitis", 14), ("surum", 40), ("not", 70),
        ("kaynak_dosya", 50),
    ], [[e["ad"], e["kisa"], e["tur"], e["kurum"], e.get("hedef_kitle"), e.get("birim"), e.get("hiyerarsi"),
         e.get("yururluk_tarihi"), e.get("yururluk_bitis"), e.get("surum"), e.get("not"), e["kaynak_dosya"]]
        for e in catalog["kullanilan"]])

    add_sheet(workbook, "Dışarıda Kalanlar", [("kaynak_dosya", 60), ("belge", 60), ("gerekce", 90)],
              [[e["kaynak_dosya"], e.get("belge", ""), e["gerekce"]] for e in catalog["disarida_birakilan"]])

    add_sheet(workbook, "Maddeler", [
        ("baglam_id", 50), ("belge_adi", 40), ("madde_etiketi", 16), ("madde_basligi", 35), ("madde_parca_no", 9),
        ("hedef_kitle", 11), ("birim", 12), ("bolum_basligi", 35), ("sayfa", 7), ("atiflar", 30),
        ("son_degisiklik", 30), ("esanlamlilar", 35), ("metin", 100),
    ], [[a["id"], f["belge_adi"], f["madde_etiketi"], f["madde_basligi"], f["madde_parca_no"], f["hedef_kitle"],
         f["birim"], f["bolum_basligi"], f["sayfa"], ", ".join(x.split("__m")[-1] for x in f["atiflar"]),
         f["son_degisiklik"], f["esanlamlilar"], a["metin"]]
        for a in articles for f in [a["ozellikler"]]])

    fields = ["belge_id", "belge_adi", "kurum", "hiyerarsi", "hedef_kitle", "birim", "konu", "madde_id", "madde_no",
              "madde_parca_no", "fikra_no", "atiflar", "yururluk_tarihi", "yururluk_bitis", "parca_turu"]
    widths = {"belge_id": 30, "belge_adi": 35, "madde_id": 35, "atiflar": 25}
    add_sheet(workbook, "Parçalar", [("id", 40)] + [(name, widths.get(name, 12)) for name in fields] + [("metin", 100)],
              [[c["id"]] + [", ".join(c["ozellikler"][n]) if n == "atiflar" else c["ozellikler"][n] for n in fields]
               + [c["metin"]] for c in chunks])

    workbook.save(OUT)
    print(f"{OUT}  (belge {len(catalog['kullanilan'])}, madde {len(articles)}, parça {len(chunks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
