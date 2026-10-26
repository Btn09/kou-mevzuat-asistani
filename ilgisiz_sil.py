#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ilgisiz_bul.py + gözden geçirme sonucu (veri/web/ilgisiz_karar.json) silinecek belgeleri uygular.

- veri/web/haric.json: silinen belge kimlikleri (tam_katalog.py bunları bir daha kataloğa almaz).
- Ham dosyalar (metin + indirilen asıl dosya / sayfa / duyuru) veri/silinen/ altına TAŞINIR (geri alınabilir).
- veri/silinen/RAPOR.md: silinen ve korunan belgelerin kategorili listesi.
Sonrasında: tam_katalog.py -> belge_isle.py --artimli (kopya klasörde) -> klasör değişimi.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

VERI = Path(__file__).resolve().parent.parent / "veri"
TRASH = VERI / "silinen"
LABELS = {"tibbi_akademik_icerik": "Tıbbi / akademik içerik (vaka sunumu, ders notu, tez özeti, lab föyü)",
          "is_ilani": "Şirket iş / staj ilanı", "etkinlik_tanitim": "KOÜ dışı etkinlik, kongre, yarışma tanıtımı",
          "ticari_urun": "Ticari ürün / yazılım / veritabanı kılavuzu", "haber_dergi": "Dergi, gazete, haber yazısı",
          "baska_kurum": "Başka kurum / şirket / vakıf bursu ve duyurusu"}


def move(relative: str) -> int:
    source = VERI / relative
    if not source.exists():
        return 0
    target = TRASH / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    return 1


def main() -> int:
    decision = json.loads((VERI / "web" / "ilgisiz_karar.json").read_text(encoding="utf-8"))
    delete = decision["sil"]
    haric_path = VERI / "web" / "haric.json"
    haric = json.loads(haric_path.read_text(encoding="utf-8")) if haric_path.exists() else []
    known = {h["ad"] for h in haric}
    haric += [{"ad": d["ad"], "kisa": d["kisa"], "kategori": d["kategori"], "sebep": d.get("sebep", "")}
              for d in delete if d["ad"] not in known]
    haric_path.write_text(json.dumps(haric, ensure_ascii=False, indent=1), encoding="utf-8")
    moved = 0
    for item in delete:
        moved += move(item["dosya"])
        if item["dosya"].startswith("web_tam/metin/"):  # metnin çıkarıldığı asıl dosya (PDF/DOCX)
            name = Path(item["dosya"]).name[:-4]
            for folder in ("web_tam/dosyalar", "web/pdf"):
                moved += move(f"{folder}/{name}")
    groups = defaultdict(list)
    for item in delete:
        groups[item["kategori"]].append(item)
    lines = ["# KOÜ ile ilgisiz bulunup silinen belgeler", "",
             f"Silinen: {len(delete)} belge (dosyalar bu klasörde; kalıcı silmek için klasörü silebilirsiniz).", ""]
    for category, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        lines += [f"## {LABELS.get(category, category)} — {len(items)}", ""]
        lines += [f"- {i['kisa']}" for i in sorted(items, key=lambda i: i["kisa"])] + [""]
    lines += ["# Korunanlar (KOÜ ile ilgili sayıldı)", ""]
    for name, items in decision["koru"].items():
        lines += [f"## {name} — {len(items)}", ""] + [f"- {i['kisa']}" for i in items] + [""]
    TRASH.mkdir(exist_ok=True)
    (TRASH / "RAPOR.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Silinen belge: {len(delete)}, taşınan dosya: {moved}, haric.json: {len(haric)} -> {TRASH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
