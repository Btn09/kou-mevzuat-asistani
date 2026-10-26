#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""veri/web/eksik_mevzuat_pdfleri.tsv'deki PDF'leri indirir (yalnızca *.kocaeli.edu.tr; kullanıcı onayıyla).

- "http://https://..." gibi bozuk yazılmış adresler düzeltilir; file:// ve KOÜ dışı adresler atlanır.
- Dosya adı: filePath parametresindeki ya da yoldaki ad; zaten indirilmişse atlanır (yeniden çalıştırılabilir).
- İçeriği PDF olmayan yanıtlar (hata sayfası vb.) kaydedilmez. İstekler arasında bekleme var.
Çıktı: veri/web/pdf/<dosya>.pdf, veri/web/pdf_indirme.tsv (adres, dosya, durum)
"""

from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from web_tara import session

ROOT = Path(__file__).resolve().parent.parent
LIST = ROOT / "veri" / "web" / "eksik_mevzuat_pdfleri.tsv"
OUT = ROOT / "veri" / "web" / "pdf"
DELAY = 0.5


def clean_url(url: str) -> str | None:
    url = re.sub(r"^https?://(https?://)", r"\1", url.strip())
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc.endswith("kocaeli.edu.tr"):
        return None
    return url.replace("http://", "https://", 1)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    http = session()
    rows = list(csv.DictReader(LIST.open(encoding="utf-8"), delimiter="\t"))
    log = []
    for index, row in enumerate(rows, start=1):
        url = clean_url(row["pdf"])
        name = re.sub(r"[^\w.\-]+", "_", row["dosya"])[:150]
        target = OUT / name
        if url is None:
            log.append((row["pdf"], name, "atlandı (KOÜ dışı/bozuk adres)"))
            continue
        if target.exists():
            log.append((url, name, "zaten var"))
            continue
        try:
            response = http.get(url, timeout=40)
            time.sleep(DELAY)
        except Exception as exc:
            log.append((url, name, f"hata: {str(exc)[:60]}"))
            continue
        if response.status_code != 200 or not response.content.startswith(b"%PDF"):
            log.append((url, name, f"PDF değil ({response.status_code})"))
            continue
        target.write_bytes(response.content)
        log.append((url, name, f"indirildi ({len(response.content) // 1024} KB)"))
        if index % 25 == 0:
            print(f"  {index}/{len(rows)}", flush=True)
    with (ROOT / "veri" / "web" / "pdf_indirme.tsv").open("w", encoding="utf-8") as output:
        output.write("adres\tdosya\tdurum\n")
        output.writelines("\t".join(item) + "\n" for item in log)
    counts: dict[str, int] = {}
    for _, _, status in log:
        key = status.split(" (")[0].split(":")[0]
        counts[key] = counts.get(key, 0) + 1
    size = sum(p.stat().st_size for p in OUT.glob("*.pdf")) // (1024 * 1024)
    print(f"Sonuç: {counts}, klasör {size} MB -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
