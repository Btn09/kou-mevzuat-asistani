#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""web_tara_tam.py'nin bulduğu PDF ve Word dosyalarını indirir (kullanıcı onayıyla: "bütün verileri al").

- Yalnızca *.kocaeli.edu.tr; pdf/doc/docx (Excel, PowerPoint, zip, resim alınmaz).
- Elimizde olanlar (yeni_pdfler, new_veri, veri/web/pdf) ve aynı adlı dosyalar tekrar indirilmez.
- 4 paralel bağlantı, dosya başına en fazla 30 MB; yeniden çalıştırılabilir (var olan atlanır).
Çıktı: veri/web_tam/dosyalar/<ad>, veri/web_tam/dosya_indirme.tsv (adres, dosya, kaynak sayfa, bağlantı metni, durum)
"""

from __future__ import annotations

import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from web_tara import session

ROOT = Path(__file__).resolve().parent.parent
LINKS = ROOT / "veri" / "web_tam" / "dosya_baglantilari.tsv"
OUT = ROOT / "veri" / "web_tam" / "dosyalar"
KEEP_EXT = {"pdf", "doc", "docx"}
MAX_BYTES = 30 * 1024 * 1024
WORKERS = 4
_local = threading.local()


def http():
    if not hasattr(_local, "s"):
        _local.s = session()
    return _local.s


def file_name(url: str) -> str:
    query = parse_qs(urlparse(url).query).get("filePath", [""])[0]
    return unquote(Path(query or urlparse(url).path).name).lower()


def clean_url(url: str) -> str | None:
    url = re.sub(r"^https?://(https?://)", r"\1", url.strip())
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc.endswith("kocaeli.edu.tr"):
        return None
    return url


def download(item: tuple[str, str, str, str]) -> tuple[str, str, str, str, str]:
    url, name, label, page = item
    target = OUT / re.sub(r"[^\w.\-]+", "_", name)[:150]
    if target.exists():
        return url, target.name, page, label, "zaten var"
    try:
        time.sleep(0.3)
        with http().get(url, timeout=60, stream=True) as response:
            if response.status_code != 200:
                return url, target.name, page, label, f"hata {response.status_code}"
            data = b""
            for chunk in response.iter_content(65536):
                data += chunk
                if len(data) > MAX_BYTES:
                    return url, target.name, page, label, "çok büyük"
    except Exception as exc:
        return url, target.name, page, label, f"hata {str(exc)[:40]}"
    if not (data.startswith(b"%PDF") or data.startswith(b"PK") or data.startswith(b"\xd0\xcf\x11\xe0")):
        return url, target.name, page, label, "dosya değil"
    target.write_bytes(data)
    return url, target.name, page, label, f"indirildi {len(data) // 1024} KB"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    have = {p.name.lower() for d in ("yeni_pdfler", "new_veri", "veri/web/pdf") for p in (ROOT / d).rglob("*") if p.is_file()}
    strip = lambda n: re.sub(r"^\d{8}_[0-9a-f]{5}_", "", n)
    have |= {strip(n) for n in have}
    items, seen = [], set()
    for line in LINKS.open(encoding="utf-8").read().splitlines()[1:]:
        parts = line.split("\t")
        url = clean_url(parts[0])
        if not url:
            continue
        name = file_name(url)
        ext = name.rsplit(".", 1)[-1] if "." in name else ""
        if ext not in KEEP_EXT or name in seen or name in have or strip(name) in have:
            continue
        seen.add(name)
        items.append((url, name, parts[1] if len(parts) > 1 else "", parts[-1] if len(parts) > 2 else ""))
    print(f"İndirilecek: {len(items)}", flush=True)
    log, start = [], time.time()
    with ThreadPoolExecutor(WORKERS) as pool:
        for index, result in enumerate(pool.map(download, items), start=1):
            log.append(result)
            if index % 250 == 0:
                rate = (time.time() - start) / index
                print(f"  {index}/{len(items)} (kalan ~{rate * (len(items) - index) / 60:.0f} dk)", flush=True)
    with (OUT.parent / "dosya_indirme.tsv").open("w", encoding="utf-8") as output:
        output.write("adres\tdosya\tsayfa\tbağlantı metni\tdurum\n")
        output.writelines("\t".join(row) + "\n" for row in log)
    counts: dict[str, int] = {}
    for row in log:
        key = row[4].split(" ")[0]
        counts[key] = counts.get(key, 0) + 1
    size = sum(p.stat().st_size for p in OUT.iterdir()) // (1024 * 1024)
    print(f"Sonuç: {counts}, {size} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
