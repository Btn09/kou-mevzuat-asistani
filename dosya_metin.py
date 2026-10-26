#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""İndirilen web dosyalarından (PDF, DOCX, DOC) metin çıkarır.

- PDF: pymupdf, sayfa işaretleriyle (__PAGE_n__) — belge_isle.pdf_raw_text ile aynı biçim.
- DOCX: python-docx (paragraflar + tablolar, satır satır).
- DOC: Microsoft Word otomasyonu (win32com; tek iş parçacığı).
Kaynaklar: veri/web_tam/dosyalar/*, veri/web/pdf/*. Çıktı: veri/web_tam/metin/<dosya>.txt (var olan atlanır).
Taranmış (metinsiz) PDF'ler boş çıkar; sonraki adımda elenir.

Kullanım:  python -X utf8 dosya_metin.py
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [ROOT / "veri" / "web_tam" / "dosyalar", ROOT / "veri" / "web" / "pdf"]
OUT = ROOT / "veri" / "web_tam" / "metin"


def pdf_text(path: Path) -> str:
    import pymupdf
    pieces = []
    with pymupdf.open(path) as pdf:
        for number, page in enumerate(pdf, start=1):
            pieces.append(f"\n__PAGE_{number}__\n" + (page.get_text("text", sort=True) or ""))
    return "".join(pieces)


def docx_text(path: Path) -> str:
    import docx
    document = docx.Document(str(path))
    lines = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            cells = list(dict.fromkeys(c.text.strip() for c in row.cells if c.text.strip()))
            if cells:
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def extract(path: Path) -> tuple[str, int]:
    target = OUT / f"{path.name}.txt"
    if target.exists():
        return path.name, -1
    try:
        text = pdf_text(path) if path.suffix.lower() == ".pdf" else docx_text(path)
    except Exception:
        text = ""
    target.write_text(text, encoding="utf-8")
    return path.name, len(text)


def doc_files_with_word(paths: list[Path]) -> int:
    """Eski .doc: Word ile açılıp metni okunur."""
    import pythoncom
    import win32com.client
    pythoncom.CoInitialize()
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    done = 0
    try:
        for path in paths:
            target = OUT / f"{path.name}.txt"
            if target.exists():
                continue
            text = ""
            try:
                document = word.Documents.Open(str(path.resolve()), ReadOnly=True, ConfirmConversions=False,
                                               AddToRecentFiles=False, Visible=False)
                text = document.Content.Text.replace("\r", "\n")
                document.Close(False)
            except Exception:
                pass
            target.write_text(text, encoding="utf-8")
            done += 1
            if done % 100 == 0:
                print(f"  doc: {done}/{len(paths)}", flush=True)
    finally:
        word.Quit()
    return done


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = [p for source in SOURCES if source.exists() for p in source.iterdir() if p.is_file()]
    fast = [p for p in files if p.suffix.lower() in (".pdf", ".docx")]
    slow = [p for p in files if p.suffix.lower() == ".doc"]
    print(f"PDF/DOCX: {len(fast)}, DOC: {len(slow)}", flush=True)
    start, empty = time.time(), 0
    with ProcessPoolExecutor(6) as pool:
        for index, (name, length) in enumerate(pool.map(extract, fast, chunksize=8), start=1):
            empty += length == 0
            if index % 500 == 0:
                print(f"  {index}/{len(fast)} ({(time.time() - start) / 60:.0f} dk)", flush=True)
    doc_files_with_word(slow)
    texts = list(OUT.glob("*.txt"))
    print(f"Bitti: {len(texts)} metin, boş (taranmış/okunamayan): {sum(1 for t in texts if t.stat().st_size < 50)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
