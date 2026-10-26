#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ücret/tutar tablosu içeren web PDF'lerinin tablolarını satır satır okunur metne çevirir.

pymupdf düz metni çok sütunlu tablolarda sütun başlıklarını kaybediyor ("₺1.438,00 | ₺5.674,00 ..." hangi
ücret olduğu belli değil). Burada her veri satırı kendi sütun başlıklarıyla yazılır:
  "Mühendislik Fakültesi — İkinci Öğretim ve Uzaktan Öğretim Öğrencilerinin Öğrenim Ücretleri / Güz YY Taksiti: ₺5.674,00; ..."
Tablo bölümü metin dosyasının BAŞINA (__TABLO__ ... __TABLO_SON__) eklenir; yeniden çalıştırınca yenilenir.
Hedef: veri/web_tam/metin/*.pdf.txt içinde para tutarı (₺ / TL) geçen belgeler.

Kullanım:  python -X utf8 tablo_metin.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXTS = ROOT / "veri" / "web_tam" / "metin"
SOURCES = [ROOT / "veri" / "web_tam" / "dosyalar", ROOT / "veri" / "web" / "pdf"]
MONEY_RE = re.compile(r"₺\s?\d|\d[\d.]*,\d{2}\s*(?:TL|₺)|\d{1,3}(?:\.\d{3})+\s*TL")
VALUE_RE = re.compile(r"\d")
NOISE_RE = re.compile(r"^\S{0,6}\s?(?:Evrak Tarih )?ve Sayısı:\s*[\d.]+-?E?\.?\d+\s*")
START, END = "__TABLO__", "__TABLO_SON__"
# Akademik takvim tabloları: "tarih | olay | | tarih" (güz / bahar). Düz metinde sütunlar karışıyor, model
# "final sınavları ne zaman?" sorusunu cevaplayamıyordu (18.10).
MONTH = r"(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)"
DATE_RE = re.compile(rf"\d{{1,2}}\s*(?:[-–]\s*\d{{1,2}}\s*)*(?:{MONTH}\s*)?(?:[-–]\s*\d{{1,2}}\s*)?{MONTH}\s+20\d\d", re.IGNORECASE)
MAX_LABEL = 110


def clean(cell: str | None) -> str:
    return " ".join((cell or "").split())


def header_labels(rows: list[list[str]]) -> tuple[str, list[str]]:
    """Başlık satırlarından (tablo adı, sütun etiketleri). Gruplu başlıklar sağa doğru doldurulur;
    '*' ile başlayan grup hücresi (dipnot) soldaki grubun devamı sayılır. Etiket = kısaltılmış grup / alt başlık."""
    title, parts = "", [[] for _ in rows[0]]
    for row_number, row in enumerate(rows):
        last = row_number == len(rows) - 1
        filled = [c for c in row if c]
        if len(filled) == 1 and row[0] and not any(parts[1:]):
            title = title or NOISE_RE.sub("", row[0]).strip()
            continue
        group = ""
        for index, cell in enumerate(row):
            if index == 0:
                continue
            if cell and not cell.startswith("*"):
                group = cell
            if cell.startswith("*") and last:
                text = cell.lstrip("*")
            else:
                text = cell if (cell and not cell.startswith("*")) else group
            if text and text not in parts[index]:
                parts[index].append(text)
    labels = []
    for column in parts:  # (grup başlığı, alt başlık)
        if len(column) > 1:
            head = " / ".join(column[:-1])
            head = head if len(head) <= MAX_LABEL else head[:MAX_LABEL].rsplit(" ", 1)[0] + "…"
            labels.append((head, column[-1]))
        else:
            labels.append(("", column[0][:MAX_LABEL] if column else ""))
    return title, labels


def table_lines(table) -> list[str]:
    rows = [[clean(c) for c in row] for row in table.extract()]
    rows = [r for r in rows if any(r)]
    if len(rows) < 3 or len(rows[0]) < 3:
        return []
    first_data = next((i for i, r in enumerate(rows) if sum(bool(VALUE_RE.search(c)) for c in r[1:]) >= 2
                       and not re.search(r"(?i)yıl|öğretim yılı", " ".join(r[1:]))), None)
    if not first_data:
        return []
    title, labels = header_labels(rows[:first_data])
    out = [f"[Tablo] {title}" if title else "[Tablo]"]
    current = ""
    context = f" ({title[:90]})" if title else ""
    for row in rows[first_data:]:
        label = row[0] or current
        current = label
        # Her satır + sütun grubu ayrı, kendi başına anlaşılır bir satır: uzun tek satırda model ücret
        # türlerini karıştırıyor ya da rakam vermiyordu.
        groups: dict[str, list[str]] = {}
        for i, cell in enumerate(row[1:], start=1):
            if cell:
                head, sub = labels[i]
                groups.setdefault(head, []).append(f"{sub or f'Sütun {i + 1}'}: {cell}")
        for head, values in groups.items():
            out.append(f"{label} — {head}{context}: " + "; ".join(values) if head
                       else f"{label}{context} — " + "; ".join(values))
    return out if len(out) > 2 else []


def source_of(text_path: Path) -> Path | None:
    name = text_path.name[:-4]
    return next((s / name for s in SOURCES if (s / name).exists()), None)


def calendar_lines(table) -> list[str]:
    """Takvim tablosu: her olay için "OLAY (tablo adı) — Güz yarıyılı: tarih; Bahar yarıyılı: tarih"."""
    rows = [[clean(c) for c in row] for row in table.extract()]
    rows = [r for r in rows if any(r)]
    dated = [r for r in rows if any(DATE_RE.search(c) for c in r)]
    if len(dated) < 5:
        return []
    title = next((NOISE_RE.sub("", c).lstrip("¹²³ ") for r in rows for c in r
                  if c and re.search(r"(?i)takvim", c)), "")
    out = [f"[Tablo] {title}" if title else "[Tablo] Akademik takvim"]
    context = f" ({title[:80]})" if title else ""
    for row in rows:
        dates = [(i, c) for i, c in enumerate(row) if DATE_RE.search(c)]
        events = [c for c in row if c and not DATE_RE.search(c) and not c.startswith("http")
                  and re.search(r"[A-Za-zÇĞİÖŞÜçğıöşü]{3}", c)]
        if not dates or not events:
            continue
        links = [c for c in row if c.startswith("http")]
        if len(dates) >= 2:  # sol sütun güz, sağ sütun bahar
            when = f"Güz yarıyılı: {dates[0][1]}; Bahar yarıyılı: {dates[-1][1]}"
        else:
            when = f"Tarih: {dates[0][1]}"
        out.append(f"{events[0]}{context} — {when}" + (f"; Bilgi: {links[0]}" if links else ""))
    return out if len(out) > 3 else []


def drop_raw_rows(body: str, pages: set[int]) -> str:
    """Tablosu çevrilen sayfalarda düz metnin tutar satırlarını (başlıksız "₺1.438,00 ₺5.674,00 ...") çıkarır:
    model başlıksız sütundan yanlış ücreti okuyordu (Mühendislik İÖ sorusunda uluslararası öğrenci ücreti).
    Dipnot gibi en fazla bir tutar geçen satırlar kalır."""
    parts = re.split(r"(__PAGE_\d+__)", body)
    out, current = [], 0
    for part in parts:
        marker = re.fullmatch(r"__PAGE_(\d+)__", part)
        if marker:
            current = int(marker.group(1))
        elif current in pages:
            part = "\n".join(l for l in part.split("\n")
                             if len(MONEY_RE.findall(l)) < 2 and len(DATE_RE.findall(l)) < 2)
        out.append(part)
    return "".join(out)


def process(text_path: Path) -> int:
    import pymupdf

    raw = text_path.read_text(encoding="utf-8", errors="ignore")
    body = raw.split(END, 1)[1].lstrip("\n") if raw.startswith(START) else raw
    is_calendar = bool(re.search(r"(?i)takvim", text_path.name + raw[:3000]))
    if not MONEY_RE.search(raw) and not is_calendar:  # ham tutar satırları önceki çalıştırmada silinmiş olabilir
        return 0
    source = source_of(text_path)
    if source is None:
        return 0
    lines: list[str] = []
    table_pages: set[int] = set()
    try:
        with pymupdf.open(source) as pdf:
            for number, page in enumerate(pdf, start=1):
                page_text = page.get_text() or ""
                calendar_page = is_calendar and len(DATE_RE.findall(page_text)) >= 5
                if not MONEY_RE.search(page_text) and not calendar_page:
                    continue
                for table in page.find_tables().tables:
                    found = (calendar_lines(table) if calendar_page else []) or table_lines(table)
                    if found:
                        lines += found
                        table_pages.add(number)
    except Exception:
        return 0
    if table_pages:
        body = drop_raw_rows(body, table_pages)
    if not lines:
        if raw.startswith(START):
            text_path.write_text(body, encoding="utf-8")
        return 0
    text_path.write_text(f"{START}\n" + "\n".join(lines) + f"\n{END}\n{body}", encoding="utf-8")
    return len(lines)


def main() -> int:
    paths = sorted(TEXTS.glob("*.pdf.txt"))
    done = rows = 0
    for number, path in enumerate(paths, start=1):
        count = process(path)
        if count:
            done += 1
            rows += count
        if number % 1000 == 0:
            print(f"  {number}/{len(paths)} (tablolu {done})", flush=True)
    print(f"Tablolu belge: {done}, satır: {rows}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
