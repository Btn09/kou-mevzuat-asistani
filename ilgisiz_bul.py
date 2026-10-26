#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KOÜ ile ilgisi olmayan belgeleri bulur (kullanıcı istedi, 21.10).

1. aşama: katalogdaki her belgenin "Site – Başlık" adı + metninin başı qwen3:14b'ye 30'arlı gruplar halinde verilir;
   K (KOÜ ile ilgili) / D (ilgisiz) + kısa sebep kategorisi döner.
2. aşama: D işaretlilerin metni (ilk ~1500 karakter) ikinci kez, tek tek denetlenir (yanlış silmeyi önlemek için).
Çıktı: veri/web/ilgisiz_aday.json (önbellek, kaldığı yerden), veri/web/ilgisiz.json (kesinleşen liste).

Kullanım:  python -X utf8 ilgisiz_bul.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import ollama

VERI = Path(__file__).resolve().parent.parent / "veri"
MODEL = "qwen3:14b"
BATCH = 30
STAGE1 = VERI / "web" / "ilgisiz_aday.json"
FINAL = VERI / "web" / "ilgisiz.json"
CATEGORIES = ("is_ilani", "baska_kurum", "tibbi_akademik_icerik", "haber_dergi", "etkinlik_tanitim",
              "ticari_urun", "genel_bilgi", "diger")
RULE = (
    "Bir Kocaeli Üniversitesi (KOÜ) asistanının veri tabanındaki belgeleri ayıklıyoruz. Asistan KOÜ'nün kurallarını, "
    "süreçlerini, birimlerini, hizmetlerini, programlarını, takvimini, ücretlerini ve KOÜ öğrenci/personelini "
    "ilgilendiren duyuruları cevaplar.\n"
    "K = KOÜ ile ilgili: KOÜ'nün yönetmelik/yönerge/usul ve esasları, başvuru-kayıt-sınav-staj-harç-burs işlemleri, "
    "birim tanıtımı ve iletişim/görev bilgisi, akademik program ve ders bilgisi, akademik takvim, KOÜ'nün kendi "
    "duyuruları, formlar, KOÜ'de uygulanan ulusal mevzuat, KOÜ'nün kendi raporları.\n"
    "D = KOÜ ile ilgisiz: başka kurum/şirketin iş veya staj ilanı, başka kurumun duyurusu/çağrısı, genel tıbbi veya "
    "akademik ders materyali (vaka sunumu, ders notu, makale, tez özeti), dergi/bülten/haber yazısı, KOÜ dışı "
    "etkinlik/konferans/webinar tanıtımı, ticari ürün veya veritabanı tanıtımı, KOÜ'ye özgü olmayan genel bilgi notu.\n"
    "ÖNEMLİ: Tüm belgeler KOÜ sitelerinden alındı; başlıktaki site adı (\"Kocaeli Üniversitesi duyurusu\", "
    "\"Mezunlarla İlişkiler Ofisi\") belgeyi KOÜ ile ilgili YAPMAZ. İçeriğe bak: KOÜ'nün kendi işleyişi değilse D.\n"
    "Örnekler: \"Mezunlarla İlişkiler Ofisi duyurusu – THY kabin memuru alım ilanı\" -> D (is_ilani); "
    "\"Kütüphane – Clarivate Ekim Webinar Programı\" -> D (etkinlik_tanitim); \"Genç Açı'nın 86. sayısı yayımlandı\" "
    "-> D (haber_dergi); \"Tıp Fakültesi – Kalp Sesleri\" -> D (tibbi_akademik_icerik); \"Kocaeli Üniversitesi "
    "duyurusu – Sözleşmeli Bilişim Personeli Alım İlanı\" -> K (KOÜ'nün kendi alımı); \"Ziraat Fakültesi – Mazeret "
    "sınavı başvuruları\" -> K; \"Kütüphane – üyelik ve ödünç verme kuralları\" -> K; \"Tıp Fakültesi – Ders Kurulu / "
    "Staj programı\" -> K (KOÜ'nün kendi eğitim programı); \"KOÜ Türk Sanat Müziği Korosu çalışmaları\" -> K (KOÜ'nün "
    "kendi etkinliği); \"KOÜ Hastanesi duyurusu\" -> K. Yani KOÜ'nün KENDİ etkinliği, kulübü, korosu, hastanesi, "
    "programı K'dir; D yalnızca içeriği KOÜ ile ilgili olmayanlar.\n"
    "Emin değilsen K de."
)
SCHEMA = {"type": "object", "properties": {"sonuc": {"type": "array", "items": {"type": "object", "properties": {
    "no": {"type": "integer"}, "karar": {"type": "string", "enum": ["K", "D"]},
    "kategori": {"type": "string", "enum": list(CATEGORIES)}}, "required": ["no", "karar", "kategori"]}}},
    "required": ["sonuc"]}
ONE_SCHEMA = {"type": "object", "properties": {"karar": {"type": "string", "enum": ["K", "D"]},
                                              "kategori": {"type": "string", "enum": list(CATEGORIES)},
                                              "sebep": {"type": "string"}}, "required": ["karar", "kategori", "sebep"]}


def text_of(entry: dict, limit: int) -> str:
    try:
        raw = (VERI / entry["dosya"]).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    raw = re.sub(r"__TABLO__.*?__TABLO_SON__|__PAGE_\d+__|^(KAYNAK|BAŞLIK|ERİŞİM|YAYIN):.*$", " ", raw, flags=re.S | re.M)
    return " ".join(raw.split())[:limit]


def catalog() -> list[dict]:
    entries = []
    for path in sorted((VERI / "web").glob("katalog_*.json")):
        entries += json.loads(path.read_text(encoding="utf-8"))
    return [e for e in entries if not e["ad"].startswith(("akts_", "web_yapi", "mevzuat_"))]  # AKTS/yapı/mevzuat KOÜ'ye ait


def stage1(entries: list[dict]) -> dict[str, dict]:
    done = json.loads(STAGE1.read_text(encoding="utf-8")) if STAGE1.exists() else {}
    todo = [e for e in entries if e["ad"] not in done]
    print(f"1. aşama: {len(todo)} belge ({len(done)} önbellekte)", flush=True)
    start = time.time()
    for offset in range(0, len(todo), BATCH):
        batch = todo[offset:offset + BATCH]
        lines = "\n".join(f"{i}. {e['kisa'][:150]} || {text_of(e, 220)}" for i, e in enumerate(batch, start=1))
        prompt = f"{RULE}\n\nHer satır: numara. Site – Başlık || metnin başı\n\n{lines}\n\nHer numara için karar ver. JSON."
        try:
            reply = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}], format=SCHEMA, think=False,
                                options={"temperature": 0, "num_ctx": 8192, "num_predict": 3000})
            results = json.loads(reply["message"]["content"])["sonuc"]
        except Exception as exc:  # bu grup sonra tekrar denenir
            print(f"  grup hatası: {str(exc)[:80]}", flush=True)
            continue
        by_no = {r["no"]: r for r in results}
        for i, entry in enumerate(batch, start=1):
            r = by_no.get(i, {"karar": "K", "kategori": "diger"})
            done[entry["ad"]] = {"karar": r["karar"], "kategori": r["kategori"]}
        STAGE1.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
        if (offset // BATCH) % 20 == 0:
            elapsed = time.time() - start
            left = elapsed / (offset + len(batch)) * (len(todo) - offset - len(batch))
            print(f"  {offset + len(batch)}/{len(todo)} | {elapsed / 60:.0f} dk, kalan ~{left / 60:.0f} dk", flush=True)
    return done


def stage2(entries: list[dict], first: dict[str, dict]) -> list[dict]:
    by_ad = {e["ad"]: e for e in entries}
    flagged = [by_ad[a] for a, r in first.items() if r["karar"] == "D" and a in by_ad]
    print(f"2. aşama: {len(flagged)} aday tek tek denetleniyor", flush=True)
    final = []
    for index, entry in enumerate(flagged, start=1):
        prompt = (f"{RULE}\n\nBELGE: {entry['kisa']}\nMETİN (başı): {text_of(entry, 1500)}\n\n"
                  "Bu belge KOÜ ile ilgili mi (K) ilgisiz mi (D)? Kategori ve tek cümle sebep ver. JSON.")
        try:
            reply = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}], format=ONE_SCHEMA,
                                think=False, options={"temperature": 0, "num_ctx": 4096, "num_predict": 300})
            r = json.loads(reply["message"]["content"])
        except Exception:
            continue
        if r["karar"] == "D":
            final.append({"ad": entry["ad"], "kisa": entry["kisa"], "dosya": entry["dosya"],
                          "kategori": r["kategori"], "sebep": r["sebep"]})
        if index % 100 == 0:
            print(f"  {index}/{len(flagged)} -> {len(final)} ilgisiz", flush=True)
    FINAL.write_text(json.dumps(final, ensure_ascii=False, indent=1), encoding="utf-8")
    return final


def main() -> int:
    entries = catalog()
    first = stage1(entries)
    final = stage2(entries, first)
    counts: dict[str, int] = {}
    for item in final:
        counts[item["kategori"]] = counts.get(item["kategori"], 0) + 1
    print(f"İlgisiz: {len(final)} / {len(entries)} belge | {counts} -> {FINAL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
