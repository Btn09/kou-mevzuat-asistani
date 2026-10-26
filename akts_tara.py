#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KOÜ AKTS Bilgi Paketi'ni (ects.kocaeli.edu.tr) toplar: tüm programlar ve dersleri.

Akış (sitenin kendi AJAX çağrıları):
  AkademikDuzey.cfm (Islem: 1 lisans, 2 önlisans, 3 yüksek lisans, 4 doktora) -> fakülte listesi
  AkademikDuzey.cfm + Fakulteid + bolumKontrol=4                         -> bölüm/program listesi
  BolumBilgiDetay.cfm?Bolumid&Dilid&DipTurid&Yazdir=1                    -> program bilgisi (yazdırma görünümü)
  BolumBilgiDetay.cfm?Bolumid&Dilid&DipTurid                             -> ders listesi (a#DersAyrinti)
  POST /KOUBS/AKTS/BolumBilgiDetay.cfm (Katalogid, EgitimYili, Bolumid, Dilid) -> ders ayrıntısı

Kaydedilen (veri/web/akts/):
  program__<bolumid>_<dip>.txt  : program bilgisi (amaç, kabul, mezuniyet koşulları, staj, not sistemi, ders listesi)
  dersler__<bolumid>_<dip>.txt  : her ders bir bölüm: kod, tür, yıl/dönem, AKTS, dil, kazanımlar, ön koşul,
                                  içerik, haftalık izlence, değerlendirme. Öğretim elemanı adları ve program
                                  yeterlilik matrisi / iş yükü tablosu alınmaz.
  programlar.json               : dizin (birim, program, düzey, dosyalar)
Yeniden çalıştırmada var olan dosyalar atlanır. 4 paralel bağlantı, istekler arası bekleme.

Kullanım:  python -X utf8 akts_tara.py
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from web_tara import html_text, session

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "veri" / "web" / "akts"
BASE = "https://ects.kocaeli.edu.tr/"
LEVELS = {"1": "Lisans", "2": "Önlisans", "3": "Yüksek Lisans", "4": "Doktora"}
WORKERS = 4
DELAY = 0.25
KEEP_SECTIONS = ["DERSİN ÖĞRENME KAZANIMLARI", "ÖN KOŞULLAR, DİĞER KOŞULLAR", "ÖNERİLEN DESTEKLEYİCİ DERSLER",
                 "DERSİN İÇERİĞİ", "HAFTALIK DERS İZLENCESİ", "DEĞERLENDİRME YÖNETMİ VE ÖLÇÜTLERİ"]
DROP_SECTIONS = ["DERS ÖĞRETİM ELEMANLARI", "PROGRAM YETERLİLİĞİ - ÖĞRENME KAZANIMLARI İLİŞKİSİ",
                 "ÖNERİLEN/İSTENEN DERS KAYNAKLARI", "AKTS İŞ YÜKÜ MOTORU"]
GENERAL_DROP = {"Ders Koordinatörü"}
_local = threading.local()


def http():
    if not hasattr(_local, "s"):
        _local.s = session()
    return _local.s


def get(url: str, **params) -> str:
    for attempt in range(4):
        try:
            time.sleep(DELAY)
            r = http().get(url, params=params, timeout=90)
            if r.status_code == 200:
                return r.text
        except Exception:
            time.sleep(3 * (attempt + 1))
    return ""


def post(url: str, data: dict) -> str:
    for attempt in range(4):
        try:
            time.sleep(DELAY)
            r = http().post(url, data=data, timeout=90)
            if r.status_code == 200:
                return r.text
        except Exception:
            time.sleep(3 * (attempt + 1))
    return ""


def attrs(tag: str) -> dict[str, str]:
    return dict(re.findall(r'(\w+)="([^"]*)"', tag))


def programs() -> list[dict]:
    found = []
    for islem, level in LEVELS.items():
        html = post(BASE + "AkademikDuzey.cfm", {"Islem": islem, "Dilid": "0"})
        for tag in re.findall(r'<a[^>]*id="FakulteListe"[^>]*>', html):
            fac = attrs(tag)
            dil, isl = (fac.get("data", "0[;]" + islem).split("[;]") + [islem])[:2]
            listing = post(BASE + "AkademikDuzey.cfm", {"Fakulteid": fac["name"], "Dilid": dil, "Islem": isl,
                                                         "bolumKontrol": "4"})
            for btag in re.findall(r'<a[^>]*id="BolumListe"[^>]*>', listing):
                b = attrs(btag)
                bdil, dip = (b.get("data", "0[;]1").split("[;]") + ["1"])[:2]
                found.append({"duzey": level, "birim": fac.get("title", ""), "program": b.get("title", ""),
                              "bolumid": b["name"], "dilid": bdil, "dipturid": dip})
        print(f"  {level}: toplam {len(found)} program", flush=True)
    return found


def compact_course(text: str) -> str:
    """Ders ayrıntısından tutulacak bölümler; genel bilgi alanları "Alan: değer" satırlarına çevrilir."""
    lines = [l for l in text.split("\n") if l.strip()]
    out, section, general = [], "", []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line in KEEP_SECTIONS or line in DROP_SECTIONS or line == "GENEL BİLGİ":
            section = line
            if line in KEEP_SECTIONS:
                out.append(f"\n{line}")
            i += 1
            continue
        if section == "GENEL BİLGİ" and i + 1 < len(lines):
            if line not in GENERAL_DROP:
                general.append(f"{line}: {lines[i + 1].strip()}")
            i += 2
            continue
        if section in KEEP_SECTIONS:
            out.append(line)
        i += 1
    return "\n".join(general) + "\n" + "\n".join(out)


def collect(program: dict) -> dict:
    key = f"{program['bolumid']}_{program['dipturid']}"
    pfile, cfile = OUT / f"program__{key}.txt", OUT / f"dersler__{key}.txt"
    header = (f"KAYNAK: {BASE}BolumBilgiDetay.cfm?Bolumid={program['bolumid']}&DipTurid={program['dipturid']}\n"
              f"BAŞLIK: {program['birim'].title()} – {program['program'].title()} ({program['duzey']})\n"
              f"ERİŞİM: {date.today().isoformat()}\n\n")
    if not pfile.exists():
        text = html_text(get(BASE + "BolumBilgiDetay.cfm", Bolumid=program["bolumid"], Dilid=program["dilid"],
                             DipTurid=program["dipturid"], Yazdir=1))
        if text:
            pfile.write_text(header + text + "\n", encoding="utf-8")
    courses = 0
    if not cfile.exists():
        listing = get(BASE + "BolumBilgiDetay.cfm", Bolumid=program["bolumid"], Dilid=program["dilid"],
                      DipTurid=program["dipturid"])
        names = list(dict.fromkeys(re.findall(r'id="DersAyrinti"[^>]*name="([^"]+)"', listing)
                                   + re.findall(r'name="([^"]+)"[^>]*id="DersAyrinti"', listing)))
        blocks = []
        for name in names:
            p = name.split("[;]")
            if len(p) < 8:
                continue
            detail = html_text(post(BASE + "KOUBS/AKTS/BolumBilgiDetay.cfm",
                                    {"Katalogid": p[2], "Pencere": "1200", "EgitimYili": p[3], "Bolumid": p[7],
                                     "Dilid": p[6]}))
            if detail:
                blocks.append(f"{p[0]} {p[1]}".upper() + "\n\n" + compact_course(detail))
                courses += 1
        if blocks:
            cfile.write_text(header.replace("BAŞLIK: ", "BAŞLIK: Ders Bilgileri – ") + "\n\n".join(blocks) + "\n",
                             encoding="utf-8")
    return {**program, "program_dosya": pfile.name if pfile.exists() else "",
            "ders_dosya": cfile.name if cfile.exists() else "", "ders_sayisi": courses}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    plist = programs()
    print(f"Program: {len(plist)}", flush=True)
    done: list[dict] = []
    start = time.time()
    with ThreadPoolExecutor(WORKERS) as pool:
        for index, result in enumerate(pool.map(collect, plist), start=1):
            done.append(result)
            if index % 10 == 0:
                rate = (time.time() - start) / index
                print(f"  {index}/{len(plist)} ({(time.time() - start) / 60:.0f} dk, kalan ~{rate * (len(plist) - index) / 60:.0f} dk)",
                      flush=True)
    (OUT / "programlar.json").write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Bitti: {len(done)} program, {sum(1 for d in done if d['ders_dosya'])} ders dosyası")
    return 0


if __name__ == "__main__":
    sys.exit(main())
