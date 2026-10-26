#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Belge kimliği -> orijinal bağlantı ve tarih etiketi (web arayüzünde kaynağın altında gösterilir).

Bağlantı: web belgelerinde kataloğun kaynak_url'si; arşiv PDF'lerinde (veri/belgeler.json) dosya adı KOÜ'nün dosya
bağlantıları listesinde (web_tam/dosya_baglantilari*.tsv) aranır.
Tarih: dosyalarda KOÜ'nün yükleme tarihi (dosya adı öneki), duyurularda yayın tarihi, Cumhurbaşkanı kararlarında
Resmî Gazete tarihi, sayfalarda siteden alındığı tarih, arşiv belgelerinde sürüm tarihi.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote

VERI = Path(__file__).resolve().parent.parent / "veri"
FILE_DATE_RE = re.compile(r"(?:^|_)(20\d\d)(\d\d)(\d\d)_[0-9a-f]{5}_")


def _date(year: str, month: str, day: str) -> str:
    return f"{day}.{month}.{year}"


def _iso(value: str) -> str:
    match = re.match(r"(\d{4})-(\d\d)-(\d\d)", value or "")
    return _date(*match.groups()) if match else ""


@lru_cache(maxsize=1)
def file_links() -> dict[str, str]:
    """Dosya adı (uzantısız, küçük harf) -> KOÜ dosya bağlantısı."""
    links: dict[str, str] = {}
    for name in ("dosya_baglantilari.tsv", "dosya_baglantilari_ek.tsv"):
        path = VERI / "web_tam" / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
            url = line.split("\t", 1)[0].strip()
            if not url.startswith("http"):
                continue
            stem = unquote(url.rsplit("/", 1)[-1].rsplit("=", 1)[-1]).rsplit(".", 1)[0].lower()
            links.setdefault(stem, url)
    return links


def _page_access(path: Path) -> str:
    try:
        with path.open(encoding="utf-8", errors="ignore") as handle:
            head = handle.read(400)
    except OSError:
        return ""
    match = re.search(r"ERİŞİM: (\d{4}-\d\d-\d\d)", head)
    return _iso(match.group(1)) if match else ""


@lru_cache(maxsize=1)
def table() -> dict[str, dict[str, str]]:
    info: dict[str, dict[str, str]] = {}
    for catalog in sorted((VERI / "web").glob("katalog_*.json")):
        for entry in json.loads(catalog.read_text(encoding="utf-8")):
            ad, url = entry["ad"], entry.get("kaynak_url", "")
            date = ""
            match = FILE_DATE_RE.search(ad)
            if ad.startswith("webf_") and match:
                date = f"KOÜ'de yayımlanma: {_date(*match.groups())}"
            elif ad.startswith("webd_") and entry.get("yururluk_tarihi"):
                date = f"Duyuru tarihi: {_iso(entry['yururluk_tarihi'])}"
            elif ad.startswith("mevzuat_"):
                rg = re.search(r"\((\d\d\.\d\d\.\d{4})\)", entry.get("kisa", ""))
                date = f"Resmî Gazete: {rg.group(1)}" if rg else "mevzuat.gov.tr güncel metin"
            elif entry.get("dosya"):
                access = _page_access(VERI / entry["dosya"])
                date = f"Siteden alındı: {access}" if access else ""
            info[ad] = {"url": url if url.startswith("http") else "", "tarih": date}
    links = file_links()
    # Yeniden adlandırılmış arşiv PDF'leri: içerik benzerliğiyle bulunan KOÜ kopyası (benzerlik >= 0.9 + ad eşleşmesi)
    matched_path = VERI / "web" / "arsiv_baglantilari.json"
    matched = json.loads(matched_path.read_text(encoding="utf-8")) if matched_path.exists() else {}
    for entry in json.loads((VERI / "belgeler.json").read_text(encoding="utf-8"))["kullanilan"]:
        source = (entry.get("kaynak_dosya") or "").rsplit(".", 1)[0].lower()
        url = links.get(source, "") or matched.get(entry["ad"], "")
        match = FILE_DATE_RE.search(source + "_")
        if entry.get("surum"):
            date = f"Sürüm: {entry['surum']}"
        elif match:
            date = f"KOÜ'de yayımlanma: {_date(*match.groups())}"
        else:
            date = ""
        info[entry["ad"]] = {"url": url, "tarih": date}
    return info


def of(belge_id: str) -> dict[str, str]:
    if belge_id.startswith("aktsders_"):  # ders birimleri: AKTS program sayfası
        belge_id = "akts_" + belge_id.removeprefix("aktsders_").replace("dersler__", "program__")
    return table().get(belge_id, {"url": "", "tarih": ""})


if __name__ == "__main__":
    info = table()
    print(f"belge: {len(info)}, bağlantılı: {sum(bool(v['url']) for v in info.values())}, "
          f"tarihli: {sum(bool(v['tarih']) for v in info.values())}")
    for key in ("kou_mimarlik_tasarim_fakultesi_staj_yonergesi", "kou_teknoloji_fakultesi_ime_yonergesi",
                "mevzuat_1_5_657", "webf_20260915_7aa94_kurum_ici_gelen_evrak_guncelleme_harc_ucretleri_10a",
                "kanun_2547_yuksekogretim"):
        print(key, info.get(key))
