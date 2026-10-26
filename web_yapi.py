#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KOÜ'nün akademik ve idari yapısını site menülerinden ayrıntılı belgeye çevirir.

Bu bilgi sayfa içeriklerinde değil menülerde durduğu için web_tara.py'de alınmıyordu ("Fen Edebiyat
Fakültesinde kaç bölüm var?" cevaplanamıyordu). Kaynak: api.kocaeli.edu.tr (Menu/GetAll, Page/GetBySlug).

1. Üniversite geneli (www menüsü): fakülteler, enstitüler, yüksekokullar, MYO'lar, rektörlüğe bağlı bölümler,
   daire başkanlıkları, birimler/ofisler, koordinatörlükler, kurullar, araştırma merkezleri (sayı + liste).
2. Her akademik birim (www "Akademik" menüsündeki her fakülte/enstitü/YO/MYO/konservatuvar) için bir belge:
   - bölüm sayısı ve listesi; birimin web adresi
   - her bölüm: web adresi, programları/anabilim dalları (menüden), "Hakkında" sayfasından kısa tanıtım
   Menü düzenleri: (a) "Bölümler" alt menüsünde bölüm adları (fakülteler), (b) "Bölümler" menüsünün altında her
   bölüm ayrı alt menü (MYO'lar).
Kişi adı içeren sayfalar (yönetim, kadro) alınmaz.

Çıktı: veri/web/sayfalar/yapi__*.txt + veri/web/katalog_yapi.json (belge_isle.py web kataloglarıyla birlikte okur)
Kullanım:  python -X utf8 web_yapi.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path
from urllib.parse import urljoin, urlparse

from web_belgeleri import TITLE_FIXES, site_birim, tr_title
from web_tara import Api, html_text

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "veri" / "web"
ABOUT_CHARS = 1200
WWW_SECTIONS = {
    "FAKÜLTELER": "fakülte", "ENSTİTÜLER": "enstitü", "YÜKSEKOKULLAR": "yüksekokul", "KONSERVATUVAR": "konservatuvar",
    "MESLEK YÜKSEKOKULLARI": "meslek yüksekokulu", "REKTÖRLÜĞE BAĞLI BÖLÜMLER": "rektörlüğe bağlı bölüm",
    "Daire Başkanlıkları": "daire başkanlığı", "Birimler / Ofisler": "birim/ofis", "Koordinatörlükler": "koordinatörlük",
    "Kurullar / Komisyonlar": "kurul/komisyon", "Diğer Birimler": "diğer birim",
    "ARAŞTIRMA VE UYGULAMA MERKEZLERİ": "araştırma ve uygulama merkezi",
}
ACADEMIC_SECTIONS = {"FAKÜLTELER", "ENSTİTÜLER", "YÜKSEKOKULLAR", "KONSERVATUVAR", "MESLEK YÜKSEKOKULLARI"}
UNIT_LIST_RE = re.compile(r"^(akademik )?(bölümler(imiz)?|programlar(ımız)?|anabilim dalları(mız)?|anasanat dalları|"
                          r"(ön ?lisans|lisans|yüksek lisans|doktora|lisansüstü) programları|akademik birimler)$",
                          re.IGNORECASE)
PROGRAM_LIST_RE = re.compile(r"program|anabilim|anasanat|bilim dal", re.IGNORECASE)
# Kelime sınırlı: "haber" "Elektronik ve Haberleşme"yi, "yönetim" "Denizcilik İşletmeleri Yönetimi"ni düşürüyordu.
NOT_UNIT_RE = re.compile(r"komisyon|rapor|\bkurul|yönetmeli|yönerge|\bform|\bilan\b|duyuru|iş akış|şema|hakkında|"
                         r"misyon|iletişim|kadro|sınav|takvim|ders program|\bstaj\b|^yönetim$|laboratuvar|atölye|"
                         r"\bmezun|web sayfası|personel|akreditasyon|\bhaberler\b|galeri|^projeler$|^sss$|sıkça|"
                         r"^anabilim/anasanat dalları$|programları$", re.IGNORECASE)
GROUP_TITLE_RE = re.compile(r"bölümü$|tıp bilimleri$|bilimleri bölümü$", re.IGNORECASE)  # Tıp, Konservatuvar
ABOUT_RE = re.compile(r"hakkında|hakkımızda|tanıtım|genel bilgi", re.IGNORECASE)


def clean(name: str) -> str:
    name = " ".join((name or "").split()).strip(" -")
    return tr_title(name) if name.isupper() else name


def host_of(url: str, base: str = "") -> str:
    host = urlparse(urljoin(base, url or "")).netloc.lower()
    return "www.kocaeli.edu.tr" if host == "kocaeli.edu.tr" else host


class Site:
    def __init__(self, api: Api, host: str) -> None:
        self.api, self.host = api, host
        info = api.get("Admin/Isp/GetIspCodeByName", IspWebUrl=host) or {}
        self.code = str(info["code"]) if info.get("code") else None
        self.menu = (api.get("Menu/GetAll", self.code, TranslationCode=0) or []) if self.code else []

    def submenus(self):
        for item in self.menu:
            for sub in item.get("subMenus") or []:
                yield item.get("title") or "", sub.get("title") or "", sub.get("subMenuItems") or []

    def page_text(self, url: str) -> str:
        parsed = urlparse(urljoin(f"https://{self.host}/", url or ""))
        if host_of(url, f"https://{self.host}/") != self.host or "/sayfalar/" not in parsed.path:
            return ""
        slug = parsed.path.split("/sayfalar/")[1].strip("/")
        data = self.api.get("Page/GetBySlug", self.code, TranslationCode=0, Slug=slug) if self.code else None
        return html_text(((data or {}).get("page") or {}).get("content") or "")

    def about(self) -> str:
        for _, sub_title, items in self.submenus():
            for entry in items:
                if ABOUT_RE.search(entry.get("title") or ""):
                    text = self.page_text(entry.get("url") or "")
                    if len(text) > 80:
                        return summarize(text)
        return ""

    def programs(self) -> list[str]:
        found: list[str] = []
        for item_title, sub_title, items in self.submenus():
            if PROGRAM_LIST_RE.search(sub_title) and not NOT_UNIT_RE.search(sub_title):
                found += [clean(e.get("title")) for e in items if e.get("title") and not NOT_UNIT_RE.search(e["title"])]
        return list(dict.fromkeys(found))


def summarize(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= ABOUT_CHARS:
        return text
    cut = text[:ABOUT_CHARS]
    return cut[:cut.rfind(". ") + 1] if ". " in cut else cut + "…"


def departments(site: Site) -> list[tuple[str, str, list[dict]]]:
    """[(bölüm adı, bölüm sitesi ya da "", alt menü öğeleri)] — iki menü düzenini de tanır."""
    found: list[tuple[str, str, list[dict]]] = []
    for item_title, sub_title, items in site.submenus():
        sub = " ".join(sub_title.split())
        if UNIT_LIST_RE.match(sub):  # (a) "Bölümler" alt menüsü: öğeler bölüm adları
            for entry in items:
                name = clean(entry.get("title"))
                if name and not NOT_UNIT_RE.search(name):
                    target = host_of(entry.get("url") or "", f"https://{site.host}/")
                    found.append((name, target if target and target != site.host else "", []))
        elif (UNIT_LIST_RE.match(" ".join(item_title.split())) or GROUP_TITLE_RE.search(sub)) \
                and not NOT_UNIT_RE.search(sub):
            # (b) "Bölümler"/"Lisans" menüsünün altında her bölüm ayrı alt menü (MYO'lar, Konservatuvar) ya da
            # Tıp'taki "Temel/Dahili/Cerrahi Tıp Bilimleri" altında anabilim dalları
            website = next((host_of(e.get("url") or "") for e in items if "web" in (e.get("title") or "").lower()), "")
            found.append((clean(sub), website, items))
    unique: dict[str, tuple[str, str, list[dict]]] = {}
    for name, website, items in found:
        unique.setdefault(name, (name, website, items))
    return list(unique.values())


def write_doc(name: str, title: str, url: str, sections: list[tuple[str, str]]) -> str:
    from belge_isle import tr_upper
    body = "\n\n".join(f"{tr_upper(heading)}\n\n{text}" for heading, text in sections)
    path = WEB / "sayfalar" / f"yapi__{name}.txt"
    path.write_text(f"KAYNAK: {url}\nBAŞLIK: {title}\nERİŞİM: {date.today().isoformat()}\n\n{body}\n", encoding="utf-8")
    return path.name


def main() -> int:
    api = Api()
    meta = json.loads((WEB / "site_bilgisi.json").read_text(encoding="utf-8"))
    for old in (WEB / "sayfalar").glob("yapi__*.txt"):
        old.unlink()
    catalog: list[dict] = []
    www = Site(api, "www.kocaeli.edu.tr")

    # ---- 1. Üniversite geneli
    sections, academic_units = [], []
    for item_title, sub_title, items in www.submenus():
        key = sub_title.strip()
        names = [clean(e.get("title")) for e in items if e.get("title")]
        if key in WWW_SECTIONS and names:
            sections.append((key, f"Kocaeli Üniversitesi'nde {len(names)} {WWW_SECTIONS[key]} bulunmaktadır: "
                                  + ", ".join(names) + "."))
        if key in ACADEMIC_SECTIONS:
            academic_units += [(clean(e.get("title")), host_of(e.get("url") or ""), WWW_SECTIONS[key]) for e in items]
    title = "Kocaeli Üniversitesi Akademik ve İdari Yapısı"
    file = write_doc("www", title, "https://www.kocaeli.edu.tr/", sections)
    catalog.append({"ad": "web_yapi__www", "kisa": title, "baslik": title, "dosya": f"web/sayfalar/{file}",
                    "birim": "tum", "kaynak_url": "https://www.kocaeli.edu.tr/"})
    print(f"Üniversite: {len(sections)} başlık, {len(academic_units)} akademik birim", flush=True)

    # ---- 2. Akademik birimler ve bölümleri
    for unit_name, host, kind in academic_units:
        if not host.endswith("kocaeli.edu.tr"):
            continue
        site = Site(api, host)
        prefix = host.split(".")[0]
        full_name = clean(TITLE_FIXES.get(prefix) or meta.get(host, {}).get("baslik") or unit_name)
        if not re.search(r"fakülte|yüksekokul|enstitü|konservatuvar", full_name, re.IGNORECASE):
            full_name = f"{unit_name} {kind.title()}" if kind != "fakülte" else unit_name
        depts = departments(site)
        noun = "anabilim dalı" if kind == "enstitü" else "bölüm"
        intro = (f"{full_name}, Kocaeli Üniversitesi'ne bağlı bir {kind}dir. Web sitesi: https://{host}/ . "
                 + (f"{full_name} bünyesinde {len(depts)} {noun} bulunmaktadır: " + ", ".join(d[0] for d in depts) + "."
                    if depts else "Bölüm listesi sitenin menüsünde yer almıyor."))
        about = site.about()
        sections = [(f"{full_name} genel bilgiler", intro + (f"\nTanıtım: {about}" if about else ""))]
        # Lisansüstü/program listeleri: menüde ad olarak (Hukuk: "Kamu Hukuku Yüksek Lisans") ya da ayrı sayfada
        # (Fen Bilimleri Enstitüsü: "Yüksek Lisans Programları" sayfası).
        listed, pages = [], []
        for item_title, sub_title, items in site.submenus():
            for entry in items:
                title = clean(entry.get("title"))
                if re.search(r"yüksek lisans|doktora|sanatta yeterlik", title, re.IGNORECASE) \
                        and not re.search(r"takvim|enstitü|form|yönetmeli|ders", title, re.IGNORECASE):
                    if re.search(r"programları$", title, re.IGNORECASE):
                        text = site.page_text(entry.get("url") or "")
                        if len(text) > 50:
                            pages.append((title, summarize(text) if len(text) > 3000 else " ".join(text.split())))
                    else:
                        listed.append(title)
        if listed:
            sections.append((f"{full_name} lisansüstü programları",
                             f"{full_name} bünyesindeki lisansüstü programlar ({len(listed)}): " + ", ".join(
                                 dict.fromkeys(listed)) + "."))
        for title, text in pages:
            sections.append((f"{full_name} {title}", text))
        for name, website, items in depts:
            parts = [f"{name}, {full_name} bünyesindedir."]
            programs: list[str] = []
            dept_about = ""
            if website.endswith("kocaeli.edu.tr"):
                parts.append(f"Web sitesi: https://{website}/ .")
                dept_site = Site(api, website)
                programs = dept_site.programs()
                dept_about = dept_site.about()
            else:  # (b) düzeni: bölüm sayfaları birimin kendi sitesinde
                programs = [clean(re.sub(r"(?i)program hakkında\s*-\s*", "", e.get("title") or "")) for e in items
                            if re.search(r"program|anabilim dalı|\babd\b", e.get("title") or "", re.IGNORECASE)
                            and not NOT_UNIT_RE.search(re.sub(r"(?i)program hakkında", "", e.get("title") or ""))]
                about_entry = next((e for e in items if ABOUT_RE.search(e.get("title") or "")), None)
                if about_entry:
                    dept_about = summarize(site.page_text(about_entry.get("url") or ""))
            if programs:
                parts.append(f"Programları / anabilim dalları ({len(programs)}): " + ", ".join(programs) + ".")
            if dept_about:
                parts.append(f"Tanıtım: {dept_about}")
            sections.append((name, " ".join(parts)))
        title = f"{full_name} Akademik Yapısı ve Bölümleri"
        file = write_doc(prefix, title, f"https://{host}/", sections)
        birim, _ = site_birim(host, meta) if host in meta else ("tum", "")
        catalog.append({"ad": f"web_yapi__{prefix}", "kisa": title, "baslik": title, "dosya": f"web/sayfalar/{file}",
                        "birim": birim, "kaynak_url": f"https://{host}/"})
        print(f"  {full_name}: {len(depts)} {noun}" + (f" ({', '.join(d[0] for d in depts[:4])}…)" if depts else ""),
              flush=True)

    for entry in catalog:
        entry.update({"tur": "web sayfası", "hiyerarsi": 5, "yapi": "serbest", "hedef_kitle": "herkes",
                      "kurum": "KOÜ", "kapsam": "web", "durum": "guncel", "madde_hedef_kitle": {},
                      "yururluk_tarihi": "", "yururluk_bitis": ""})
    (WEB / "katalog_yapi.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Yapı belgesi: {len(catalog)} -> {WEB / 'katalog_yapi.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
