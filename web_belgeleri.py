#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Web'den toplanan sayfaları ve indirilen PDF'leri belge_isle.py'nin okuyacağı kataloğa çevirir.

Girdi:  veri/web/secilen.json (web_ayikla.py), veri/web/pdf/*.pdf (web_pdf_indir.py),
        veri/web/site_bilgisi.json (site -> öğrenci bilgi sistemi kodu, başlık), veri/web/pdf_baglantilari.tsv
Çıktı:  veri/web/katalog_web.json  (belge_isle.py ana kataloğa ekler)
        veri/web/birimler_ek.json  (rag_v2.BIRIMLER'e eklenen yeni fakülte/MYO adları)
        veri/web/pdf_ayiklama.tsv  (her PDF'in kararı)

Birim: sitenin öğrenci bilgi sistemi kodundan. Fakülte/MYO kodu "#02#"; bölüm kodu "#0201#" -> ilk iki hane
fakülte. Enstitü siteleri "enstitu"; daire başkanlıkları, koordinatörlükler, merkezler "tum".
PDF ayıklama: metinsiz (taranmış), elimizdeki veriyle aynı (kelime 8'lisi örtüşmesi %40+), PDF'ler arası kopya
ve aynı başlıklı eski sürüm atlanır; kalanlar qwen3:14b konu kontrolünden geçer (web_ayikla.classify).

Kullanım:  python -X utf8 web_belgeleri.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import pymupdf

from web_ayikla import EXISTING_OVERLAP, classify, shingles

ROOT = Path(__file__).resolve().parent.parent
VERI = ROOT / "veri"
WEB = VERI / "web"
ARTICLES = VERI / "islenmis" / "maddeler.jsonl"
PDF_CACHE = WEB / "pdf_konu_kontrol.json"
MIN_PDF_CHARS = 300

# Mevcut birim anahtarları (rag_v2.BIRIMLER) ile aynı fakülteler; diğerleri site önekinden yeni anahtar alır.
KNOWN_FACULTY_KEYS = {
    "fef": "fen_edebiyat", "mf": "muhendislik", "tip": "tip", "if": "iletisim", "mtf": "mimarlik", "dishek": "dis",
    "teknoloji": "teknoloji", "denizcilik": "denizcilik", "ziraat": "ziraat", "turizm": "turizm",
    "dk": "konservatuvar", "kartepeturizm": "kartepe_myo", "kmyo": "kocaeli_myo",
    "degirmenderemyo": "degirmendere_myo", "foimyo": "ford_otosan_myo",
}
# Tüm hazırlık öğrencilerine hizmet veren birim: fakülteye özel sayılmaz.
GENERAL_FACULTY_SITES = {"yabancidiller"}
# Kayıtta OBS kodu olmayan ya da yanlış eşlenen siteler (23.10 denetimi): atip/pediatri Tıp'ın, aotomotiv/ateknoloji
# Teknoloji Fakültesi'nin akademik siteleri; www ana sitedir (kod 1000'in ilk iki hanesi "10" Mimarlık sanılıyordu).
SITE_UNIT_OVERRIDES = {"www": "tum", "atip": "tip", "pediatri": "tip", "aotomotiv": "teknoloji",
                       "ateknoloji": "teknoloji", "yapayzeka": "fen_edebiyat", "kousim": "tip",
                       "otomotivabd": "enstitu"}
TITLE_FIXES = {"tip": "Tıp Fakültesi", "dishek": "Diş Hekimliği Fakültesi",
               "uzunyolmyo": "Hereke Ömer İsmet Uzunyol Meslek Yüksekokulu",
               "hakmyo": "Hereke Asım Kocabıyık Meslek Yüksekokulu",
               "golcukmyo": "Gölcük Meslek Yüksekokulu", "kandiramyo": "Kandıra Meslek Yüksekokulu",
               "izmitmyo": "İzmit Meslek Yüksekokulu",
               # Kayıtta adı "Kocaeli Üniversitesi" ya da boş görünen siteler (23.10): belge adı belirsiz kalıyordu
               "yapayzeka": "Yapay Zeka ve Makine Öğrenmesi Bölümü (Fen Edebiyat Fakültesi)",
               "atip": "Tıp Fakültesi", "aotomotiv": "Otomotiv Mühendisliği (Teknoloji Fakültesi)",
               "ateknoloji": "Teknoloji Fakültesi", "etikkurul": "Etik Kurullar", "kulturevi": "Kültür Evi",
               "tipmezun": "Tıp Fakültesi Mezunları", "kousim": "Tıp Fakültesi Simülasyon Merkezi"}
DOC_TYPES = [(r"yönetmeli", "yönetmelik", 2), (r"yönerge", "yönerge", 3), (r"usul ve esas|esaslar", "usul ve esaslar", 4),
             (r"senato karar", "senato kararı", 3), (r"kılavuz|rehber", "kılavuz", 5)]


GENERIC_LABEL_RE = re.compile(r"^(için )?(dosya(yı)?( indir)?|tıklayınız\.*|indir|link|buraya tıklayın\w*|"
                              r"görüntüle|detay\w*|pdf)\.*$", re.IGNORECASE)
TITLE_HINT_RE = re.compile(r"yönerge|yönetmeli|esas|ilke|kılavuz|rehber|usul|prosedür|talimat|iş akış|"
                           r"görev tanım|sözleşme|karar|staj|uygulama|takvim", re.IGNORECASE)


def pdf_title(text: str) -> str:
    """PDF'in ilk satırlarından başlık: belge türü sözcüğü geçen ya da en uzun BÜYÜK HARFLİ satır(lar)."""
    from belge_isle import fix_font_glyphs

    if text.startswith("__TABLO__"):
        # tablo_metin.py bölümü: yıl içeren tablo adı en iyi başlık; yoksa asıl metinden seçilir
        section, _, text = text.partition("__TABLO_SON__")
        names = [" ".join(l[8:].split()) for l in section.split("\n") if l.startswith("[Tablo] ")]
        dated = [n for n in names if re.search(r"\d{4}\s*-\s*\d{4}", n)
                 and re.search(r"(?i)ücret|katkı|harç|tarife|fiyat", n)]
        if dated:
            return tr_title(dated[0]) if dated[0].isupper() else dated[0]
    lines = [" ".join(l.split()) for l in fix_font_glyphs(text).split("\n")[:25]]
    # EBYS evrak damgası ("Evrak Tarih ve Sayısı: 26.08.2026-E.1020878") başlık değil
    lines = [re.sub(r"^Evrak Tarih ve Sayısı:\s*\S+\s*", "", l) for l in lines]
    lines = [l.lstrip("¹²³*•-– ") for l in lines if not l.startswith("__PAGE_")]  # dipnot/madde imi önekleri ("¹KOCAELİ ÜNİVERSİTESİ ...")
    lines = [l for l in lines if 6 <= len(l) <= 160 and not re.fullmatch(r"[\d\W]+", l)]
    hinted = [l for l in lines if TITLE_HINT_RE.search(l)]
    if hinted:
        index = lines.index(hinted[0])
        # Başlık genelde 2-3 satıra bölünür ("KOCAELİ ÜNİVERSİTESİ / ... FAKÜLTESİ / STAJ YÖNERGESİ").
        title = " ".join(lines[max(0, index - 1):index + 1]) if index and lines[index - 1].isupper() else hinted[0]
    else:
        upper = [l for l in lines if sum(c.isupper() for c in l) > 0.7 * sum(c.isalpha() for c in l)]
        title = max(upper, key=len) if upper else ""
    title = re.sub(r"^(T\.?C\.?\s*)?KOCAELİ ÜNİVERSİTESİ\s*", "", title, flags=re.IGNORECASE).strip(" -–:")
    return tr_title(title) if title.isupper() else title


def tr_title(text: str) -> str:
    """Türkçe büyük/küçük harf: str.title() "FAKÜLTESİ"ni "Fakültesi̇" yapıyordu (noktalı i birleşik karakteri)."""
    from raft_format import tr_capitalize
    return " ".join(tr_capitalize(word) for word in text.split())


def site_birim(host: str, meta: dict[str, dict]) -> tuple[str, str]:
    """(birim anahtarı, birim adı)."""
    prefix = host.split(".")[0]
    info = meta.get(host, {})
    if prefix in SITE_UNIT_OVERRIDES:
        key = SITE_UNIT_OVERRIDES[prefix]
        return key, "" if key == "tum" else TITLE_FIXES.get(key, "")
    if info.get("tur") == 1:  # kurumsal ana site
        return "tum", ""
    title = TITLE_FIXES.get(prefix) or (info.get("baslik") or "").strip()
    if "enstitü" in title.lower():
        return "enstitu", "Enstitü (yüksek lisans, doktora)"
    code = (info.get("obs") or "").split(";")[0].strip("#")
    if not code or prefix in GENERAL_FACULTY_SITES:
        return "tum", ""
    faculty = code[:2]
    for site, other in meta.items():
        if (other.get("obs") or "").strip("#") == faculty:
            fprefix = site.split(".")[0]
            if fprefix in GENERAL_FACULTY_SITES:
                return "tum", ""
            ftitle = TITLE_FIXES.get(fprefix) or other.get("baslik", "")
            return KNOWN_FACULTY_KEYS.get(fprefix, fprefix), tr_title(ftitle) if ftitle.isupper() else ftitle
    return "tum", ""


def doc_type(title: str) -> tuple[str, int]:
    lowered = title.lower()
    for pattern, name, level in DOC_TYPES:
        if re.search(pattern, lowered):
            return name, level
    return "web sayfası", 5


def main() -> int:
    meta = json.loads((WEB / "site_bilgisi.json").read_text(encoding="utf-8"))
    pages = json.loads((WEB / "secilen.json").read_text(encoding="utf-8"))
    catalog: list[dict] = []
    extra_units: dict[str, str] = {}

    def add(entry: dict, host: str) -> None:
        birim, name = site_birim(host, meta)
        if birim not in ("tum", "enstitu") and name:
            extra_units.setdefault(birim, name)
        site_title = TITLE_FIXES.get(host.split(".")[0]) or meta.get(host, {}).get("baslik") or host
        entry.update({"birim": birim, "kurum": "KOÜ", "kapsam": "web", "durum": "guncel",
                      "site": host, "site_adi": site_title, "madde_hedef_kitle": {},
                      "yururluk_tarihi": "", "yururluk_bitis": ""})
        catalog.append(entry)

    # ---- Web sayfaları (serbest metin)
    for page in pages:
        host = urlparse(page["url"]).netloc
        site_title = TITLE_FIXES.get(host.split(".")[0]) or meta.get(host, {}).get("baslik") or host
        kind, level = doc_type(page["baslik"])
        add({"ad": "web_" + page["dosya"][:-4].replace("-", "_"), "kisa": f"{site_title} – {page['baslik']}",
             "baslik": page["baslik"], "tur": kind, "hiyerarsi": level, "yapi": "serbest",
             "hedef_kitle": page.get("kitle") or "herkes", "dosya": f"web/sayfalar/{page['dosya']}",
             "kaynak_url": page["url"]}, host)
    print(f"Web sayfası: {len(pages)}")

    # ---- PDF'ler
    links = {}
    for url, label, page in csv.reader((WEB / "pdf_baglantilari.tsv").open(encoding="utf-8"), delimiter="\t"):
        query = parse_qs(urlparse(url).query).get("filePath", [""])[0]
        name = re.sub(r"[^\w.\-]+", "_", unquote(Path(query or urlparse(url).path).name).lower())[:150]
        links.setdefault(name, (url, label, page))
    existing: set[tuple[str, ...]] = set()
    with ARTICLES.open(encoding="utf-8") as input_file:
        for line in input_file:
            row = json.loads(line)
            if not row["ozellikler"]["belge_id"].startswith("web"):  # önceki web işlemesi kendisiyle eşleşmesin
                existing |= shingles(row["metin"])
    cache = json.loads(PDF_CACHE.read_text(encoding="utf-8")) if PDF_CACHE.exists() else {}
    decisions, kept_pdf, kept_sh = [], {}, []
    for path in sorted((WEB / "pdf").glob("*.pdf"), reverse=True):  # tarih önekiyle: en yeni önce
        url, label, page = links.get(path.name, ("", "", ""))
        try:
            with pymupdf.open(path) as pdf:
                text = "\n".join(p.get_text() for p in pdf)
                first = text[:800]
        except Exception:
            decisions.append((path.name, "açılamadı")); continue
        if len(text.strip()) < MIN_PDF_CHARS:
            decisions.append((path.name, "metin yok (taranmış)")); continue
        sh = shingles(text)
        overlap = len(sh & existing) / len(sh) if sh else 0
        if overlap > EXISTING_OVERLAP:
            decisions.append((path.name, f"elimizde var (%{100 * overlap:.0f})")); continue
        if any(sh and len(sh & other) / len(sh) > 0.6 for other in kept_sh):
            decisions.append((path.name, "PDF'ler arası kopya / eski sürüm")); continue
        title = label.strip()
        if not title or GENERIC_LABEL_RE.match(title):  # "Dosya", "tıklayınız", "İNDİR": başlık PDF'in kendisinden
            title = pdf_title(text) or re.sub(r"^\d{8}_[0-9a-f]{5}_|-[0-9a-f]{3}\.pdf$|\.pdf$", "",
                                              path.name).replace("-", " ").replace("_", " ")
        if path.name not in cache:
            cache[path.name] = classify(title, text)
            PDF_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
        verdict = cache[path.name]
        if not verdict.get("faydali"):
            decisions.append((path.name, f"konu dışı: {verdict.get('neden', '')[:60]}")); continue
        kept_sh.append(sh)
        host = urlparse(page).netloc or "www.kocaeli.edu.tr"
        kind, level = doc_type(title + " " + first)
        articles = len(re.findall(r"(?im)^\s*(?:MADDE|Madde)\s+\d+", text))
        site_title = TITLE_FIXES.get(host.split(".")[0]) or meta.get(host, {}).get("baslik") or host
        add({"ad": "webpdf_" + re.sub(r"\W+", "_", path.stem.lower())[:70], "kisa": f"{site_title} – {title[:90]}",
             "baslik": title, "tur": kind, "hiyerarsi": level, "yapi": "" if articles >= 3 else "serbest",
             "hedef_kitle": verdict.get("kitle") or "herkes", "dosya": f"web/pdf/{path.name}",
             "kaynak_url": url}, host)
        kept_pdf[path.name] = title
        decisions.append((path.name, "ALINDI"))
    with (WEB / "pdf_ayiklama.tsv").open("w", encoding="utf-8") as output:
        output.write("dosya\tkarar\n")
        output.writelines(f"{name}\t{reason}\n" for name, reason in decisions)
    counts: dict[str, int] = {}
    for _, reason in decisions:
        key = reason.split(" (")[0].split(":")[0]
        counts[key] = counts.get(key, 0) + 1
    print(f"PDF: {counts}")

    # Aynı ada sahip belgeler (farklı sitelerde aynı başlık) ayırt edilsin
    seen: dict[str, int] = {}
    for entry in catalog:
        seen[entry["kisa"]] = seen.get(entry["kisa"], 0) + 1
    (WEB / "katalog_web.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    (WEB / "birimler_ek.json").write_text(json.dumps(extra_units, ensure_ascii=False, indent=1), encoding="utf-8")
    by_unit: dict[str, int] = {}
    for entry in catalog:
        by_unit[entry["birim"]] = by_unit.get(entry["birim"], 0) + 1
    print(f"Katalog: {len(catalog)} belge, birimler: {by_unit}")
    print(f"Yeni birim adı: {len(extra_units)} -> {WEB / 'birimler_ek.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
