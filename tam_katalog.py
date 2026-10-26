#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tam taramanın (web_tara_tam.py, web_tara ek siteler, web_dosya_indir.py + dosya_metin.py, akts_tara.py) çıktısını
belge_isle.py kataloğuna çevirir.

Ana katalog (veri/web/katalog_tam.json): site sayfaları, eski site sayfaları, işlem içeren duyurular, indirilen dosyalar
(PDF/DOCX/DOC), AKTS program bilgileri. Eski katalog_web.json'un yerini alır (o sayfa ve PDF'ler bunun alt kümesi).
Ders kataloğu (veri/web/dersler_katalog.json): AKTS ders dosyaları; ayrı koleksiyona girer (ana aramayı boğmasın).

Elenenler: kısa/boş (taranmış) metin, kişi listeleri, çoğunluğu sayı olan tablolar (sınav/ders programı, liste),
konferans/tören/etkinlik duyuruları, elimizdeki veriyle ya da birbiriyle aynı belgeler (kelime 8'lisi örneklem
"parmak izi" ile yakın kopya tespiti; aynı belgenin en yeni kopyası kalır).
Birim: site -> öğrenci bilgi sistemi kodu (web_belgeleri.site_birim); bilinmeyen siteler API'den sorulur.

Kullanım:  python -X utf8 tam_katalog.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from pdf_to_chromadb import normalize_for_matching
from web_belgeleri import GENERIC_LABEL_RE, SITE_UNIT_OVERRIDES, TITLE_FIXES, doc_type, pdf_title, site_birim, tr_title

ROOT = Path(__file__).resolve().parent.parent
VERI = ROOT / "veri"
TAM = VERI / "web_tam"
WEB = VERI / "web"
SKETCH = 64
NEAR_DUP = 0.6
EXISTING_DUP = 0.5
STAFF_HINT_RE = re.compile(r"personel|akademisyen|öğretim üyesi|öğretim elemanı|ek ders|maaş|görevlendirme|bap\b|"
                           r"ihale|satın alma|doğrudan temin|harcırah|kadro|atama|terfi|izin|sicil|bordro|döner sermaye",
                           re.IGNORECASE)
STUDENT_HINT_RE = re.compile(r"öğrenci|ders kaydı|kayıt yenileme|harç|staj|sınav|mezuniyet|diploma|burs|yatay geçiş|"
                             r"çift anadal|erasmus|hazırlık", re.IGNORECASE)
CENTRAL_SITES = {"www", "odb", "perdb", "imidb", "sgdb", "sksdb", "int", "yos", "bap", "kalite", "egitimkoord"}
FOREIGN_RE = re.compile(r"(?i)gebze teknik|\bgtü\b|\bgtu\b|\badep\b|proje ortağı olmak|_is_birligi_cag|"
                        r"fırat|firat_|özyeğin|ozyegin")
EVENT_RE = re.compile(r"konferans|seminer|webinar|söyleşi|panel\b|sempozyum|kongre|tören|konser|sergi|yarışma|"
                      r"festival|ziyaret|anma|kutlama|gezisi|turnuva|şenlik|tebrik|başsağlığı|vefat", re.IGNORECASE)
PROCEDURE_RE = re.compile(r"başvuru|kayıt|harç|katkı payı|ücret|sınav|takvim|son gün|tarih|burs|staj|ime\b|mezun|"
                          r"diploma|yatay geçiş|dikey geçiş|muafiyet|intibak|kontenjan|ilan|erasmus|farabi|mevlana|"
                          r"belge|form|dilekçe|yönerge|yönetmelik|esas|duyurulur|ilan edil|teslim|ödeme", re.IGNORECASE)
PEOPLE_WORD_RE = re.compile(r"\b(Prof\.|Doç\.|Dr\.|Öğr\.|Arş\.|Gör\.|Uzm\.|Okt\.)")


def body_of(raw: str) -> tuple[dict, str]:
    if raw.startswith("KAYNAK:"):
        head, _, text = raw.partition("\n\n")
        meta = dict(re.findall(r"^(\w+): (.*)$", head, re.M))
        return meta, text
    return {}, raw


def sketch(text: str) -> list[int]:
    words = re.findall(r"\w+", text.replace("İ", "i").replace("I", "ı").lower())
    hashes = {int(hashlib.md5(" ".join(words[i:i + 8]).encode()).hexdigest()[:12], 16)
              for i in range(max(0, len(words) - 7))}
    return sorted(hashes)[:SKETCH]


class DupIndex:
    """Örneklem parmak izleri (en küçük 64 özet) üzerinden yakın kopya araması."""

    def __init__(self) -> None:
        self.index: dict[int, list[int]] = defaultdict(list)
        self.sketches: list[list[int]] = []

    def similar(self, sk: list[int], threshold: float) -> bool:
        if not sk:
            return False
        counts: dict[int, int] = defaultdict(int)
        for h in sk:
            for doc in self.index.get(h, ()):
                counts[doc] += 1
        return any(c / min(len(sk), len(self.sketches[d])) >= threshold for d, c in counts.items())

    def add(self, sk: list[int]) -> None:
        doc = len(self.sketches)
        self.sketches.append(sk)
        for h in sk:
            self.index[h].append(doc)


# Kelime içinde büyük/küçük harf karmaşası ("tIyuLAcAK", bitişik "AhmetYILMAZ") ve harf-rakam karışık tarih/saat
# ("l0.10.20l7", "09:@"): bozuk OCR ya da ad-soyad dolu liste/çizelge (ders-sınav programı, personel listesi,
# sonuç listesi). Denetim (19.10): oranı 0.08'i aşan 38 belgenin hepsi bu türdendi.
OCR_MIX_RE = re.compile(r"\b(?=\w*[a-zçğıöşü][A-ZÇĞİÖŞÜ])(?=\w*[A-ZÇĞİÖŞÜ][a-zçğıöşü])[A-Za-zÇĞİÖŞÜçğıöşü]{5,}\b")
OCR_DIGIT_RE = re.compile(r"\b(?=[\w.|]*\d)(?=[\w.|]*[lIO|])[0-9lIO|]{2,}[.:][0-9lIO|@]{2,}\b|@")
NOISY_TEXT = 0.08


def noisy_text(text: str) -> bool:
    body = re.sub(r"__TABLO__.*?__TABLO_SON__|https?://\S+", " ", text, flags=re.S)  # bağlantılar (OpenFile) sayılmaz
    tokens = len(re.findall(r"\S+", body))
    return tokens >= 40 and (len(OCR_MIX_RE.findall(body)) + len(OCR_DIGIT_RE.findall(body))) / tokens > NOISY_TEXT


def garbled_title(title: str) -> bool:
    """OCR'dan gelen bozuk başlık ("tIyuLAcAK EsAsLARA iıişrhr yöNnRcn")."""
    return len(OCR_MIX_RE.findall(title)) >= 2


def numeric_table(text: str) -> bool:
    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    return digits > 0.35 * max(1, letters + digits)


def people_list(text: str) -> bool:
    return len(PEOPLE_WORD_RE.findall(text)) > max(5, len(text) / 250)


# Ad listeleri (sonuç/yerleşim/asil-yedek/maç bileti listeleri, "Ekibimiz" sayfaları): unvansız oldukları için
# people_list yakalamıyordu; 23.10'da veride ~260 belge bulundu (Erasmus dil sınavı sonuçları, yatay geçiş listeleri).
NAME_RE = re.compile(r"\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]+ [A-ZÇĞİÖŞÜ]{2,}(?: [A-ZÇĞİÖŞÜ]{2,})?\b")
MASKED_NAME_RE = re.compile(r"[A-ZÇĞİÖŞÜa-z]\*{2,}|\d{2,}\*{3,}")
NAME_LIST_TITLE_RE = re.compile(r"sonuc|liste|yerlesim|asil|yedek|kazanan|basari durum|degerlendirme|mac bilet|"
                                r"ekibimiz|personeller\b|personelimiz|eslestirme|kabul edilen")
NOT_NAME_LIST_TITLE_RE = re.compile(r"ders liste|ders program|secmeli ders|basili evrak|oz degerlendirme|yonerge|"
                                    r"yonetmelik|kilavuz|esaslar|rapor|akts|olcme|hizmet standart|bilgilendirme|"
                                    r"anket|form(u|lari)?\b|bidr|kidr|kontrol listesi|dosya icerik|hizmetin adi|"
                                    r"mali kontrol")
STRONG_LIST_TITLE_RE = re.compile(r"liste|sonuc|kazanan|asil|yedek|danisman")
# 1000 karakterde ad sayısı. 4 iken uzun tablolar kaçıyordu (Mühendislik yatay geçiş sonuçları: 488 KB'ta 1350 maskeli
# ad = 2.8); öğrenci numarası da ad sayılır. "Doğa Altun" gibi yazımlar (TITLE_NAME_RE) normal metinde de geçtiği
# için yalnız çok yoğunsa sayılır (Mimarlık staj listeleri, 23.10).
NAME_LIST_DENSITY = 1.5
TITLE_NAME_DENSITY = 6
STUDENT_NO_RE = re.compile(r"\b[12]\d{8}\b")
RESULT_STATUS_RE = re.compile(r"\b(BAŞARILI|BAŞARISIZ|ASİL|YEDEK|KABUL|RET|GEÇTİ|KALDI|UYGUN DEĞİL)\b")
TITLE_NAME_RE = re.compile(r"\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,} [A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,}\b")


def name_list(title: str, text: str) -> bool:
    normalized = normalize_for_matching(title)
    if not NAME_LIST_TITLE_RE.search(normalized) or NOT_NAME_LIST_TITLE_RE.search(normalized):
        return False
    size = max(1.0, len(text) / 1000)
    names = len(NAME_RE.findall(text)) + len(MASKED_NAME_RE.findall(text)) + len(STUDENT_NO_RE.findall(text))
    title_names = len(TITLE_NAME_RE.findall(text))
    statuses = len(RESULT_STATUS_RE.findall(text))  # satır başına sonuç: adlar sütunlara dağılmış tablolar
    return ((names >= 15 and names / size >= NAME_LIST_DENSITY)
            or (STRONG_LIST_TITLE_RE.search(normalized) and statuses >= 10 and statuses / size >= 1)
            or (STRONG_LIST_TITLE_RE.search(normalized) and title_names >= 25
                and title_names / size >= TITLE_NAME_DENSITY))


def audience(text: str) -> str:
    staff = len(STAFF_HINT_RE.findall(text[:6000]))
    student = len(STUDENT_HINT_RE.findall(text[:6000]))
    return "personel" if staff > 2 * max(1, student) else ("ogrenci" if student > 2 * max(1, staff) else "herkes")


def ensure_site_meta(hosts: set[str], meta: dict) -> None:
    from web_tara import Api
    api = Api()
    for host in sorted(hosts - set(meta)):
        try:
            info = api.get("Admin/Isp/GetIspCodeByName", IspWebUrl=host) or {}
            title = ""
            if info.get("code"):
                common = api.get("Common/GetAll", str(info["code"]), TranslationCode=0) or {}
                title = ((common.get("systemSettings") or {}).get("title") or "").strip()
            meta[host] = {"kod": info.get("code"), "tur": info.get("siteType"), "obs": info.get("obsFacultyProgramCode"),
                          "baslik": title}
        except Exception:
            meta[host] = {"kod": None, "tur": None, "obs": None, "baslik": ""}


def main() -> int:
    meta_path = WEB / "site_bilgisi.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pages = sorted((TAM / "sayfalar").glob("*.txt")) + sorted((TAM / "html").glob("*.txt"))
    announcements = sorted((TAM / "duyurular").glob("*.txt"), reverse=True)  # en yeni önce
    texts = sorted((TAM / "metin").glob("*.txt"), reverse=True)  # tarih önekli adlar: en yeni önce
    # dosya -> (adres, bulunduğu sayfa, bağlantı metni)
    file_meta: dict[str, tuple[str, str, str]] = {}
    for tsv in (TAM / "dosya_indirme.tsv", WEB / "pdf_indirme.tsv"):
        if tsv.exists():
            for line in tsv.read_text(encoding="utf-8").splitlines()[1:]:
                p = line.split("\t")
                if len(p) >= 5:  # dosya_indirme.tsv: adres, dosya, sayfa, metin, durum
                    file_meta.setdefault(p[1], (p[0], p[2], p[3]))
                elif len(p) >= 3:  # pdf_indirme.tsv: adres, dosya, durum
                    file_meta.setdefault(p[1], (p[0], "", ""))
    hosts = set()
    for path in pages + announcements:
        meta_line = body_of(path.read_text(encoding="utf-8", errors="ignore")[:400])[0]
        hosts.add(urlparse(meta_line.get("KAYNAK", "")).netloc.lower())
    for name, (url, page, _) in file_meta.items():
        hosts.add(urlparse(page or url).netloc.lower())
    hosts.discard("")
    ensure_site_meta({h for h in hosts if h.endswith("kocaeli.edu.tr") and h != "api.kocaeli.edu.tr"}, meta)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    # Mevcut (web dışı) verinin parmak izleri: aynı içerik ikinci kez girmesin
    existing = DupIndex()
    by_doc: dict[str, list[str]] = defaultdict(list)
    for line in (VERI / "islenmis" / "maddeler.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        belge = row["ozellikler"]["belge_id"]
        if not belge.startswith(("web", "webpdf")):
            by_doc[belge].append(row["metin"])
    for chunks in by_doc.values():
        existing.add(sketch(" ".join(chunks)))

    seen = DupIndex()
    # Silinen (KOÜ ile ilgisiz) belgelerin içerikçe kopyaları da alınmaz: aslı silinince kopyası "kopya" sayılmayıp
    # kataloğa giriyordu (22.10: iş ilanları, vaka sunumları birden çok sitede yayımlanmış).
    removed = DupIndex()
    for folder in ("metin", "sayfalar", "duyurular"):
        for path in (VERI / "silinen" / "web_tam" / folder).glob("*.txt"):
            removed.add(sketch(re.sub(r"__PAGE_\d+__", " ", path.read_text(encoding="utf-8", errors="ignore"))))
    catalog, reasons = [], defaultdict(int)
    extra_units: dict[str, str] = json.loads((WEB / "birimler_ek.json").read_text(encoding="utf-8"))
    excluded = {h["ad"] for h in json.loads((WEB / "haric.json").read_text(encoding="utf-8"))} \
        if (WEB / "haric.json").exists() else set()

    def site_title(host: str) -> str:
        prefix = host.split(".")[0]
        title = TITLE_FIXES.get(prefix) or meta.get(host, {}).get("baslik") or host
        return tr_title(title) if title.isupper() else title

    def admit(text: str, kind_reason: str = "") -> bool:
        if len(text.strip()) < 300:
            reasons["kısa/boş"] += 1
            return False
        if people_list(text):
            reasons["kişi listesi"] += 1
            return False
        if numeric_table(text):
            reasons["sayı tablosu"] += 1
            return False
        if noisy_text(text):
            reasons["bozuk metin / ad listesi"] += 1
            return False
        sk = sketch(text)
        if removed.similar(sk, NEAR_DUP):
            reasons["KOÜ dışı içerik (kopya)"] += 1
            return False
        if existing.similar(sk, EXISTING_DUP):
            reasons["elimizde var"] += 1
            return False
        if seen.similar(sk, NEAR_DUP):
            reasons["kopya"] += 1
            return False
        seen.add(sk)
        return True

    def add(entry: dict, host: str, text: str) -> None:
        # Başka kurumun kurallarını taşıyan belgeler (KOÜ sitesinde yayımlanmış ortak çağrılar): BAP sorusunda
        # GTÜ'nün çağrı metninden "ebap.gtu.edu.tr, 2.500.000 TL" cevabı geliyordu (18.10, kullanıcı istedi).
        if entry["ad"] in excluded or "BBClone" in entry["kisa"]:  # BBClone: site istatistik yazılımı sayfası  # KOÜ ile ilgisiz bulunup silinenler (ilgisiz_bul.py / ilgisiz_sil.py, 22.10)
            reasons["KOÜ dışı içerik"] += 1
            return
        if name_list(f"{entry['kisa']} {entry['baslik']}", text):
            reasons["ad listesi"] += 1
            return
        if FOREIGN_RE.search(f"{entry['kisa']} {entry['ad']}") or "gtu.edu.tr" in text.lower():
            reasons["başka kurum"] += 1
            return
        birim, unit_name = site_birim(host, meta) if host in meta or host.split(".")[0] in SITE_UNIT_OVERRIDES \
            else ("tum", "")
        if birim not in ("tum", "enstitu") and unit_name:
            extra_units.setdefault(birim, unit_name)
        kind, level = doc_type(entry["baslik"] + " " + text[:600])
        entry.update({"tur": entry.get("tur") or kind, "hiyerarsi": entry.get("hiyerarsi") or level, "birim": birim,
                      "kurum": "KOÜ", "kapsam": "web", "durum": "guncel", "madde_hedef_kitle": {},
                      "hedef_kitle": audience(entry["baslik"] + " " + text), "site": host,
                      "yururluk_tarihi": entry.get("yururluk_tarihi", ""), "yururluk_bitis": ""})
        entry["yapi"] = "" if len(re.findall(r"(?im)^\s*(?:MADDE|Madde)\s+\d+", text)) >= 3 else "serbest"
        catalog.append(entry)

    # ---- Sayfalar (yeni ve eski siteler)
    for path in pages:
        head, text = body_of(path.read_text(encoding="utf-8", errors="ignore"))
        host = urlparse(head.get("KAYNAK", "")).netloc.lower()
        title = head.get("BAŞLIK", "").strip() or path.stem.split("__")[-1].replace("-", " ")
        if admit(text):
            add({"ad": f"webt_{path.stem[:90]}", "kisa": f"{site_title(host)} – {title}"[:160], "baslik": title,
                 "dosya": f"web_tam/{path.parent.name}/{path.name}", "kaynak_url": head.get("KAYNAK", "")}, host, text)
    print(f"Sayfa: {len(catalog)}", flush=True)

    # ---- Duyurular: işlem içerenler; etkinlik/tören duyuruları değil
    before = len(catalog)
    for path in announcements:
        head, text = body_of(path.read_text(encoding="utf-8", errors="ignore"))
        title = head.get("BAŞLIK", "")
        if EVENT_RE.search(title) and not PROCEDURE_RE.search(title):
            reasons["etkinlik duyurusu"] += 1
            continue
        if not PROCEDURE_RE.search(title + " " + text[:1500]):
            reasons["işlem içermeyen duyuru"] += 1
            continue
        host = urlparse(head.get("KAYNAK", "")).netloc.lower()
        date = (head.get("YAYIN", "")[:10] or "").strip()
        if admit(text if len(text) >= 300 else text + " " * 300):
            add({"ad": f"webd_{path.stem[:90]}", "kisa": f"{site_title(host)} duyurusu – {title} ({date})"[:180],
                 "baslik": title, "tur": "duyuru", "hiyerarsi": 6, "yururluk_tarihi": date,
                 "dosya": f"web_tam/duyurular/{path.name}", "kaynak_url": head.get("KAYNAK", "")}, host, text)
    print(f"Duyuru: {len(catalog) - before}", flush=True)

    # ---- İndirilen dosyalar
    before = len(catalog)
    code_host = {info["kod"]: host for host, info in meta.items() if isinstance(info, dict) and info.get("kod")}
    def file_host(name: str) -> str:
        url, page, _ = file_meta.get(name, ("", "", ""))
        host = urlparse(page or url).netloc.lower() or "www.kocaeli.edu.tr"
        if host == "api.kocaeli.edu.tr":
            # Bulunduğu sayfa bilinmiyor: filePath'in başındaki site kodu (ör. 1084 = odb) siteyi verir
            code = re.search(r"filePath=(\d+)/", url)
            host = code_host.get(int(code.group(1)), "www.kocaeli.edu.tr") if code else "www.kocaeli.edu.tr"
        return host

    # Kopyalarda merkezi sitenin sürümü kalsın: 2026-2027 genel akademik takvimi İlahiyat sitesindeki kopyası
    # yüzünden "İlahiyat Fakültesi – Akademik Takvim" olarak kalıyor, fakülteye özel sanılıp geriye düşüyordu.
    texts.sort(key=lambda p: file_host(p.name[:-4]).split(".")[0] not in CENTRAL_SITES)  # kararlı: tarih sırası korunur
    for path in texts:
        name = path.name[:-4]
        raw = path.read_text(encoding="utf-8", errors="ignore")
        text = re.sub(r"__PAGE_\d+__", " ", raw)
        url, page, label = file_meta.get(name, ("", "", ""))
        host = file_host(name)
        if not admit(text):
            continue
        title = label.strip()
        if not title or GENERIC_LABEL_RE.match(title) or len(title) < 6:
            title = pdf_title(text)
            if not title or garbled_title(title):  # dosya adından başlık ("kou gorevlendirme yonergesi")
                title = re.sub(r"^\d{8}_[0-9a-f]{5}_|-[0-9a-f]{3}$", "", Path(name).stem).replace("-", " ").replace("_", " ")
        add({"ad": f"webf_{re.sub(r'[^a-z0-9]+', '_', Path(name).stem.lower())[:80]}", "kisa": f"{site_title(host)} – {title}"[:160],
             "baslik": title, "dosya": f"web_tam/metin/{path.name}", "kaynak_url": url}, host, text)
    print(f"Dosya: {len(catalog) - before}", flush=True)

    # ---- AKTS program bilgileri (ana) ve ders dosyaları (ayrı katalog)
    before = len(catalog)
    courses = []
    akts = WEB / "akts"
    programs = {p["program_dosya"]: p for p in json.loads((akts / "programlar.json").read_text(encoding="utf-8"))}
    for path in sorted(akts.glob("program__*.txt")):
        head, text = body_of(path.read_text(encoding="utf-8", errors="ignore"))
        info = programs.get(path.name, {})
        akts_title = head.get("BAŞLIK", path.stem).replace("i̇", "i")  # str.title() kalıntısı
        add({"ad": f"akts_{path.stem}", "kisa": f"AKTS Bilgi Paketi – {akts_title}"[:160],
             "baslik": head.get("BAŞLIK", ""), "tur": "program bilgisi", "hiyerarsi": 5,
             "dosya": f"web/akts/{path.name}", "kaynak_url": head.get("KAYNAK", "")}, "ects.kocaeli.edu.tr", text)
        catalog[-1]["yapi"] = "serbest"
        catalog[-1]["akts_birim"] = info.get("birim", "")
    for path in sorted(akts.glob("dersler__*.txt")):
        head, _ = body_of(path.read_text(encoding="utf-8", errors="ignore")[:500])
        courses.append({"ad": f"aktsders_{path.stem}", "kisa": head.get("BAŞLIK", path.stem).replace("i̇", "i")[:160],
                        "baslik": head.get("BAŞLIK", ""), "dosya": f"web/akts/{path.name}",
                        "kaynak_url": head.get("KAYNAK", "")})
    print(f"AKTS program: {len(catalog) - before}, ders dosyası: {len(courses)}", flush=True)

    # Belge kimlikleri benzersiz olmalı: aynı adlı .pdf/.docx ya da kısaltılmış uzun adlar çakışıyordu
    # (ChromaDB DuplicateIDError).
    used: dict[str, int] = {}
    for entry in catalog:
        base = entry["ad"]
        used[base] = used.get(base, 0) + 1
        if used[base] > 1:
            entry["ad"] = f"{base}_{used[base]}"
    (WEB / "katalog_tam.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    (WEB / "dersler_katalog.json").write_text(json.dumps(courses, ensure_ascii=False, indent=1), encoding="utf-8")
    (WEB / "birimler_ek.json").write_text(json.dumps(extra_units, ensure_ascii=False, indent=1), encoding="utf-8")
    old = WEB / "katalog_web.json"
    if old.exists():  # yerini katalog_tam.json aldı
        old.rename(WEB / "katalog_web.eski")
    by_kind: dict[str, int] = defaultdict(int)
    for entry in catalog:
        by_kind[entry["ad"].split("_")[0]] += 1
    print(f"Katalog: {len(catalog)} belge {dict(by_kind)}\nElenen: {dict(reasons)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
