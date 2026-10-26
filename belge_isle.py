#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""veri/belgeler.json'daki belgeleri işler: metin, değişiklikler, madde/fıkra parçaları, ChromaDB.

Akış:
  1) Her belgenin metni okunur (yanında elle düzeltilmiş .txt varsa o kullanılır).
  2) Maddelere ve fıkralara ayrılır.
  3) veri/degisiklikler.json'daki değişiklikler ilgili maddelere işlenir (değişiklik belgeleri ayrıca
     indekslenmez; aksi halde eski ve yeni hüküm yan yana aranır).
  4) Fıkra parçaları (arama birimi) ve madde metinleri (LLM bağlam birimi) yazılır, ChromaDB'ye yüklenir.

Çıktılar (veri/islenmis/):
  maddeler.jsonl  -> madde metinleri (LLM'e verilen bağlam; veri üretimi de bunu kullanır)
  parcalar.jsonl  -> fıkra + eşanlamlı parçaları (embedding/arama birimi)
  rapor.txt       -> belge başına madde/parça sayısı ve uyarılar
ChromaDB: ~/kou_rag_chroma (KOU_CHROMA_DIR ile değiştirilebilir), koleksiyon "kou_mevzuat".

Kullanım:
    python -X utf8 belge_isle.py --sadece-json   # sadece JSONL + rapor (Ollama gerekmez)
    python -X utf8 belge_isle.py                 # + embedding + ChromaDB
"""

from __future__ import annotations

import json
import os
import shutil
import re
import sys
import time
from pathlib import Path
from typing import Any

from pdf_to_chromadb import (
    BATCH_SIZE,
    CHUNK_OVERLAP_CHARS,
    EMBEDDING_MODEL,
    FIKRA_RE,
    HEADING_TAG,
    MAX_CHUNK_CHARS,
    SECTION_TAG,
    STUDENT_ALIASES as SINGLE_DOC_ALIASES,
    clean_chunk_text,
    normalize_for_matching,
    normalize_raw_text,
    page_at,
    page_markers,
    pdf_raw_text,
    split_long_text,
)

# ------------------------- AYARLAR -------------------------
VERI = Path(__file__).resolve().parent.parent / "veri"
CATALOG = VERI / "belgeler.json"
WEB_CATALOG = VERI / "web" / "katalog_web.json"  # web katalogları bu klasörde: katalog_*.json (varsa eklenir)
CHANGES = VERI / "degisiklikler.json"
OUT_DIR = VERI / "islenmis"
ARTICLES_JSONL = OUT_DIR / "maddeler.jsonl"
CHUNKS_JSONL = OUT_DIR / "parcalar.jsonl"
REPORT = OUT_DIR / "rapor.txt"
TOPICS_JSON = OUT_DIR / "konu_etiketleri.json"  # konu_etiketle.py çıktısı: madde_id -> konu
TOPIC_LABELS: dict[str, str] = (
    json.loads(TOPICS_JSON.read_text(encoding="utf-8")) if TOPICS_JSON.exists() else {}
)
# Tek konulu belgelerde modelin "diger" dediği maddeler belgenin konusunu alır (KDV tebliğinin
# tamamı tevkifatla ilgili; model üniversiteyle ilgisiz vergi bölümlerini "diger" saymıştı).
DOC_TOPIC_FALLBACK = {"kdv_genel_tebligi_117_tevkifat": "satin_alma_ihale"}
# Metni yalnızca "(Mülga: ...)" / "(İptal: ...)" notundan ibaret maddeler: aranacak içerik yok.
REPEAL_NOTE_RE = re.compile(r"\((?:Ek|Mülga|Değişik|İptal)[^)]*\)", re.IGNORECASE)


def topic_for(belge_id: str, madde_id: str) -> str:
    topic = TOPIC_LABELS.get(madde_id, "")
    return DOC_TOPIC_FALLBACK.get(belge_id, topic) if topic == "diger" else topic


def is_repealed_only(article: dict[str, Any]) -> bool:
    text = " ".join([article["preamble"]] + [t for _, t in article["paragraphs"]])
    if not re.search(r"Mülga|İptal", text, re.IGNORECASE):
        return False
    remaining = re.sub(r"[\s.,;:–\-…()\d]+", " ", REPEAL_NOTE_RE.sub("", text)).strip()
    return len(remaining) < 25
# ChromaDB'nin indeks katmanı Windows'ta ASCII dışı karakter içeren yolları ("KOÜ_RAG") büyük
# koleksiyonlarda okuyamıyor ("Error loading hnsw index"); veritabanı bu yüzden ev klasöründe durur.
CHROMA_PATH = Path(os.environ.get("KOU_CHROMA_DIR", Path.home() / "kou_rag_chroma"))
COLLECTION_NAME = "kou_mevzuat"
# Bundan uzun maddeler fıkra/bent gruplarına bölünür: LLM bağlamına 4 madde sığmalı
# (ör. 2547/54 disiplin maddesi tek başına ~18 bin karakter).
MAX_ARTICLE_CHARS = 3500
# Kılavuz gibi madde yapısı olmayan belgelerde atlanacak satırlar (antet, sayfa no, birim kısaltması).
GUIDE_NOISE_RE = re.compile(r"^(T\.C\.|KOCAELİ ÜNİVERSİTESİ|Ö\.İ\.D\.B\.|\d{1,2})$")
GUIDE_ITEM_RE = re.compile(r"^(\d+)\)\s*")
BENT_SPLIT_RE = re.compile(r"\s(?=[a-zçğıöşü]\) [A-ZÇĞİÖŞÜ])")

# PDF fontu/OCR kaynaklı bozulmalar; normalizasyondan ÖNCE ham metne uygulanır.
TEXT_FIXES: list[tuple[re.Pattern[str], str]] = [
    (re.compile("൴"), "i"),  # bazı PDF fontları 'i'yi bu karakterle veriyor
    (re.compile(r"(?im)^([ \t]*(?:GEÇİCİ[ \t]+|EK[ \t]+)?MADDE[ \t]+)l(\d*)"), r"\g<1>1\2"),  # OCR: "MADDE l5"
    (re.compile(r"(?im)^([ \t]*(?:GEÇİCİ[ \t]+|EK[ \t]+)?MADDE)-(\d+)"), r"\1 \2"),  # yazım: "MADDE-9-"
    # Satır sonunda biten "Madde 3-": tire, satır sonu kelime bölmesi sanılıp sonraki satırla birleşiyordu
    # ("Madde 3a) ..."). Farklı tire karakterleri ("MADDE 5 ‒") de en dash olur.
    (re.compile(r"(?im)^([ \t]*(?:GEÇİCİ[ \t]+|EK[ \t]+)?MADDE[ \t]+\d+)[ \t]*-[ \t]*$"), r"\1 –"),
    (re.compile(r"(?im)^([ \t]*(?:GEÇİCİ[ \t]+|EK[ \t]+)?MADDE[ \t]+\d+)[ \t]*[‒‐‑]"), r"\1 –"),
    (re.compile(r"(?im)^([ \t]*(?:GEÇİCİ[ \t]+|EK[ \t]+)?MADDE[ \t]+\d+)[ \t]*,"), r"\1-"),  # yazım: "MADDE 10,"
    (re.compile(r"(?im)^[ \t]*Sayfa[ \t]*:?[ \t]*\d+[ \t]*(?:/[ \t]*\d+)?[ \t]*$"), ""),  # sayfa altlığı "Sayfa 1 / 2"
    # Kanunlarda madde başlığı iki noktayla biter ("Öğrencilerin disiplin işleri:" + "Madde 54 –");
    # iki nokta kalırsa başlık, cümle sonu sanılıp önceki maddeye yapışıyor.
    (re.compile(r"(?m)^([ \t]*[A-ZÇĞİÖŞÜ][^\n:]{2,80}):[ \t]*\n(?=(?:[ \t]*\n)*[ \t]*"
                r"(?:(?:GEÇİCİ|Geçici|EK|Ek)[ \t]+)?(?:MADDE|Madde)[ \t]+\d+)"), r"\1\n"),
]


BARE_ARTICLE_LINE_RE = re.compile(r"(?im)^([ \t]*(?:GEÇİCİ[ \t]+|EK[ \t]+)?MADDE[ \t]+\d+)[ \t]*$")


# Türkçe karakterleri yanlış kod sayfasıyla gömülmüş PDF'ler: "Kocaelġ ünġversġtesġ", "ĠġLETME" (i/İ/ş/Ş).
FONT_GLYPH_MAP = str.maketrans({"ġ": "i", "Ġ": "İ", "ģ": "ş", "Ģ": "Ş"})


def fix_font_glyphs(raw: str) -> str:
    return raw.translate(FONT_GLYPH_MAP)


def strip_symbol_glyphs(raw: str) -> str:
    """Symbol/Wingdings madde imleri özel kullanım alanında gelir (ör. U+F076 "❖"); başlığa yapışıp kaynak satırını bozuyordu."""
    return fix_font_glyphs(re.sub("[\ue000-\uf8ff]", "", raw.replace("\uf020", " ")))


def is_form_like(article: dict[str, Any]) -> bool:
    """Web belgelerinde doldurulacak form/tablo sayfası ("Tarih: Yapılan İş: İmza:") bağlam birimi olmaz."""
    text = " ".join([article["preamble"]] + [p[1] for p in article["paragraphs"]])
    words = re.findall(r"[^\W\d_]{2,}", text)
    return len(words) < 15 or text.count(":") > max(4, len(words) / 6)


# Kişisel liste bölümü (maskeli TC/ad, öğrenci numarası, "Ad SOYAD" dizisi): kurallı bir duyurunun sonuna eklenmiş
# asil/yedek listesi gibi. Belge kalır, yalnız liste bölümü atlanır (23.10; kullanıcı: kişisel listeler olmasın).
MASKED_ID_RE = re.compile(r"\b\d{2,3}\*{3,}\d{0,3}\b|\b[A-ZÇĞİÖŞÜ]{1,3}\*{2,}")
STUDENT_NO_RE = re.compile(r"\b[12]\d{8}\b")


def is_person_list(article: dict[str, Any]) -> bool:
    """Ad soyad deseni ("Ahmet YILMAZ") kullanılmaz: "Fakültesi TÜBİTAK", "Ders II" gibi yanlış eşleşme çok."""
    text = " ".join([article["preamble"]] + [p[1] for p in article["paragraphs"]])
    return len(MASKED_ID_RE.findall(text)) >= 3 or len(STUDENT_NO_RE.findall(text)) >= 3


# Belge sonundaki ekler ("EK-1 DİĞER HİZMETLER KARŞILIĞI ... CETVELİ", "EK-1 LİSANS DİPLOMA ÖRNEĞİ"):
# son maddeye yapışmasın. "EK MADDE 1" etiketi bununla eşleşmez (rakamdan önce tire şart).
LAW_APPENDIX_MARKERS = ["DÖNEMİNDE UYGULANAN EŞİK DEĞERLER", "EK VE DEĞİŞİKLİK GETİREN"]
APPENDIX_RE = re.compile(r"\s(?:EK|Ek)-\d+\s+[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ ]{5,}")
# Kılavuzlardaki yatay geçiş kontenjan tablosu satırı:
# "25 Mühendislik Fakültesi Mekatronik Mühendisliği* SAY 1 1 0 0" (ön lisansta 2 sayı).
QUOTA_ROW_RE = re.compile(
    r"^(\d+) (.+?(?:Fakültesi|Yüksekokulu|Konservatuvarı)) (.+?) (SAY|EA|SÖZ|DİL|TYT) ((?:\d+ ?){2,4})$"
)

# Satır başındaki madde etiketi (bazı esaslarda "ESAS 1-"); ardından tire/iki nokta/nokta veya "("
# gelmeli. Böylece satır başına denk gelen "Madde 16/4 çerçevesinde" gibi atıflar yeni madde sanılmaz.
ARTICLE_RE = re.compile(
    r"(?im)^[ \t]*(?P<label>(?:(?:GEÇİCİ|EK)[ \t]+)?(?:MADDE|ESAS)[ \t]+(?P<num>\d+)(?:/[A-ZÇĞİÖŞÜ]|[A-Z])?)"
    r"[ \t]*(?:[–—\-:.][ \t]*|(?=\())"
)
HEADING_BEFORE_RE = re.compile(rf"{HEADING_TAG} ([^\n]+)\n(?:__PAGE_\d+__\n)*$")
SECTION_LINE_RE = re.compile(rf"{SECTION_TAG} ([^\n]+)")

# Öğrencilerin kullandığı ifadeler (sadece arama + bağlamda gösterim). Belge adı -> madde -> ifadeler.
STUDENT_ALIASES: dict[str, dict[str, list[str]]] = {
    "kou_onlisans_lisans_egitim_ogretim_yonetmeligi_2019":
        SINGLE_DOC_ALIASES["kouonlisansvelisansegitimogretimyonetmeligi.pdf"],
    "yok_gecis_cift_anadal_yandal_kredi_transferi_yonetmeligi": {
        "MADDE 18": ["yandal", "yan dal", "yandal başvurusu", "yandal şartları", "yandal sertifikası"],
        "MADDE 16": ["ÇAP şartları", "çift anadal için ortalama kaç olmalı"],
        "MADDE 9": ["kurum içi yatay geçiş", "kendi üniversitemde bölüm değiştirmek"],
        "MADDE 11": ["kurumlar arası yatay geçiş", "başka üniversiteye geçmek"],
        "EK MADDE 1": ["puanla yatay geçiş", "merkezi yerleştirme puanıyla yatay geçiş", "ek madde 1 ile geçiş"],
        "MADDE 14": ["yurt dışından yatay geçiş"],
    },
    "kou_basari_notu_degerlendirme_usul_esaslar": {
        "MADDE 10": ["çan eğrisi", "bağıl değerlendirme", "sınıf ortalamasına göre not"],
        "MADDE 9": ["mutlak değerlendirme"],
        "MADDE 8": ["final barajı", "finalde en az kaç almalıyım", "bütünleme alt sınırı"],
        "MADDE 11": ["vize final yüzdesi", "harf notu nasıl hesaplanır"],
        "MADDE 7": ["devamsız öğrencinin notu"],
    },
    "kou_yuzde_on_basari_degerlendirme_yonergesi": {
        "MADDE 4": ["yüzde on", "%10'a girmek", "ilk yüzde on", "sınıf sıralaması"],
        "MADDE 6": ["yüzde on listesine kimler giremez"],
    },
    "yok_uygulamali_egitimler_cerceve_yonetmeligi": {
        "MADDE 13": ["staj", "zorunlu staj", "staj kaç gün", "staj yeri"],
        "MADDE 12": ["işletmede mesleki eğitim", "İME", "işyeri eğitimi"],
        "MADDE 18": ["yurt dışında staj"],
        "MADDE 19": ["staj muafiyeti", "önceki stajın sayılması"],
        "GEÇİCİ MADDE 1": ["staj ücreti", "stajyer maaşı", "stajda devlet katkısı"],
    },
    "kou_yabanci_diller_yuksekokulu_egitim_sinav_yonergesi": {
        "MADDE 5": ["hazırlık", "İngilizce hazırlık", "hazırlık sınıfı"],
        "MADDE 10": ["proficiency", "hazırlık muafiyet sınavı", "yeterlik sınavı"],
        "MADDE 11": ["YDS ile hazırlıktan muafiyet", "TOEFL ile muafiyet"],
        "MADDE 14": ["hazırlıkta kalmak", "hazırlığı geçemezsem"],
        "MADDE 16": ["hazırlıkta kayıt dondurma"],
        "MADDE 9": ["hazırlık devamsızlık"],
    },
    "kou_ozel_ogrenci_yonergesi": {
        "MADDE 5": ["başka üniversitede özel öğrenci olmak", "özel öğrenci olarak ders almak"],
    },
    "kou_cift_anadal_programi_yonetmeligi": {
        "MADDE 6": ["ÇAP başvuru şartları", "çift anadal ortalama şartı"],
        "MADDE 8": ["ÇAP'tan çıkarılmak"],
        "MADDE 12": ["ÇAP ücreti"],
    },
    "kou_yaz_ogretimi_esaslari": {
        "ESAS 1": ["yaz okulu", "başka üniversitede yaz okulu", "yaz okulu taban puan şartı"],
        "ESAS 2": ["yaz okulunda kaç ders alabilirim", "yaz okulu ders sınırı"],
        "ESAS 3": ["kayıt dondurduğum dönemin dersini yaz okulunda almak"],
        "ESAS 7": ["yaz okulunda geçme notu"],
        "ESAS 10": ["yaz okulu sonrası tek ders sınavı"],
    },
    "kou_basariya_dayali_yatay_gecis_kilavuzu_2026_2027": {
        "KISIM 1": ["yatay geçiş başvuru tarihleri", "yatay geçiş takvimi", "yatay geçiş başvurusu ne zaman",
                    "yatay geçiş başvuru adresi", "yatay geçiş sonuçları ne zaman açıklanır"],
        "KISIM 2": ["kurum içi yatay geçiş şartları", "yatay geçiş için ortalama kaç olmalı",
                    "yatay geçiş değerlendirme puanı nasıl hesaplanır"],
        "KISIM 5": ["yatay geçiş için gerekli belgeler", "yatay geçişte hangi belgeler yüklenir"],
    },
    "kou_muafiyet_ve_intibak_yonergesi": {
        "MADDE 6": ["muafiyet başvurusu", "ders muafiyeti ne zaman"],
        "MADDE 10": ["intibak", "üst sınıfa intibak", "muaf olunan dersin notu"],
    },
    "kanun_2547_yuksekogretim": {
        "MADDE 39": ["yurt dışı görevlendirme", "kongre için görevlendirme", "39. madde görevlendirme"],
        "MADDE 24": ["doçentlik başvurusu", "doçentlik şartları"],
        "MADDE 26": ["profesörlüğe atama"],
        "MADDE 36": ["öğretim elemanı çalışma esasları", "dışarıda çalışma izni"],
        "MADDE 54": ["disiplin cezası", "kopya çekersem ne olur", "intihal cezası", "uzaklaştırma cezası",
                     "okuldan atılma", "kınama cezası", "disiplin soruşturması", "disiplin cezasına itiraz"],
    },
    # ---- Akademik personel ----
    "kou_ders_gorevlendirme_ek_ders_yonergesi": {
        "MADDE 10": ["ders yükü", "zorunlu ders yükü", "haftalık ders yükü kaç saat"],
        "MADDE 11": ["ek ders ücreti", "ek ders hesabı", "ek ders nasıl hesaplanır"],
        "MADDE 12": ["sınav ücreti", "sınav görevi ücreti"],
        "MADDE 8": ["başka üniversitede ders vermek"],
    },
    "kanun_6245_harcirah": {
        "MADDE 14": ["geçici görev yolluğu", "görev yolluğu", "kongre harcırahı"],
        "MADDE 27": ["yol masrafı", "yurt içi yol gideri"],
        "MADDE 33": ["yurt içi gündelik", "harcırah gündeliği"],
        "MADDE 34": ["yurt dışı gündelik", "yurt dışı harcırah"],
    },
    "kou_bap_uygulama_yonergesi": {
        "MADDE 9": ["BAP proje türleri"],
        "MADDE 10": ["BAP başvurusu", "proje başvurusu nasıl yapılır"],
        "MADDE 13": ["BAP ara raporu"],
        "MADDE 14": ["BAP sonuç raporu"],
        "MADDE 17": ["proje süresi uzatma", "projeye ek süre"],
        "MADDE 18": ["proje bütçesi", "BAP bütçe limiti"],
        "MADDE 19": ["bursiyer", "projeden burs almak"],
    },
    "kou_lisansustu_egitim_ogretim_yonetmeligi_2023": {
        "MADDE 33": ["doktora yeterlik sınavı"],
        "MADDE 34": ["TİK", "tez izleme komitesi"],
        "MADDE 35": ["tez önerisi"],
        "MADDE 22": ["yüksek lisans azami süre", "yüksek lisans kaç yılda bitmeli"],
        "MADDE 31": ["doktora azami süre", "doktora kaç yıl sürer"],
    },
}


def tr_upper(text: str) -> str:
    return text.replace("i", "İ").replace("ı", "I").upper()


def read_raw_text(entry: dict[str, Any]) -> str:
    """Satır yapısı korunmuş ham metin (serbest metin ayrıştırıcısı için): .txt varsa o, yoksa PDF."""
    pdf_path = VERI / entry["dosya"]
    override = pdf_path.with_suffix(".txt")
    raw = override.read_text(encoding="utf-8") if override.exists() else pdf_raw_text(pdf_path)
    if raw.startswith("KAYNAK:"):
        # web_tara.py sayfası: kaynak/erişim satırları metin değil; sayfa başlığı ilk bölümün başlığı olur.
        head, _, raw = raw.partition("\n\n")
        title = re.search(r"BAŞLIK: (.*)", head)
        raw = f"\n{tr_upper(title.group(1).strip())}\n\n{raw}" if title else raw
    for pattern, replacement in TEXT_FIXES[:1]:  # yalnızca karakter düzeltmesi; diğerleri MADDE satırları için
        raw = pattern.sub(replacement, raw)
    return strip_symbol_glyphs(raw)


def read_document_text(entry: dict[str, Any]) -> str:
    pdf_path = VERI / entry["dosya"]
    override = pdf_path.with_suffix(".txt")
    raw = override.read_text(encoding="utf-8") if override.exists() else pdf_raw_text(pdf_path)
    if entry["tur"] == "kanun":
        # mevzuat.gov.tr'nin işlenmiş kanun metinlerinin sonuna eklediği tablolar kanun metni değildir;
        # satırlarındaki "Madde 44", "MADDE 62 (h)" sahte madde üretiyordu. İlk tablonun sayfasından kesilir.
        found = [i for i in (raw.find(marker) for marker in LAW_APPENDIX_MARKERS) if i >= 0]
        if found:
            page_start = raw.rfind("__PAGE_", 0, min(found))
            raw = raw[:page_start if page_start > 0 else min(found)]
    for wrong, right in entry.get("metin_duzeltmeleri", []):  # belgeye özel OCR düzeltmeleri (katalogda)
        raw = raw.replace(wrong, right)
    for pattern, replacement in TEXT_FIXES:
        raw = pattern.sub(replacement, raw)
    if entry["tur"] != "kanun":
        # Tiresiz etiket satırı ("MADDE 1", Teknoloji/Turizm İME yönergeleri). Kanunlarda değil: sondaki
        # eşik değer tablosunun "MADDE 8" satırı sahte madde oluyordu.
        raw = BARE_ARTICLE_LINE_RE.sub(r"\1 –", raw)
    return normalize_raw_text(strip_symbol_glyphs(raw))


def parse_articles(text: str) -> list[dict[str, Any]]:
    """Normalize metni madde listesine çevirir: etiket, başlık, bölüm, giriş, fıkralar, sayfalar."""
    positions, pages = page_markers(text)
    sections = [(m.start(), m.group(1).strip()) for m in SECTION_LINE_RE.finditer(text)]
    matches = list(ARTICLE_RE.finditer(text))
    articles: list[dict[str, Any]] = []
    current_section = ""
    for index, match in enumerate(matches):
        region_start = matches[index - 1].end() if index else 0
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        heading_match = HEADING_BEFORE_RE.search(text, region_start, match.start())
        heading = heading_match.group(1).strip() if heading_match else ""
        heading = re.sub(r":\s*\d*$", "", heading)  # kanunlarda dipnot no yapışıyor: "Döner sermaye:6869"
        new_sections = [s for pos, s in sections if region_start <= pos < match.start()]
        if new_sections:
            current_section = new_sections[0] + (" – " + " ".join(new_sections[1:]) if len(new_sections) > 1 else "")

        body = text[match.end():end]
        appendix = APPENDIX_RE.search(body)
        if appendix:
            body = body[:appendix.start()]
        fikra_matches = list(FIKRA_RE.finditer(body))
        preamble = clean_chunk_text(body[: fikra_matches[0].start()] if fikra_matches else body)
        paragraphs = []
        for i, fikra in enumerate(fikra_matches):
            fikra_end = fikra_matches[i + 1].start() if i + 1 < len(fikra_matches) else len(body)
            content = clean_chunk_text(body[fikra.end():fikra_end])
            if content:
                paragraphs.append([int(fikra.group(1)), content])
        # "MADDE 1 - Amaç\n(1) ..." biçimi: başlık etiketten sonra gelir.
        if not heading and fikra_matches and preamble and len(preamble) <= 80 and not preamble.endswith("."):
            heading, preamble = preamble, ""

        articles.append({
            "label": tr_upper(re.sub(r"\s+", " ", match.group("label")).strip()),
            "number": int(match.group("num")),
            "heading": heading,
            "section": current_section,
            "preamble": preamble,
            "paragraphs": paragraphs,
            "start_page": page_at(match.start(), positions, pages),
            "end_page": page_at(max(match.start(), end - 1), positions, pages),
            "degisiklik": "",
        })
    return articles


def is_guide_heading(line: str) -> bool:
    """Kılavuzlarda BÜYÜK HARFLİ bölüm başlığı; rakamlı tablo satırları ("YDS 55 80 90") hariç."""
    letters = [c for c in line if c.isalpha()]
    digits = sum(c.isdigit() for c in line)
    return (
        len(line) > 12 and len(letters) >= 10
        and sum(c.isupper() for c in letters) / len(letters) > 0.85
        and digits <= 9 and not GUIDE_ITEM_RE.match(line)
        and len(re.findall(r"\b\d+\b", line)) < 3
    )


def sentence_case(text: str) -> str:
    lowered = text.replace("İ", "i").replace("I", "ı").lower()
    return lowered[:1].replace("i", "İ").upper() + lowered[1:]


def quota_table_articles(raw_text: str, first_number: int) -> tuple[str, list[dict[str, Any]]]:
    """Kontenjan tablosunu metinden ayırır; her birim (fakülte/MYO) için okunur cümlelerden bir bölüm üretir.

    Düz metne dökülen tabloda sütunlar karışır ("SAY 1 1 0 0"); model hangi sayının hangi yarıyıla ait
    olduğunu bilemez. Dönüş: (tablo çıkarılmış metin, birim bölümleri).
    """
    title = raw_text.find("KONTENJANLARI")
    if title < 0:
        return raw_text, []
    # Tablo başlığı iki satır yukarıdan başlıyor ("2026-2027 GÜZ YARIYILI BAŞARIYA DAYALI 3. Yarıyıl ...").
    start = raw_text.rfind("\n", 0, raw_text.rfind("\n", 0, title))
    by_unit: dict[str, list[str]] = {}
    for line in raw_text[start:].split("\n"):
        row = QUOTA_ROW_RE.match(re.sub(r"\s+", " ", line).strip())
        if not row:
            continue
        _, unit, program, score_type, numbers = row.groups()
        values = numbers.split()
        prep = ", zorunlu hazırlık" if program.endswith("*") else ""
        program = program.rstrip("*").strip()
        text = (f"{program} ({score_type} puan türü{prep}): 3. yarıyıl için birim içi {values[0]}, "
                f"birimler arası {values[1]} kontenjan")
        if len(values) == 4:
            text += f"; 5. yarıyıl için birim içi {values[2]}, birimler arası {values[3]} kontenjan"
        by_unit.setdefault(unit, []).append(text + ".")
    articles = []
    for offset, (unit, rows) in enumerate(by_unit.items()):
        articles.append({
            "label": f"KISIM {first_number + offset}", "number": first_number + offset,
            "heading": f"Kurum içi yatay geçiş kontenjanları – {unit}", "section": "",
            "preamble": "", "paragraphs": [[i, row] for i, row in enumerate(rows, start=1)],
            "start_page": 0, "end_page": 0, "degisiklik": "",
        })
    return raw_text[:start], articles


def parse_guide(raw_text: str) -> list[dict[str, Any]]:
    """Madde yapısı olmayan kılavuzlar: BÜYÜK HARFLİ başlık = bölüm (KISIM n), "1)" satırları = madde."""
    raw_text, quota_articles = quota_table_articles(raw_text, first_number=1000)
    positions, pages = page_markers(raw_text)
    articles: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    heading_buffer: list[str] = []
    offset = 0
    for raw_line in raw_text.split("\n"):
        line_start = offset
        offset += len(raw_line) + 1
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or GUIDE_NOISE_RE.match(line) or line.startswith("__PAGE_"):
            continue
        if is_guide_heading(line):
            heading_buffer.append(line)
            continue
        if heading_buffer:  # birikmiş başlık satırları yeni bir bölüm açar
            current = {
                "label": f"KISIM {len(articles) + 1}", "number": len(articles) + 1,
                "heading": sentence_case(" ".join(heading_buffer).strip(" -–")), "section": "", "preamble": "",
                "paragraphs": [], "start_page": page_at(line_start, positions, pages),
                "end_page": page_at(line_start, positions, pages), "degisiklik": "",
            }
            articles.append(current)
            heading_buffer = []
        if current is None:
            continue
        current["end_page"] = page_at(line_start, positions, pages)
        item = GUIDE_ITEM_RE.match(line)
        if item:
            current["paragraphs"].append([int(item.group(1)), line[item.end():]])
        elif current["paragraphs"]:
            current["paragraphs"][-1][1] += " " + line
        else:
            current["preamble"] = (current["preamble"] + " " + line).strip()
    articles = [a for a in articles if a["paragraphs"] or a["preamble"]]
    # Kontenjan bölümleri metindeki bölümlerin ardından numaralanır.
    for offset, quota in enumerate(quota_articles, start=len(articles) + 1):
        quota.update(label=f"KISIM {offset}", number=offset)
    return articles + quota_articles


SECTION_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)\.\s+(.*)$")


def parse_numbered_sections(raw_text: str) -> list[dict[str, Any]]:
    """Madde yerine "3.2.1." biçiminde numaralı tebliğler (ör. KDV Genel Tebliği).

    BÜYÜK HARFLİ numaralı satır = başlık ("3.2. KISMİ TEVKİFAT UYGULANACAK HİZMETLER"); diğer numaralı
    satırlar = paragraf birimi, etiketi "BÖLÜM 3.2.1" (kaynak satırında tebliğin kendi numarası görünür).
    """
    raw_text = raw_text.split("-" * 40)[0]  # sondaki dipnot listesi
    positions, pages = page_markers(raw_text)
    titles: dict[str, str] = {}
    units: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    offset = 0
    for raw_line in raw_text.split("\n"):
        line_start = offset
        offset += len(raw_line) + 1
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or line.startswith("__PAGE_") or "Resmî Gazete" in line or "RESMÎ GAZETE" in line:
            continue
        numbered = SECTION_NUMBER_RE.match(line)
        if numbered:
            number, rest = numbered.groups()
            letters = [c for c in rest if c.isalpha()]
            is_title = letters and sum(c.isupper() for c in letters) / len(letters) > 0.8
            # En yakın başlıklı üst bölüm: 3.2.1.2.1 -> 3.2.1.2 -> 3.2.1 ("Yapım işleri ...").
            ancestors = [number.rsplit(".", depth)[0] for depth in range(1, number.count(".") + 1)]
            parent_title = next((titles[a] for a in ancestors if a in titles), "")
            current = {
                "label": f"BÖLÜM {number}", "number": len(units) + 1,
                "heading": sentence_case(rest) if is_title else parent_title,
                "section": titles.get(number.split(".")[0], ""),
                "preamble": "" if is_title else rest, "paragraphs": [],
                "start_page": page_at(line_start, positions, pages),
                "end_page": page_at(line_start, positions, pages), "degisiklik": "",
            }
            if is_title:
                titles[number] = sentence_case(rest)
            units.append(current)
        elif current is not None:
            current["preamble"] = (current["preamble"] + " " + line).strip()
            current["end_page"] = page_at(line_start, positions, pages)
    # Sadece başlıktan ibaret (altında metni olmayan) bölümler bağlam birimi olmaz.
    return [unit for unit in units if unit["preamble"]]


PLAIN_MAX_CHARS = 1800  # serbest metin bölümü bundan uzunsa satır sınırından bölünür
PLAIN_MIN_CHARS = 60
# tablo_metin.py bölümü (__TABLO__ ... __TABLO_SON__): satırlar kendi başına anlamlı ("YARIYIL SONU SINAVLARI
# (... 2026-2027 AKADEMİK TAKVİMİ) — Güz: ..."); 1800 karakterlik bölümlerde doğru satır aramada kayboluyordu.
TABLE_MAX_CHARS = 600  # bundan kısa bölüm (kapak sayfası, "Tarih: 10.03.2026") bağlam birimi olmaz
NUMBERED_HEADING_RE = re.compile(r"^\d+\s*[-–]\s")  # "1- ÖĞRENCİ ÖĞRENİM HAREKETLİLİĞİ ..."
ITEM_START_RE = re.compile(r"^(\d+[.)\-]|[a-zçğıöşü]\)|[•\-–*])")


def is_plain_heading(line: str, after_blank: bool) -> bool:
    """Serbest metinde bölüm başlığı: boş satırdan sonra gelen BÜYÜK HARFLİ kısa satır.

    Boş satır şartı, büyük harfle yazılmış kuralların devam satırlarını ("KİMLİK KARTLARI SINAV BOYUNCA ...")
    başlık sanmamak için; rakamla başlayan madde satırları ("2. ÖĞRENCİ KİMLİK ...") başlık değildir.
    """
    if not after_blank or len(line) > 110 or line.endswith((",", ";", ".")):
        return False
    numbered = NUMBERED_HEADING_RE.match(line)
    if not numbered and line[:1].isdigit():
        return False
    body = line[numbered.end():] if numbered else line
    letters = [c for c in body if c.isalpha()]
    return len(letters) >= 6 and sum(c.isupper() for c in letters) / len(letters) > 0.85


def end_table_row(line: str) -> str:
    """DOCX/PDF tablo satırı ("7-11 Eylül 2026 | Güz Yarıyılında Açılan Derslerin ÖBS'de İlanı") sona ayraç alır.
    Satırlar boşlukla birleşince bir sonraki satırın tarihi önceki olayın tarihi gibi okunuyordu: "... İlanı 11 Eylül
    2026 | Yeni Öğrencilerin ..." (23.10, Diş Hekimliği takvimi; hem model hem veri üretici yanlış tarih verdi)."""
    table_row = " | " in line or (" — " in line and ":" in line)  # tablo_metin satırı: "Olay — Tarih: 7 Eylül 2026"
    return f"{line};" if table_row and not line.endswith((".", ";", ":", ",")) else line


def parse_plain(raw_text: str) -> list[dict[str, Any]]:
    """Madde numarası olmayan belgeler (staj esasları, rehber, duyuru): başlıklara ve boyuta göre bölümler.

    Etiket "KISIM n" (kılavuzlardaki gibi): kaynak satırı "Belge – Bölüm başlığı" olur. Başlık yoksa ya da
    bölüm uzunsa, iki noktayla biten kısa satır ("Öğrencilerin ... Uygulamaları Gereken Esaslar:") ara başlık
    sayılır; o da yoksa bölüm PLAIN_MAX_CHARS'ta satır sınırından bölünür (başlık korunur).
    """
    positions, pages = page_markers(raw_text)
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    after_blank = True
    offset = 0
    in_table = False

    def open_section(heading: str, line_start: int) -> dict[str, Any]:
        page = page_at(line_start, positions, pages)
        section = {"heading": heading, "lines": [], "start_page": page, "end_page": page}
        sections.append(section)
        return section

    for raw_line in raw_text.split("\n"):
        line_start = offset
        offset += len(raw_line) + 1
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or line.startswith("__PAGE_"):
            after_blank = True
            continue
        if line in ("__TABLO__", "__TABLO_SON__"):
            in_table, after_blank = line == "__TABLO__", True
            current = None  # tablo bölümü kendi kısımlarında; düz metin yeni kısımla başlar
            continue
        if in_table:
            if line.startswith("[Tablo]"):
                current = open_section(line[len("[Tablo]"):].strip() or "Tablo", line_start)
                continue
            size = sum(len(text) for text in (current or {}).get("lines", []))
            if current is None or size + len(line) > TABLE_MAX_CHARS:
                current = open_section(current["heading"] if current else "Tablo", line_start)
            current["lines"].append(line)
            current["end_page"] = page_at(line_start, positions, pages)
            continue
        if GUIDE_NOISE_RE.match(line):
            continue
        # Henüz metni olmayan bölümün başlığı alt satırda sürebilir ("KOCAELİ MESLEK YÜKSEKOKULU" + "STAJ İŞLEMLERİ").
        heading_continues = current is not None and not current["lines"]
        if is_plain_heading(line, after_blank or heading_continues):
            if current is not None and not current["lines"]:  # art arda başlık satırları birleşir
                current["heading"] = f"{current['heading']} {line}"
            else:
                current = open_section(line, line_start)
            after_blank = False
            continue
        size = sum(len(text) for text in (current or {}).get("lines", []))
        colon_heading = (line.endswith(":") and len(line) <= 100 and line[:1].isupper()
                         and not ITEM_START_RE.match(line) and size >= 400)
        if current is None or colon_heading or size + len(line) > PLAIN_MAX_CHARS:
            if colon_heading:
                heading = line.rstrip(":")
                # Önceki satır noktalamasız kısa bir satırsa başlığın ilk yarısıdır ("... Gereken" + "Esaslar:").
                previous = current["lines"][-1] if current and current["lines"] else ""
                if previous and len(previous) <= 100 and not previous.endswith((".", ":", ";", ",")) \
                        and not ITEM_START_RE.match(previous):
                    heading = f"{current['lines'].pop()} {heading}"
            else:
                heading = current["heading"] if current else ""
            current = open_section(heading, line_start)
            if colon_heading:
                after_blank = False
                continue
        current["lines"].append(line)
        current["end_page"] = page_at(line_start, positions, pages)
        after_blank = False

    articles = []
    for section in sections:
        text = clean_chunk_text(" ".join(end_table_row(line) for line in section["lines"]))
        if len(text) < PLAIN_MIN_CHARS:
            continue
        number = len(articles) + 1
        articles.append({
            "label": f"KISIM {number}", "number": number,
            "heading": sentence_case(section["heading"].strip(" -–:")) if section["heading"] else "",
            "section": "", "preamble": text, "paragraphs": [],
            "start_page": section["start_page"], "end_page": section["end_page"], "degisiklik": "",
        })
    return articles


ORDINALS = ["BİRİNCİ", "İKİNCİ", "ÜÇÜNCÜ", "DÖRDÜNCÜ", "BEŞİNCİ", "ALTINCI", "YEDİNCİ", "SEKİZİNCİ"]


def prefix_restarted_numbering(articles: list[dict[str, Any]]) -> int:
    """Her bölümde 1'den yeniden başlayan madde numaraları (ör. Mimarlık ve Tasarım Fakültesi Staj Yönergesi):
    etikete bölüm eklenir ("İKİNCİ BÖLÜM MADDE 1"), yoksa iki ayrı "Madde 1" kaynak satırında ayırt edilemez.
    Dönüş: yeniden başlama sayısı (0 ise belgeye dokunulmaz)."""
    plain = [a for a in articles if re.fullmatch(r"MADDE \d+", a["label"])]
    restarts = [i for i in range(1, len(plain)) if plain[i]["number"] <= plain[i - 1]["number"]]
    if not restarts:
        return 0
    part = 0
    for index, article in enumerate(plain):
        if index in restarts:
            part += 1
        article["label"] = f"{ORDINALS[part]} BÖLÜM {article['label']}"
    return len(restarts)


def split_long_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """MAX_ARTICLE_CHARS'tan uzun maddeleri fıkra gruplarına, uzun fıkraları bent gruplarına böler.

    Parçalar aynı madde etiketini ve fıkra numarasını korur; kaynak gösterimi değişmez.
    """
    result: list[dict[str, Any]] = []
    for article in articles:
        size = len(article["preamble"]) + sum(len(text) for _, text in article["paragraphs"])
        if size <= MAX_ARTICLE_CHARS:
            result.append(article)
            continue
        if len(article["preamble"]) > MAX_ARTICLE_CHARS // 2:
            # Uzun giriş metni (fıkra numarası olmayan eski kanun maddeleri, kılavuzlardaki tablolar)
            # gruplamaya dahil edilir; fıkra no 0: kaynak satırında fıkra yazılmaz.
            pieces = split_long_text(article["preamble"], MAX_ARTICLE_CHARS, 0)
            article = {**article, "preamble": "", "paragraphs": [[0, p] for p in pieces] + article["paragraphs"]}
        groups: list[list[list]] = [[]]
        for number, text in article["paragraphs"]:
            pieces = [text]
            if len(text) > MAX_ARTICLE_CHARS:
                bents = BENT_SPLIT_RE.split(text)
                intro, pieces = bents[0], []
                for bent in bents[1:]:
                    if pieces and len(pieces[-1]) + len(bent) <= MAX_ARTICLE_CHARS:
                        pieces[-1] += " " + bent
                    else:
                        # Fıkranın giriş cümlesi her parçada tekrarlanır ("Disiplin cezaları ...:").
                        pieces.append(f"{intro} … {bent}" if len(intro) <= 300 else bent)
                pieces = pieces or [text]
                # Tek bir bent de çok uzunsa (ör. 2547/58 döner sermaye) cümle sınırlarından bölünür.
                pieces = [part for piece in pieces for part in (
                    split_long_text(piece, MAX_ARTICLE_CHARS, 0) if len(piece) > MAX_ARTICLE_CHARS else [piece])]
            for piece in pieces:
                current_size = sum(len(t) for _, t in groups[-1])
                if groups[-1] and current_size + len(piece) > MAX_ARTICLE_CHARS:
                    groups.append([])
                groups[-1].append([number, piece])
        for index, group in enumerate(groups, start=1):
            numbers = sorted({n for n, _ in group})
            first_text = group[0][1]
            # "Disiplin amirleri: ..." -> alt başlık; "(Değişik: RG ...)" gibi notlar başlık değildir.
            topic = re.match(r"^([A-ZÇĞİÖŞÜ][^:.;()]{2,80}):", first_text)
            if numbers == [0]:  # fıkra numarası olmayan eski kanun maddesi
                subtitle = f"bölüm {index}/{len(groups)}"
            elif topic and len(numbers) == 1:
                subtitle = topic.group(1)
            else:
                subtitle = f"fıkra {numbers[0]}" if len(numbers) == 1 else f"fıkra {numbers[0]}-{numbers[-1]}"
            result.append({
                **article,
                "heading": f"{article['heading']} – {subtitle}" if article["heading"] else subtitle,
                "preamble": article["preamble"] if index == 1 else "",
                "paragraphs": group,
                "id_suffix": f"_b{index}",
            })
    return result


def apply_changes(articles: list[dict[str, Any]], change: dict[str, Any], warnings: list[str]) -> None:
    by_label = {article["label"]: article for article in articles}

    def article_for(label: str) -> dict[str, Any] | None:
        article = by_label.get(label)
        if article is None:
            warnings.append(f"değişiklik uygulanamadı, madde yok: {label}")
        return article

    for op in change["islemler"]:
        kind = op["tur"]
        if kind == "bolum_basligi":
            for article in articles:
                if article["section"].startswith(op["bolum"]):
                    article["section"] = f"{op['bolum']} – {op['yeni']}"
            continue
        labels = op.get("maddeler") or [op["madde"]]
        for label in labels:
            article = article_for(label)
            if article is None:
                continue
            if kind == "madde_degistir":
                article["preamble"] = ""
                article["paragraphs"] = [[int(k), v] for k, v in op["fikralar"].items()]
            elif kind == "fikra_degistir":
                existing = {number for number, _ in article["paragraphs"]}
                for number, content in op["fikralar"].items():
                    if int(number) not in existing:
                        warnings.append(f"{label}: değiştirilecek fıkra yok ({number})")
                article["paragraphs"] = [
                    [n, op["fikralar"].get(str(n), content)] for n, content in article["paragraphs"]
                ]
            elif kind == "cumle_degistir":
                for paragraph in article["paragraphs"]:
                    if str(paragraph[0]) == op["fikra"]:
                        if paragraph[1].count(op["eski"]) != 1:
                            warnings.append(f"{label}/{op['fikra']}: değiştirilecek cümle bulunamadı")
                        paragraph[1] = paragraph[1].replace(op["eski"], op["yeni"])
            elif kind == "madde_kaldir":
                article["preamble"] = ""
                article["paragraphs"] = [[0, "Bu madde yürürlükten kaldırılmıştır."]]
            else:
                warnings.append(f"bilinmeyen değişiklik türü: {kind}")
                continue
            article["degisiklik"] = change["etiket"]


def article_slug(label: str) -> str:
    return re.sub(r"\W+", "_", normalize_for_matching(label)).strip("_")


def madde_no_of(label: str) -> str:
    """'MADDE 22' -> '22', 'GEÇİCİ MADDE 1' -> 'geçici 1', 'ESAS 2' -> 'esas 2', 'BÖLÜM 3.2.1' -> 'bölüm 3.2.1'."""
    words = label.split()
    number = words[-1]
    kind = [w for w in words[:-1] if w != "MADDE"]
    lowered = [w.replace("İ", "i").replace("I", "ı").lower() for w in kind]
    return " ".join(lowered + [number])


# Madde içi atıf: "20 nci maddedeki", "16 ncı maddenin", "Madde 24". Hemen öncesinde "sayılı" geçiyorsa
# başka bir mevzuata atıftır ("2547 sayılı Kanunun 44 üncü maddesi") ve alınmaz.
REFERENCE_RE = re.compile(
    r"(\d+)\s*[’']?\s*(?:inci|nci|üncü|uncu|ıncı|ncı|ünci|unci)\s+madde|\bmadde\s+(\d+(?:/[A-ZÇĞİÖŞÜ])?)",
    re.IGNORECASE,
)


def internal_references(text: str) -> set[str]:
    found: set[str] = set()
    for match in REFERENCE_RE.finditer(text):
        if "sayılı" in text[max(0, match.start() - 60):match.start()]:
            continue
        found.add((match.group(1) or match.group(2)).upper())
    return found


def build_records(entry: dict[str, Any], articles: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Parça şeması (15 alan): belge_id, belge_adi, kurum, hiyerarsi, hedef_kitle, birim, konu, madde_id,
    madde_no, madde_parca_no, fikra_no, atiflar, yururluk_tarihi, yururluk_bitis, parca_turu.

    Gösterim için gereken diğer bilgiler (madde başlığı, bölüm, sayfa, değişiklik notu, eşanlamlılar,
    belge uyarısı) parçalarda değil, madde kayıtlarında (maddeler.jsonl) bir kez tutulur.
    Madde kaydının kimliği "<madde_id>#<madde_parca_no>": uzun maddeler birden çok bağlam birimine bölünür.
    """
    aliases_by_label = STUDENT_ALIASES.get(entry["ad"], {})
    madde_kitle = entry.get("madde_hedef_kitle", {})

    # 1) Madde kimlikleri: bölünmüş parçalar aynı maddeye bağlanır; aynı etiket ikinci kez geçerse
    #    (ör. ÇAP yönetmeliğinde sonda ikinci bir "MADDE 1") ayrı madde sayılır.
    label_counts: dict[str, int] = {}
    id_owner: dict[str, tuple[str, int]] = {}  # madde_id -> (etiket, kaçıncı geçiş)
    keyed: list[tuple[dict[str, Any], str, int]] = []
    for article in articles:
        suffix = article.get("id_suffix", "")
        part_no = int(suffix[2:]) if suffix.startswith("_b") else 1
        if part_no == 1:
            label_counts[article["label"]] = label_counts.get(article["label"], 0) + 1
        owner = (article["label"], label_counts[article["label"]])
        base = f"{entry['ad']}__m{article_slug(madde_no_of(article['label']))}"
        madde_id, extra = base, 2
        # Sadeleştirme farklı etiketleri aynı kimliğe düşürebiliyor ("53/C" ve "53/Ç" -> m53_c).
        while id_owner.setdefault(madde_id, owner) != owner:
            madde_id, extra = f"{base}_{extra}", extra + 1
        keyed.append((article, madde_id, part_no))

    # 2) Atıflar madde düzeyinde: tüm parçaların metninden, yalnızca bu belgede var olan maddelere.
    id_by_no = {madde_no_of(a["label"]): mid for a, mid, _ in keyed}
    references: dict[str, set[str]] = {}
    for article, madde_id, _ in keyed:
        text = " ".join([article["preamble"]] + [t for _, t in article["paragraphs"]])
        targets = {id_by_no[no] for no in internal_references(text) if no in id_by_no}
        references.setdefault(madde_id, set()).update(targets - {madde_id})

    article_records: list[dict] = []
    chunk_records: list[dict] = []
    for article, madde_id, part_no in keyed:
        label, heading = article["label"], article["heading"]
        heading_part = f" – {heading}" if heading else ""
        aliases = aliases_by_label.get(label, [])
        fields = {
            "belge_id": entry["ad"],
            "belge_adi": entry["kisa"],
            "kurum": entry["kurum"],
            "hiyerarsi": entry["hiyerarsi"],
            "hedef_kitle": madde_kitle.get(label, entry["hedef_kitle"]),
            "birim": entry["birim"],
            "konu": topic_for(entry["ad"], madde_id),  # konu_etiketle.py (madde düzeyinde, LLM)
            "madde_id": madde_id,
            "madde_no": madde_no_of(label),
            "madde_parca_no": part_no,
            "atiflar": sorted(references.get(madde_id, set())),
            "yururluk_tarihi": entry.get("yururluk_tarihi", ""),
            "yururluk_bitis": entry.get("yururluk_bitis", ""),
        }

        lines = [f"{label}{heading_part}"]
        if article["degisiklik"]:
            lines.append(f"(Bu madde {article['degisiklik']}.)")
        if article["preamble"]:
            lines.append(article["preamble"])
        lines += [f"({n}) {content}" if n else content for n, content in article["paragraphs"]]
        article_records.append({
            "id": f"{madde_id}#{part_no}",
            "metin": "\n".join(lines),
            "ozellikler": {
                **fields,
                # Gösterim/bağlam alanları (parçalarda yok):
                "madde_etiketi": label,
                "madde_basligi": heading,
                "bolum_basligi": article["section"],
                "sayfa": article["start_page"],
                "son_degisiklik": article["degisiklik"],
                "esanlamlilar": "; ".join(aliases),
                "belge_notu": entry.get("baglam_notu", ""),
            },
        })

        pieces = ([[0, article["preamble"]]] if article["preamble"] else []) + article["paragraphs"]
        chunk_index = 0  # bentlerine bölünen fıkra aynı numarayla birden çok kez gelebilir
        for number, content in pieces:
            prefix = f"{entry['kisa']} | {label}{heading_part}" + (f" – ({number})" if number else " –")
            for part in split_long_text(content, max(200, MAX_CHUNK_CHARS - len(prefix) - 20), CHUNK_OVERLAP_CHARS):
                chunk_index += 1
                chunk_records.append({
                    "id": f"{madde_id}__b{part_no}__c{chunk_index}",
                    "metin": f"{prefix} {part}",
                    "ozellikler": {**fields, "fikra_no": number, "parca_turu": "fikra"},
                })
        if aliases and part_no == 1:
            chunk_records.append({
                "id": f"{madde_id}__esanlamli",
                "metin": f"{entry['kisa']} | {label}{heading_part} – Öğrencilerin kullandığı ifadeler: "
                         + "; ".join(aliases),
                "ozellikler": {**fields, "fikra_no": 0, "parca_turu": "esanlamli"},
            })
    return article_records, chunk_records


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def embed_with_retry(ollama: Any, texts: list[str], attempts: int = 8) -> list[list[float]]:
    """Ollama, her girdi için llama-server'a ayrı bağlantı açıyor; Windows'ta kapanan bağlantılar ~2 dk
    TIME_WAIT'te kalıp binlerce parçalık toplu işte yuva tamponunu dolduruyor ("bind: ... arabellek alanı").
    Hata alınca bağlantıların boşalması beklenip tekrar denenir."""
    for attempt in range(attempts):
        try:
            return ollama.embed(model=EMBEDDING_MODEL, input=texts)["embeddings"]
        except ollama.ResponseError as error:
            if attempt == attempts - 1:
                raise
            wait = 30 * (attempt + 1)
            print(f"\n  Ollama hatası ({str(error)[:60]}...), {wait} sn bekleniyor", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def load_chroma(chunks: list[dict]) -> None:
    import chromadb
    import ollama

    # Tam yeniden kurulum: silinen/değişen maddeler kalmasın. Koleksiyonu aynı süreçte silip yeniden
    # oluşturmak (delete_collection) Windows'ta yarım kalan indeks klasörleri bırakıp okumayı bozuyordu;
    # bu klasör yalnızca bu script'in çıktısı olduğundan tamamen silinir.
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)
    unique: dict[str, dict] = {}
    for chunk in chunks:  # yedek önlem: aynı kimlikli parça ikinci kez eklenmez (ChromaDB hata veriyor)
        unique.setdefault(chunk["id"], chunk)
    if len(unique) < len(chunks):
        print(f"  uyarı: {len(chunks) - len(unique)} yinelenen parça kimliği atlandı", flush=True)
    chunks = list(unique.values())
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        vectors = embed_with_retry(ollama, [c["metin"] for c in batch])
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["metin"] for c in batch],
            # ChromaDB metadata'sı liste tutamıyor: atıflar virgüllü metin olarak saklanır.
            metadatas=[{**c["ozellikler"], "atiflar": ",".join(c["ozellikler"]["atiflar"])} for c in batch],
            embeddings=vectors,
        )
        print(f"  Embedding: {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}", end="\r")
    print(f"\nChromaDB: {collection.count()} parça -> {CHROMA_PATH} ({COLLECTION_NAME})")


def update_chroma(chunks: list[dict]) -> None:
    """Artımlı güncelleme (--artimli): yalnızca yeni/değişen parçalar gömülür, artık olmayanlar silinir.
    Tam kurulum ~1 saat GPU; birkaç belgelik düzeltmede dakikalar. Canlı sunucunun klasörüne yazma:
    önce kopyala (KOU_CHROMA_DIR ile kopyayı göster), sonra klasörleri değiştir."""
    import chromadb
    import ollama

    unique: dict[str, dict] = {}
    for chunk in chunks:
        unique.setdefault(chunk["id"], chunk)
    wanted = {cid: (c["metin"], {**c["ozellikler"], "atiflar": ",".join(c["ozellikler"]["atiflar"])})
              for cid, c in unique.items()}
    collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(COLLECTION_NAME)
    existing: dict[str, tuple[str, dict]] = {}
    total = collection.count()
    for offset in range(0, total, 5000):
        page = collection.get(limit=5000, offset=offset, include=["documents", "metadatas"])
        existing.update({i: (d, m) for i, d, m in zip(page["ids"], page["documents"], page["metadatas"])})
    stale = [i for i in existing if i not in wanted]
    changed = [i for i, value in wanted.items() if existing.get(i) != value]
    print(f"Artımlı: mevcut {len(existing)}, silinecek {len(stale)}, eklenecek/değişen {len(changed)}", flush=True)
    if len(changed) > 0.3 * max(1, len(existing)) and "--zorla" not in sys.argv[1:]:
        print("Değişen parça çok fazla (metadata karşılaştırması bozuk olabilir); tam kurulum yap ya da --zorla.")
        return
    for start in range(0, len(stale), 5000):
        collection.delete(ids=stale[start:start + 5000])
    for start in range(0, len(changed), BATCH_SIZE):
        batch = changed[start:start + BATCH_SIZE]
        collection.upsert(ids=batch, documents=[wanted[i][0] for i in batch], metadatas=[wanted[i][1] for i in batch],
                          embeddings=embed_with_retry(ollama, [wanted[i][0] for i in batch]))
    print(f"ChromaDB: {collection.count()} parça -> {CHROMA_PATH} ({COLLECTION_NAME})")


def main() -> int:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))["kullanilan"]
    # web_belgeleri.py (sayfalar ve PDF'ler) ve web_yapi.py (fakülte/bölüm yapısı) katalogları
    for web_catalog in sorted(WEB_CATALOG.parent.glob("katalog_*.json")):
        catalog += json.loads(web_catalog.read_text(encoding="utf-8"))
    changes = {c["belge"]: c for c in json.loads(CHANGES.read_text(encoding="utf-8"))} if CHANGES.exists() else {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_articles: list[dict] = []
    all_chunks: list[dict] = []
    report: list[str] = []
    for entry in catalog:
        if entry["tur"] == "değişiklik":
            report.append(f"[işlendi] {entry['kisa']} -> {entry.get('degistirdigi')}")
            continue
        warnings: list[str] = []
        if entry.get("kapsam") == "web":  # web belgesi: türüne bakılmaksızın serbest metin ya da MADDE yapısı
            articles = parse_plain(read_raw_text(entry)) if entry.get("yapi") == "serbest" \
                else parse_articles(read_document_text(entry))
            forms = [a for a in articles if is_form_like(a)]
            if forms:
                articles = [a for a in articles if not is_form_like(a)]
                warnings.append(f"bilgi: form/tablo benzeri {len(forms)} bölüm atlandı")
            people = [a for a in articles if is_person_list(a)]
            if people:
                articles = [a for a in articles if not is_person_list(a)]
                warnings.append(f"bilgi: kişisel liste {len(people)} bölüm atlandı")
        elif entry["tur"] == "kılavuz":
            articles = parse_guide(strip_symbol_glyphs(pdf_raw_text(VERI / entry["dosya"])))
        elif entry["tur"] == "tebliğ":
            articles = parse_numbered_sections(strip_symbol_glyphs(pdf_raw_text(VERI / entry["dosya"])))
        elif entry.get("yapi") == "serbest":
            articles = parse_plain(read_raw_text(entry))
        else:
            articles = parse_articles(read_document_text(entry))
        if not articles:
            warnings.append("HİÇ MADDE BULUNAMADI — belge indekslenmedi")
        if entry["ad"] in changes:
            apply_changes(articles, changes[entry["ad"]], warnings)
        # Yalnızca katalogda işaretli belgelerde: diğerlerinde tekrar eden numara değişiklik/yazım kaynaklı.
        restarted = prefix_restarted_numbering(articles) if entry.get("bolum_bazli_numara") else 0
        if restarted:
            warnings.append(f"bilgi: madde numarası {restarted} kez 1'den yeniden başlıyor; etiketlere bölüm eklendi")
        repealed = [a["label"] for a in articles if is_repealed_only(a)]
        if repealed:
            articles = [a for a in articles if not is_repealed_only(a)]
            warnings.append(f"bilgi: içeriksiz mülga/iptal madde atlandı: {len(repealed)} ({', '.join(repealed[:5])}…)")
        original_count = len(articles)
        articles = split_long_articles(articles)
        if len(articles) > original_count:
            warnings.append(f"bilgi: uzun maddeler bölündü ({original_count} -> {len(articles)} bağlam birimi)")
        no_heading = [a["label"] for a in articles if not a["heading"]]
        empty = [a["label"] for a in articles if not a["paragraphs"] and not a["preamble"]]
        if no_heading:
            warnings.append(f"başlıksız madde: {len(no_heading)} ({', '.join(no_heading[:6])}{'…' if len(no_heading) > 6 else ''})")
        if empty:
            warnings.append(f"boş madde: {', '.join(empty)}")
        numbers = [a["number"] for a in articles if a["label"].startswith("MADDE")]
        # En küçük numaradan başlanır: tek maddelik kanun alıntısında (Madde 54) 1-53 eksik sayılmasın.
        gaps = sorted(set(range(min(numbers, default=1), max(numbers, default=0) + 1)) - set(numbers))
        if gaps:
            warnings.append(f"eksik madde numaraları: {gaps[:10]}")

        article_records, chunk_records = build_records(entry, articles)
        all_articles += article_records
        all_chunks += chunk_records
        changed = sum(1 for a in articles if a["degisiklik"])
        report.append(f"{'⚠' if any(not w.startswith('bilgi') for w in warnings) else '✓'} {entry['kisa']}: {len(articles)} madde, {len(chunk_records)} parça"
                      + (f", {changed} madde değişiklikle güncellendi" if changed else ""))
        report += [f"    - {w}" for w in warnings]

    write_jsonl(all_articles, ARTICLES_JSONL)
    write_jsonl(all_chunks, CHUNKS_JSONL)
    report.append(f"\nTOPLAM: {len(all_articles)} madde, {len(all_chunks)} parça")
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))

    if "--artimli" in sys.argv[1:]:
        update_chroma(all_chunks)
    elif "--sadece-json" not in sys.argv[1:]:
        load_chroma(all_chunks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
