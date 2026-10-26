#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PDF yönetmeliklerini çıkarır, MADDE/fıkra bazında parçalar ve ChromaDB'ye aktarır.

Kurulum:
    pip install pymupdf chromadb ollama
    ollama pull bge-m3

Klasör:
    proje/
      pdf_to_chromadb.py
      pdf_yonetmelikler/   <- PDF dosyalarını buraya koy

Kullanım:
    python pdf_to_chromadb.py              # chunk'la + embedding + ChromaDB
    python pdf_to_chromadb.py --sadece-json  # sadece JSONL üret (Ollama/ChromaDB gerekmez)

Çıktılar:
    prepared_chunks.jsonl -> fıkra bazlı chunk'lar (arama/embedding birimi)
    articles.jsonl        -> madde bazlı tam metinler (LLM'e verilen bağlam birimi,
                             fine-tuning veri üretimi de bunu kullanır)

NOT: RAG sorgularında da aynı embedding modelini (EMBEDDING_MODEL) kullanın.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from bisect import bisect_right
from pathlib import Path
from typing import Any

import pymupdf

# ------------------------- AYARLAR -------------------------
PDF_DIR = Path("pdf_yonetmelikler")
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "universite_yonetmelikleri"
EMBEDDING_MODEL = "bge-m3"  # Mevcut RAG sorgularındaki embedding modeliyle aynı olmalı.
BATCH_SIZE = 32
MAX_CHUNK_CHARS = 1800
CHUNK_OVERLAP_CHARS = 180
PREPARED_JSONL = Path("prepared_chunks.jsonl")
ARTICLES_JSONL = Path("articles.jsonl")

# Belgeye özel bilgileri burada tanımlayın. Anahtar olarak PDF_DIR içindeki
# göreli yolu (tercih edilir) veya dosya adını kullanabilirsiniz.
# Örnek değerlerdir; gerçek sürüm/tarih/status bilgileriyle değiştirin.
# status için önerilen değerler: "current", "superseded", "unknown".
DOCUMENT_METADATA: dict[str, dict[str, str]] = {
    "kouonlisansvelisansegitimogretimyonetmeligi.pdf": {
        "document_title": "KOÜ Ön Lisans ve Lisans Eğitim-Öğretim Yönetmeliği",
        "version": "2019-10-09 (RG 30913)",
    },
}

DEFAULT_DOCUMENT_METADATA = {
    "version": "2019",
    "status": "active",
    "language": "tr",
    "topic": "genel",
}

# Öğrencilerin kullandığı ama yönetmelikte geçmeyen ifadeler ("kayıt dondurma" -> izinli
# sayılma, "büt" -> bütünleme). Her madde için ayrı bir "alias" chunk'ı olarak sadece aramaya
# eklenir; LLM'e yine maddenin orijinal metni (articles.jsonl) gider. Yönetmelikte karşılığı
# olmayan kavramları (yandal, Farabi...) EKLEME: model o soruları reddetmeli.
STUDENT_ALIASES: dict[str, dict[str, list[str]]] = {
    "kouonlisansvelisansegitimogretimyonetmeligi.pdf": {
        "MADDE 4": ["GANO nedir", "AGNO ne demek", "akademik danışman kimdir", "e-ders nedir"],
        "MADDE 5": ["azami süre", "okulu en fazla kaç yılda bitirmeliyim", "maksimum öğrenim süresi",
                    "süre aşımı", "okulu bırakırsam önlisans diploması alır mıyım"],
        "MADDE 6": ["üniversiteye ilk kayıt", "kesin kayıt", "kayıt için gereken belgeler"],
        "MADDE 7": ["oryantasyon", "tanıtım günleri"],
        "MADDE 8": ["danışman hocam kim", "danışman ne işe yarar"],
        "MADDE 9": ["başka üniversiteye geçmek", "üniversite değiştirmek", "kurumlar arası yatay geçiş"],
        "MADDE 10": ["bölüm değiştirmek", "bölümler arası geçiş", "kurum içi yatay geçiş"],
        "MADDE 11": ["DGS", "dikey geçişle gelen öğrenci", "önlisanstan lisansa geçiş"],
        "MADDE 12": ["ders seçimi", "ders kaydı", "ders ekle bırak", "ekle sil haftası", "kayıt yenileme",
                     "alttan ders", "kaldığım dersi tekrar almak", "danışman onayı", "geç kayıt",
                     "harcı yatırmadım ders seçemedim"],
        "MADDE 14": ["zorunlu ders", "seçmeli ders", "ön şartlı ders", "ön koşullu ders"],
        "MADDE 15": ["üniversite seçmeli dersi", "ÜSD", "ortak seçmeli ders"],
        "MADDE 16": ["kredi sınırı", "AKTS limiti", "üstten ders almak", "en fazla kaç AKTS alabilirim",
                     "ders yükü"],
        "MADDE 17": ["devamsızlık", "devamsızlık sınırı", "yoklama", "derse kaç hafta gelmeyebilirim",
                     "devamsızlıktan kalmak", "devam zorunluluğu"],
        "MADDE 18": ["sınavlar nasıl yapılır", "engelli öğrenci sınavı", "sınav programı"],
        "MADDE 19": ["vize", "vize sınavı", "ara sınav", "kısa sınav", "quiz", "ödev notu", "vize yüzdesi"],
        "MADDE 20": ["final", "final sınavı", "finale girme şartı", "bütünleme", "büt", "büte kimler girer",
                     "dönem sonu sınavı"],
        "MADDE 21": ["mazeret sınavı", "telafi sınavı", "raporluyken sınava giremedim", "sağlık raporu ile sınav"],
        "MADDE 22": ["tek ders sınavı", "mezuniyete tek ders kaldı"],
        "MADDE 23": ["notlar ne zaman açıklanır", "sınav sonuçları ne zaman ilan edilir"],
        "MADDE 24": ["harf notları", "AA BA BB CB CC", "DC ile geçer miyim", "şartlı geçme", "FF ne demek",
                     "kaç alırsam geçerim", "not sistemi", "harf notu katsayıları"],
        "MADDE 27": ["not itirazı", "sınav sonucuna itiraz", "notum yanlış girilmiş", "sınav kağıdımı görmek"],
        "MADDE 28": ["dönem ortalaması", "YANO", "dönem not ortalaması nasıl hesaplanır"],
        "MADDE 29": ["onur belgesi", "yüksek onur belgesi", "onur öğrencisi olmak için ortalama"],
        "MADDE 30": ["GANO kaç olmalı", "başarılı öğrenci sayılmak", "genel ortalama"],
        "MADDE 31": ["programdan kaldırılan ders", "seçmeli dersi değiştirmek"],
        "MADDE 32": ["hazırlık sınıfı", "İngilizce hazırlık"],
        "MADDE 34": ["ne zaman mezun sayılırım", "mezuniyet tarihi"],
        "MADDE 35": ["diploma", "transkript", "diploma eki", "mezuniyet belgesi"],
        "MADDE 36": ["bitirme projesi", "bitirme tezi", "mezuniyet projesi"],
        "MADDE 37": ["staj", "zorunlu staj", "işyeri eğitimi", "işletmede mesleki eğitim"],
        "MADDE 38": ["okulu bırakmak", "kaydımı sildirmek", "okuldan ayrılmak", "ilişik kesme"],
        "MADDE 39": ["kayıt dondurma", "kaydımı dondurmak", "dönem dondurmak", "okula ara vermek",
                     "bir dönem izin almak", "hastalık izni"],
        "MADDE 40": ["Erasmus", "yurt dışında okumak", "değişim programı", "yurt dışı ders denkliği"],
        "MADDE 41": ["yurt içi değişim programı", "başka üniversitede bir dönem okumak"],
        "MADDE 42": ["ikinci öğretim", "gece öğretimi", "ikinci öğretimden birinci öğretime geçiş"],
        "MADDE 43": ["çift anadal", "ÇAP", "iki bölüm birden okumak"],
        "MADDE 44": ["uzaktan eğitim", "online ders", "çevrimiçi ders"],
        "MADDE 45": ["yaz okulu", "yaz dönemi"],
    },
}

# Benzer ama FARKLI kavramlar için uyarı (model "yandal" sorusuna çift anadal maddesiyle
# "evet" diyordu). Bağlamda maddenin altında gösterilir.
STUDENT_NOTES: dict[str, dict[str, str]] = {
    "kouonlisansvelisansegitimogretimyonetmeligi.pdf": {
        "MADDE 43": "Bu madde çift anadal (ÇAP) içindir; yandal programı bu yönetmelikte düzenlenmez.",
    },
}

# Chunk metninden otomatik konu tahmini için basit anahtar kelime listeleri.
# Bu etiketler yardımcı niteliktedir; kritik kullanımda elle kontrol edin.
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "harc_ucret": ("harc", "katki payi", "ogrenim ucreti", "ucret", "borc"),
    "kayit": ("kayit", "kayit yenileme", "universiteye kayit", "ogrenci isleri"),
    "yatay_gecis": ("yatay gecis", "kurumlar arasi gecis", "kurum ici gecis"),
    "sinav": ("sinav", "butunleme", "vize", "final", "ozel yetenek"),
    "dersler": ("ders secimi", "ders kaydi", "ders programi", "danisman"),
    "mezuniyet_diploma": ("diploma", "mezuniyet", "transkript", "not dokumu"),
    "burs": ("burs", "yemek bursu", "burslu"),
    "yemekhane": ("yemekhane", "yemek", "ogun"),
    "kutuphane": ("kutuphane", "kitap odunc", "veritabani", "acik erisim"),
    "kampus_karti": ("kampus kart", "ogrenci karti", "akilli kart"),
    "ulasim": ("ulasim", "otobus", "umuttepe", "e-komobil", "ekomobil"),
    "disiplin": ("disiplin", "sorusturma", "disiplin cezasi"),
}

# MADDE 6, GEÇİCİ MADDE 1, EK MADDE 2 gibi başlıklar.
ARTICLE_RE = re.compile(
    r"(?im)^[ \t]*(?P<label>(?:(?:GEÇİCİ|EK)[ \t]+)?MADDE[ \t]+\d+[A-Z]?)"
    r"[ \t]*(?:[–—\-:][ \t]*)?"
)
# "ESAS 1-": bazı esaslar maddeleri böyle numaralandırıyor (ör. KOÜ Yaz Öğretimine İlişkin Esaslar).
ARTICLE_LINE_RE = re.compile(
    r"(?i)^[ \t]*(?:(?:GEÇİCİ|EK)[ \t]+)?(?:MADDE|ESAS)[ \t]+\d+"
)
FIKRA_LINE_RE = re.compile(r"^[ \t]*\(\d+\)[ \t]*")
FIKRA_RE = re.compile(r"(?m)^[ \t]*\((\d+)\)[ \t]*")
PAGE_TAG_RE = re.compile(r"__PAGE_(\d+)__")
PAGE_TAG_LINE_RE = re.compile(r"__PAGE_\d+__")
# Madde başlığı ("Amaç", "Tek ders sınavı") ve bölüm başlığı ("İKİNCİ BÖLÜM") işaretleri.
# PDF'te bu satırlar MADDE satırının hemen üstündedir; işaretlenmezse önceki maddeye yapışırlar.
HEADING_TAG = "__HEADING__"
SECTION_TAG = "__SECTION__"
STRUCTURE_TAG_RE = re.compile(rf"(?:{HEADING_TAG}|{SECTION_TAG})[^\n]*")
SECTION_LINE_RE = re.compile(r"^\S+[ \t]+BÖLÜM$")


def normalize_for_matching(value: str) -> str:
    """Türkçe metni anahtar kelime eşleştirmesi için sadeleştirir."""
    value = value.casefold().replace("ı", "i")
    value = unicodedata.normalize("NFKD", value)
    return "".join(char for char in value if not unicodedata.combining(char))


def resolve_document_metadata(pdf_path: Path) -> dict[str, str]:
    """Dosya yapılandırmasını varsayılanlarla birleştirir; gerçek tarih/sürüm uydurmaz."""
    metadata = dict(DEFAULT_DOCUMENT_METADATA)

    keys: list[str] = [pdf_path.name, pdf_path.stem]
    try:
        keys.insert(0, pdf_path.relative_to(PDF_DIR).as_posix())
    except ValueError:
        pass

    for key in keys:
        configured = DOCUMENT_METADATA.get(key)
        if configured:
            metadata.update({str(k): str(v) for k, v in configured.items()})
            break

    metadata["document_title"] = metadata.get("document_title", "") or pdf_path.stem.replace("_", " ").replace("-", " ")
    # Bu değer, sınıflandırıcı bir konu bulamazsa geri dönüş olarak kullanılır.
    metadata["topic"] = metadata.get("topic", "genel") or "genel"
    return metadata


def infer_chunk_topic(text: str, fallback_topic: str = "genel", heading: str = "") -> str:
    """Chunk metninden anahtar kelime tabanlı bir konu etiketi tahmin eder.

    Madde başlığındaki eşleşmeler metindekilerden daha güçlü bir sinyaldir; 3 kat sayılır.
    """
    normalized_text = normalize_for_matching(text)
    normalized_heading = normalize_for_matching(heading)
    scores: dict[str, int] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            normalized_keyword = normalize_for_matching(keyword)
            if normalized_keyword in normalized_text:
                score += 1
            if normalized_heading and normalized_keyword in normalized_heading:
                score += 3
        if score:
            scores[topic] = score
    if not scores:
        return fallback_topic
    # Eşitlikte TOPIC_KEYWORDS sırasını koruyarak deterministik sonuç üret.
    return max(scores, key=scores.get)


def stable_document_id(pdf_path: Path) -> str:
    """Aynı dosya yoluna tekrar işlendiğinde aynı document_id üretir."""
    value = str(pdf_path.resolve()).lower().encode("utf-8")
    return hashlib.sha1(value).hexdigest()[:12]


def is_heading_like(line: str, allow_trailing_comma: bool = False) -> bool:
    """Kısa, büyük harfle başlayan ve cümle gibi bitmeyen satırlar başlık adayıdır.

    Uzun bölüm başlıkları iki satıra bölünebilir ("..., Sınavlar," + "Notlar ve Diploma");
    bölüm satırları için sondaki virgüle izin verilir.
    """
    forbidden_endings = (".", ";", ":") if allow_trailing_comma else (".", ",", ";", ":")
    return (
        0 < len(line) <= 100
        and line[0].isupper()
        and not line.endswith(forbidden_endings)
        and not ARTICLE_LINE_RE.match(line)
        and not FIKRA_LINE_RE.match(line)
        and not PAGE_TAG_LINE_RE.fullmatch(line)
    )


def find_structure_lines(lines: list[str]) -> dict[int, str]:
    """MADDE satırlarının üstündeki madde/bölüm başlığı satırlarını bulur.

    Beklenen düzen (sayfa işaretleri araya girebilir):
        İKİNCİ BÖLÜM                       -> bölüm
        Eğitim-Öğretim ile İlgili Esaslar  -> bölüm
        Eğitim-öğretim süresi ve düzeyleri -> madde başlığı
        MADDE 5 – (1) ...
    """
    tags: dict[int, str] = {}
    for index, line in enumerate(lines):
        if not ARTICLE_LINE_RE.match(line):
            continue
        candidates: list[int] = []
        cursor = index - 1
        while cursor >= 0 and len(candidates) < 4:
            if PAGE_TAG_LINE_RE.fullmatch(lines[cursor]):
                cursor -= 1
                continue
            # İlk aday madde başlığıdır ve katı kurala uyar; üsttekiler bölüm başlığı olabilir.
            if cursor in tags or not is_heading_like(lines[cursor], allow_trailing_comma=bool(candidates)):
                break
            candidates.append(cursor)
            if SECTION_LINE_RE.match(lines[cursor]):
                break
            cursor -= 1

        if not candidates:
            continue
        tags[candidates[0]] = HEADING_TAG
        # Üstteki adaylar ancak bir "... BÖLÜM" satırıyla bitiyorsa bölüm başlığıdır.
        if len(candidates) > 1 and SECTION_LINE_RE.match(lines[candidates[-1]]):
            for candidate in candidates[1:]:
                tags[candidate] = SECTION_TAG
    return tags


def pdf_raw_text(pdf_path: Path) -> str:
    """PDF metnini sayfa işaretleriyle (__PAGE_n__) ham olarak çıkarır."""
    pieces: list[str] = []
    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text("text", sort=True) or ""
            pieces.append(f"\n__PAGE_{page_number}__\n")
            pieces.append(page_text)
    return "".join(pieces)


def extract_and_normalize_pdf(pdf_path: Path) -> str:
    """PDF metnini sayfa işaretleriyle çıkarır; fiziksel satır sonlarını düzeltir."""
    return normalize_raw_text(pdf_raw_text(pdf_path))


def normalize_raw_text(raw_text: str) -> str:
    """Ham metindeki fiziksel satır sonlarını düzeltir; başlık/madde/fıkra satırlarını korur.

    PDF'ten veya elle düzeltilmiş .txt'den gelen metin aynı yoldan geçer.
    """
    raw_text = raw_text.replace("\r", "\n").replace("\u00a0", " ")
    # PDF satır sonunda bölünmüş kelimeleri birleştir: "yete-\nnek" -> "yetenek".
    raw_text = re.sub(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)", "", raw_text)

    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    structure_tags = find_structure_lines(lines)

    # Başlıkları, sayfa işaretlerini ve yeni fıkra başlangıçlarını koru;
    # diğer fiziksel satırları boşlukla birleştir.
    output: list[str] = []
    for line_index, line in enumerate(lines):
        is_page = bool(PAGE_TAG_LINE_RE.fullmatch(line))
        is_article = bool(ARTICLE_LINE_RE.match(line))
        is_fikra = bool(FIKRA_LINE_RE.match(line))

        if line_index in structure_tags:
            line = f"{structure_tags[line_index]} {line}"

        if is_page or is_article or is_fikra or line_index in structure_tags:
            if output and not output[-1].endswith("\n"):
                output.append("\n")
            output.append(line)
            output.append("\n")
        else:
            if output and not output[-1].endswith(("\n", " ")):
                output.append(" ")
            output.append(line)

    return "".join(output).strip()


def clean_chunk_text(text: str) -> str:
    """Sayfa/başlık işaretlerini kaldırır ve chunk içindeki boşlukları düzenler."""
    text = STRUCTURE_TAG_RE.sub(" ", text)
    text = PAGE_TAG_RE.sub(" ", text)
    text = text.replace("\u00ad", "")  # soft hyphen
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Uzun bir fıkrayı cümle/kelime sınırlarını tercih ederek alt parçalara böler."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        hard_end = min(start + max_chars, text_length)
        end = hard_end

        if hard_end < text_length:
            # Önce cümle, sonra noktalı virgül, en son boşluk sınırı ara.
            candidates = [
                text.rfind(". ", start, hard_end),
                text.rfind("; ", start, hard_end),
                text.rfind(" ", start, hard_end),
            ]
            best = max(candidates)
            if best > start + int(max_chars * 0.55):
                end = best + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= text_length:
            break

        start = max(start + 1, end - overlap_chars)

    return chunks


def page_markers(normalized_text: str) -> tuple[list[int], list[int]]:
    positions: list[int] = []
    pages: list[int] = []
    for match in PAGE_TAG_RE.finditer(normalized_text):
        positions.append(match.start())
        pages.append(int(match.group(1)))
    return positions, pages


def page_at(offset: int, marker_positions: list[int], marker_pages: list[int], default: int = 1) -> int:
    index = bisect_right(marker_positions, offset) - 1
    return marker_pages[index] if index >= 0 else default


def article_slug(label: str) -> str:
    """'GEÇİCİ MADDE 1' -> 'gecici_madde_1' (madde kimliği için)."""
    return re.sub(r"\W+", "_", normalize_for_matching(label)).strip("_")


def make_chunk_records(
    pdf_path: Path, normalized_text: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Metni fıkra bazlı chunk'lara ve madde bazlı tam metinlere dönüştürür.

    Fıkra chunk'ları arama (embedding) için, madde metinleri LLM'e bağlam olarak
    verilmek için kullanılır. Her chunk, ait olduğu maddeyi parent_article_id ile gösterir.
    """
    doc_id = stable_document_id(pdf_path)
    document_metadata = resolve_document_metadata(pdf_path)
    title = document_metadata["document_title"]
    page_positions, page_numbers = page_markers(normalized_text)
    article_matches = list(ARTICLE_RE.finditer(normalized_text))
    section_matches = [
        (match.start(), match.group(1).strip())
        for match in re.finditer(rf"{SECTION_TAG} ([^\n]+)", normalized_text)
    ]
    heading_before_re = re.compile(rf"{HEADING_TAG} ([^\n]+)\n(?:__PAGE_\d+__\n)*$")
    aliases_by_label = STUDENT_ALIASES.get(pdf_path.name, {})
    notes_by_label = STUDENT_NOTES.get(pdf_path.name, {})
    records: list[dict[str, Any]] = []
    article_records: list[dict[str, Any]] = []

    def base_metadata() -> dict[str, Any]:
        return {
            "document_id": doc_id,
            "document_title": title,
            "source_file": pdf_path.name,
            "version": document_metadata["version"],
            "status": document_metadata["status"],
            "language": document_metadata["language"],
        }

    def add_record_group(
        *,
        prefix: str,
        article_label: str,
        content: str,
        paragraph_number: int,
        global_start: int,
        global_end: int,
        article_number: int,
        article_meta: dict[str, Any],
    ) -> None:
        content = clean_chunk_text(content)
        if not content:
            return

        content_parts = split_long_text(
            content,
            max_chars=max(200, MAX_CHUNK_CHARS - len(prefix) - 20),
            overlap_chars=CHUNK_OVERLAP_CHARS,
        )
        start_page = page_at(global_start, page_positions, page_numbers)
        end_page = page_at(max(global_start, global_end - 1), page_positions, page_numbers, start_page)

        for part_index, content_part in enumerate(content_parts, start=1):
            final_text = f"{prefix} {content_part}".strip()
            identity = f"{doc_id}|{article_label}|{paragraph_number}|{part_index}|{final_text}"
            chunk_id = hashlib.sha1(identity.encode("utf-8")).hexdigest()
            records.append(
                {
                    "id": f"{doc_id}_{chunk_id[:20]}",
                    "text": final_text,
                    "metadata": {
                        **base_metadata(),
                        **article_meta,
                        "article_label": article_label,
                        "article_number": article_number,
                        "paragraph_number": paragraph_number,
                        "start_page": int(start_page),
                        "end_page": int(end_page),
                        "chunk_part": part_index,
                        "chunk_type": "fikra",
                    },
                }
            )

    if article_matches:
        current_section = ""
        for article_index, article_match in enumerate(article_matches):
            region_start = article_matches[article_index - 1].end() if article_index else 0
            block_end = (
                article_matches[article_index + 1].start()
                if article_index + 1 < len(article_matches)
                else len(normalized_text)
            )
            body_start = article_match.end()
            body = normalized_text[body_start:block_end]
            label = re.sub(r"\s+", " ", article_match.group("label")).strip().upper()
            number_match = re.search(r"(\d+)", label)
            article_number = int(number_match.group(1)) if number_match else 0

            heading_match = heading_before_re.search(normalized_text, region_start, article_match.start())
            heading = heading_match.group(1).strip() if heading_match else ""
            new_sections = [
                text for position, text in section_matches if region_start <= position < article_match.start()
            ]
            if new_sections:
                # "DÖRDÜNCÜ BÖLÜM" + bölüm adı (birden fazla satıra bölünmüş olabilir).
                current_section = new_sections[0]
                if len(new_sections) > 1:
                    current_section += " – " + " ".join(new_sections[1:])

            heading_part = f" – {heading}" if heading else ""
            parent_id = f"{doc_id}_{article_slug(label)}"
            article_meta = {
                "parent_article_id": parent_id,
                "article_title": heading,
                "section_title": current_section,
                # Konu madde düzeyinde belirlenir; aynı maddenin fıkraları aynı etiketi taşır.
                "topic": infer_chunk_topic(clean_chunk_text(body), document_metadata["topic"], heading),
            }
            article_paragraphs: list[str] = []
            paragraph_matches = list(FIKRA_RE.finditer(body))

            if paragraph_matches:
                # Fıkra numarasından önce maddeye ait bağımsız bir giriş metni varsa kaybetme.
                preamble = clean_chunk_text(body[: paragraph_matches[0].start()])
                if preamble:
                    article_paragraphs.append(preamble)
                    add_record_group(
                        prefix=f"{label}{heading_part} –",
                        article_label=label,
                        content=preamble,
                        paragraph_number=0,
                        global_start=body_start,
                        global_end=body_start + paragraph_matches[0].start(),
                        article_number=article_number,
                        article_meta=article_meta,
                    )

                for paragraph_index, paragraph_match in enumerate(paragraph_matches):
                    paragraph_end = (
                        paragraph_matches[paragraph_index + 1].start()
                        if paragraph_index + 1 < len(paragraph_matches)
                        else len(body)
                    )
                    paragraph_text = body[paragraph_match.start():paragraph_end]
                    paragraph_number = int(paragraph_match.group(1))
                    # İlk (1) bilgisini metinden alıp her chunk'ta prefix'e koy.
                    cleaned_paragraph = clean_chunk_text(paragraph_text)
                    leading = re.match(r"^\((\d+)\)\s*", cleaned_paragraph)
                    if leading:
                        cleaned_paragraph = cleaned_paragraph[leading.end():].strip()
                    article_paragraphs.append(f"({paragraph_number}) {cleaned_paragraph}")
                    paragraph_prefix = f"{label}{heading_part} – ({paragraph_number})"

                    # Her alt chunk'ın başında madde, başlık ve fıkra numarası kalır.
                    add_record_group(
                        prefix=paragraph_prefix,
                        article_label=label,
                        content=cleaned_paragraph,
                        paragraph_number=paragraph_number,
                        global_start=body_start + paragraph_match.start(),
                        global_end=body_start + paragraph_end,
                        article_number=article_number,
                        article_meta=article_meta,
                    )
            else:
                article_paragraphs.append(clean_chunk_text(body))
                add_record_group(
                    prefix=f"{label}{heading_part} –",
                    article_label=label,
                    content=body,
                    paragraph_number=0,
                    global_start=article_match.start(),
                    global_end=block_end,
                    article_number=article_number,
                    article_meta=article_meta,
                )

            aliases = aliases_by_label.get(label, [])
            if aliases:
                alias_text = f"{label}{heading_part} – Öğrencilerin kullandığı ifadeler: {'; '.join(aliases)}"
                alias_page = int(page_at(article_match.start(), page_positions, page_numbers))
                records.append(
                    {
                        "id": f"{parent_id}_alias",
                        "text": alias_text,
                        "metadata": {
                            **base_metadata(),
                            **article_meta,
                            "article_label": label,
                            "article_number": article_number,
                            "paragraph_number": 0,
                            "start_page": alias_page,
                            "end_page": alias_page,
                            "chunk_part": 1,
                            # Sadece arama içindir; LLM'e madde metni (articles.jsonl) verilir.
                            "chunk_type": "alias",
                        },
                    }
                )

            article_paragraphs = [paragraph for paragraph in article_paragraphs if paragraph]
            if article_paragraphs:
                article_records.append(
                    {
                        "id": parent_id,
                        "text": f"{label}{heading_part}\n" + "\n".join(article_paragraphs),
                        "metadata": {
                            **base_metadata(),
                            **article_meta,
                            "article_label": label,
                            "article_number": article_number,
                            # LLM'e bağlamda gösterilir (raft_format.format_context): öğrencinin
                            # "kayıt dondurma" sorusunu "izinli sayılma" maddesiyle bağlayabilsin.
                            "student_aliases": "; ".join(aliases),
                            "student_note": notes_by_label.get(label, ""),
                            "start_page": int(page_at(article_match.start(), page_positions, page_numbers)),
                            "end_page": int(page_at(max(body_start, block_end - 1), page_positions, page_numbers)),
                        },
                    }
                )
    else:
        # Hiç MADDE başlığı algılanmadıysa metni kaybetmemek için genel parçalara ayır.
        # Bu durum OCR, alışılmadık PDF düzeni veya başlık biçimi nedeniyle oluşabilir.
        whole_text = clean_chunk_text(normalized_text)
        if whole_text:
            fallback_parts = split_long_text(whole_text, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS)
            for part_index, part in enumerate(fallback_parts, start=1):
                identity = f"{doc_id}|UNPARSED|{part_index}|{part}"
                chunk_id = hashlib.sha1(identity.encode("utf-8")).hexdigest()
                records.append(
                    {
                        "id": f"{doc_id}_{chunk_id[:20]}",
                        "text": part,
                        "metadata": {
                            **base_metadata(),
                            "topic": infer_chunk_topic(part, document_metadata["topic"]),
                            "parent_article_id": "",
                            "article_title": "",
                            "section_title": "",
                            "article_label": "UNPARSED",
                            "article_number": 0,
                            "paragraph_number": 0,
                            "start_page": 1,
                            "end_page": len(page_numbers) or 1,
                            "chunk_part": part_index,
                            "chunk_type": "fikra",
                        },
                    }
                )

    return records, article_records


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """Gözle kontrol edilebilmesi için hazırlanan parçaları JSONL'ye yazar."""
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def embed_records(records: list[dict[str, Any]]) -> list[list[float]]:
    import ollama

    embeddings: list[list[float]] = []
    for start in range(0, len(records), BATCH_SIZE):
        batch_records = records[start:start + BATCH_SIZE]
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=[record["text"] for record in batch_records],
        )
        batch_embeddings = response["embeddings"]
        if len(batch_embeddings) != len(batch_records):
            raise RuntimeError("Ollama'nın ürettiği embedding sayısı metin sayısıyla eşleşmiyor.")
        embeddings.extend(batch_embeddings)
        print(f"  Embedding: {min(start + len(batch_records), len(records))}/{len(records)}")
    return embeddings


def upsert_pdf_records(collection: Any, pdf_path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        print(f"  UYARI: {pdf_path.name} için kaydedilecek parça bulunamadı.")
        return

    # Önce embedding'leri oluşturuyoruz; hata olursa eski kayıtları silmeyelim.
    embeddings = embed_records(records)
    doc_id = records[0]["metadata"]["document_id"]

    # Aynı PDF yeniden çalıştırıldığında eski chunk'ları yenileriyle değiştirir.
    collection.delete(where={"document_id": doc_id})

    for start in range(0, len(records), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_records = records[start:end]
        collection.upsert(
            ids=[record["id"] for record in batch_records],
            documents=[record["text"] for record in batch_records],
            metadatas=[record["metadata"] for record in batch_records],
            embeddings=embeddings[start:end],
        )
    print(f"  ChromaDB'ye yazıldı: {len(records)} parça")


def test_retrieval(collection: Any) -> None:
    query_text = "Üniversiteye kayıt işlemlerini hangi birim yürütür?"
    print(f"\nÖrnek arama: {query_text}")
    import ollama

    response = ollama.embed(model=EMBEDDING_MODEL, input=query_text)
    query_embedding = response["embeddings"][0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    if not docs:
        print("  Sonuç bulunamadı.")
        return

    for index, document in enumerate(docs):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        print(f"\n[{index + 1}] {metadata.get('source_file', '?')} | "
              f"{metadata.get('article_label', '?')} | "
              f"sayfa {metadata.get('start_page', '?')}-{metadata.get('end_page', '?')} | "
              f"distance={distance}")
        print(document)


def main() -> int:
    only_json = "--sadece-json" in sys.argv[1:]

    if not PDF_DIR.exists():
        print(f"PDF klasörü bulunamadı: {PDF_DIR.resolve()}")
        print("Klasörü oluşturup PDF dosyalarını içine koyun.")
        return 1

    pdf_paths = sorted(path for path in PDF_DIR.rglob("*.pdf") if path.is_file())
    if not pdf_paths:
        print(f"{PDF_DIR.resolve()} içinde PDF bulunamadı.")
        return 1

    collection = None
    if not only_json:
        try:
            import chromadb

            client = chromadb.PersistentClient(path=CHROMA_PATH)
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            print(f"ChromaDB başlatılamadı: {exc}")
            return 1

    all_records: list[dict[str, Any]] = []
    all_articles: list[dict[str, Any]] = []
    print(f"PDF sayısı: {len(pdf_paths)}")
    if only_json:
        print("Mod: sadece JSONL (embedding/ChromaDB atlanıyor)\n")
    else:
        print(f"Embedding modeli: {EMBEDDING_MODEL}")
        print(f"ChromaDB koleksiyonu: {COLLECTION_NAME}\n")

    try:
        for pdf_path in pdf_paths:
            print(f"İşleniyor: {pdf_path}")
            try:
                normalized_text = extract_and_normalize_pdf(pdf_path)
            except Exception as exc:
                print(f"  HATA: PDF okunamadı: {exc}")
                continue

            visible_text = clean_chunk_text(normalized_text)
            if len(visible_text) < 80:
                print("  UYARI: Çok az metin çıkarıldı. PDF taranmış olabilir; OCR uygulanması gerekebilir.")
                continue

            records, articles = make_chunk_records(pdf_path, normalized_text)
            if not records:
                print("  UYARI: Metin bulundu ama chunk üretilemedi.")
                continue
            if any(record["metadata"]["article_label"] == "UNPARSED" for record in records):
                print("  UYARI: MADDE başlıkları algılanamadı; genel metin chunk'ları oluşturuldu.")
                print("  prepared_chunks.jsonl dosyasını kontrol edin; bu PDF için başlık regex'i gerekebilir.")

            missing_titles = [a["metadata"]["article_label"] for a in articles if not a["metadata"]["article_title"]]
            if missing_titles:
                print(f"  UYARI: Başlığı bulunamayan maddeler: {', '.join(missing_titles)}")

            all_records.extend(records)
            all_articles.extend(articles)
            print(f"  {len(articles)} madde, {len(records)} chunk")
            if collection is None:
                continue
            try:
                upsert_pdf_records(collection, pdf_path, records)
            except Exception as exc:
                print(f"  HATA: Embedding/ChromaDB yazımı başarısız: {exc}")
                print(f"  Ollama'da '{EMBEDDING_MODEL}' modelinin hazır olduğundan emin olun.")

    except KeyboardInterrupt:
        print("İşlem kullanıcı tarafından durduruldu.")
        return 130

    if all_records:
        write_jsonl(all_records, PREPARED_JSONL)
        write_jsonl(all_articles, ARTICLES_JSONL)
        print(f"\nHazırlanan chunk'lar: {len(all_records)}")
        print(f"Kontrol dosyası: {PREPARED_JSONL.resolve()}")
        print(f"Madde dosyası: {ARTICLES_JSONL.resolve()} ({len(all_articles)} madde)")
        if collection is None:
            return 0
        print(f"ChromaDB kayıt sayısı (koleksiyondaki toplam): {collection.count()}")
        try:
            test_retrieval(collection)
        except Exception as exc:
            print(f"Örnek arama yapılamadı: {exc}")
    else:
        print("Hiç kayıt hazırlanamadı. PDF metnini ve klasör yolunu kontrol edin.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
