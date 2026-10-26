#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 RAG (PLAN_v3.md Faz 1-3): hoca ve öğrenci için TEK akış, hafızalı sohbet.

Farklar (rag_v2'ye göre):
- Rol yok, sert filtre yok: her soru tüm belgelerde aranır. Sorudan/sohbetten bilinen birim ve kapsam yalnızca
  sıralamada öncelik verir (yumuşak öncelik). Hoca öğrenci kuralını, öğrenci genel kuralı kaçırmaz.
- Hibrit arama: anlam (bge-m3 + ChromaDB) + kelime (BM25) adayları birleşir, çapraz kodlayıcı sıralar.
  "22/d", "3(f)", "39 uncu madde" gibi tam ifadeler kaçmaz.
- Kanunların geçici maddeleri ve yürürlükten kalkmış belgeler geri itilir (silinmez).
- Eşanlamlılar: "kayıt dondurma" -> izinli sayılma, "yolluk" -> harcırah, "22/d" -> doğrudan temin ...
- Atıf takibi: en iyi madde aynı belgede başka maddeye atıf yapıyorsa o madde de bağlama eklenir.
- Netleştirme (kodla, tek soru, seçenekli): birim (staj, İME, devamsızlık ...) ve satın alma kapsamı
  (BAP/araştırma projesi = 3(f) Usul ve Esasları mı, genel alım = 4734 mı).
- Hafıza yalnızca açık sohbet boyunca (Sohbet3 nesnesi); kapanınca gider.

Kullanım:  python -X utf8 rag_v3.py            (sohbet)
           python -X utf8 rag_v3.py --arama "staj defteri teslimi"
"""

from __future__ import annotations

import datetime as dt
import difflib
import functools
import json
import os
import math
import re
import sys
from collections import Counter
from typing import Any

import chromadb
import ollama

from belge_isle import ARTICLES_JSONL, CHROMA_PATH, CHUNKS_JSONL, COLLECTION_NAME, EMBEDDING_MODEL
from kaynak_dogrula import verify
from pdf_to_chromadb import normalize_for_matching
from sayi_kontrol import answer_numbers, unsupported, with_warning
from raft_format import REFUSAL_ANSWER, SYSTEM_PROMPT_V2, SYSTEM_PROMPT_V3, build_messages
from rag_v2 import (BIRIMLER, FOLLOW_UP_RE, LU_ONLY, OWN_REGULATION, SKIP_WORDS_RE, STAFF_REFUSAL, STAFF_RE,
                    DEPARTMENT_NAMES, Reranker, departments_in, units_in)

CHAT_MODEL = "kou-asistan-v4"  # 18.10: v4 %80 (v2 %76), web %73 (v2 %67)
CONTEXT_SIZE = int(os.environ.get("KOU_CONTEXT_SIZE", "4"))  # bağlama giren madde sayısı (deney için ortam değişkeni)
# Cevap modelinin sıcaklığı (deney için ortam değişkeni; kullanıcı: skora katkısı kanıtlanmadan değiştirme).
# 23.10 deneyi: 0 ile iki çalıştırma birebir aynı (182/182 cevap), skor 151/174; 0.2 ile 152/174 (fark gürültü).
TEMPERATURE = float(os.environ.get("KOU_TEMPERATURE", "0.2"))
DENSE_CANDIDATES = 80  # tam veride (124 bin parça) 40 -> 80: arama +3 soru (17.10)
SPARSE_CANDIDATES = 80
AMBIGUITY_PROBE = 6
COURSE_COLLECTION = "kou_dersler"  # dersler_isle.py
COURSE_CANDIDATES = 20
# Ders sorusu: ders adı/kodu, AKTS, müfredat, seçmeli/zorunlu, ön koşul, haftalık konular ("BLM301", "kaç akts")
COURSE_RE = re.compile(r"\bders(i|in|inin|ine|inde|ler|leri|lerin|lerinin|ten|te|e)?\b|\bakts\b|\bkredi|\bmufredat|\bders plan|\bsecmeli|"
                       r"\bzorunlu ders|\bon ?kosul|\bhaftalik|\b[a-z]{2,4} ?\d{3}\b|\bders kodu|\bmufredat")
# Devam sorusu kalıpları (rag_v2.FOLLOW_UP_RE + "bu hallerde", "bu durumda", "bu konuda" ...).
# "Bu değerlendirme için kimler sorumlu?" gibi işaret sıfatıyla başlayan soru öncekine bağlıdır (23.10: sohbet_13
# bağımsız arandı, bağlam boş kaldı). "bu yıl / bu dönem" zaman ifadesidir, devam sayılmaz.
FOLLOW_UP_V3_RE = re.compile(FOLLOW_UP_RE.pattern + r"|\bbu (hal|durum|konu|surec|islem|belge)"
                             r"|^(bu|su|o) (?!yil|donem|yariyil|hafta|ay\b|sene)\w{3,}")
GENERAL_UNITS = {"tum", "enstitu"}

# Yumuşak öncelikler (çapraz kodlayıcı puanına eklenir; puanlar kabaca -10..+10 aralığında).
PRIOR_TRANSITIONAL = -1.5  # geçici madde
PRIOR_DEPRECATED = -1.5  # yürürlükten kalkmış belge
# KOÜ sitelerinden toplanan sayfa/PDF (web_belgeleri.py): resmî yönetmelik/yönergeyle eşit eşleşirse resmî metin
# öne geçsin ("mücbir sebep" sorusunda Erasmus sözleşmesi YÖK yönetmeliğinin önüne geçiyordu).
PRIOR_WEB = -0.75
# Kural koymayan web belgeleri (rapor, plan, sunum, form, tutanak, ders notu): tam veriyle (17.10) "Senato hangi
# görevi yapar?" sorusunda performans programları, BAP sorularında sunumlar yönetmeliklerin önüne geçti.
PRIOR_REPORT = -1.5
REPORT_RE = re.compile(r"faaliyet rapor|faaliyetraporu|performans|stratejik ?plan|oz ?degerlendirme|ozdegerlendirme|"
                       r"ic ?kontrol|mali durum|malidurum|yatirim|sunum|toplanti|tutanak|hedef kart|gosterge|anket|"
                       r"eylem ?plan|\bbidr\b|\bkidr\b|iyilestirici|ders notlari|\bform|dilekce|rapor|surec yonetim")
# En iyi parçanın ham reranker puanı bunun altındaysa cevap üretilmez (bilgi yok). Testte (17.10) kapsam dışı
# soruların 7'si -0.03 ve altında, cevaplanabilir soruların en düşüğü 0.85 idi.
RELEVANCE_MIN = 0.4
# Devam sorusunda önceki cevabın dayandığı belge: "Bu hallerde öğrenci ne yapmalı?" birleşik aramada konu
# belgesini kaybedip başka belgelere gidiyordu (tam veri testi, 17.10).
PRIOR_SAME_DOC = 1.5
CONTEXT_DUP = 0.7  # iki maddenin anlamlı sözcüklerinin bu kadarı ortaksa kopya sayılır
CONTEXT_SCAN = 30  # kopya atlanırken en fazla bu kadar sıralı madde taranır
# Ders sorusunda programın adı geçiyorsa o programın ders/AKTS birimleri: "yapay zeka ve makine öğrenmesi
# dersleri" Robotik ve Yapay Zeka (önlisans) dersini getiriyordu (18.10).
PRIOR_PROGRAM = 2.0
# Kullanıcı bölümünü söylediyse ("İnşaat Mühendisliği") o bölümün belgesi öne, aynı fakültenin başka bölümünün
# belgesi geriye (23.10: İnşaat seçilmişken staj defteri cevabı Elektronik-Haberleşme esaslarından geldi).
PRIOR_OWN_DEPARTMENT = 1.5
PRIOR_OTHER_DEPARTMENT = -1.0
BOLUM_GENEL = "__fakulte_geneli__"  # kullanıcı "bölüm fark etmez" dedi: bölüme özgü belgeler daha da geriye
BOLUM_TARAMA = 12  # bölüm sorusu için bakılan en ilgili madde sayısı
PROGRAM_ASK_RE = re.compile(r"akts|kredi|mezuniyet|mezun ol|ders plan|mufredat|program yeterlil|kazanim")
BOLUM_SECENEK = 6  # en fazla bu kadar bölüm seçeneği
PROGRAM_STOP = {"ve", "ile", "bolumu", "programi", "anabilim", "dali"}
# Geçmiş eğitim-öğretim yılına ait belge (soru yıl belirtmiyorsa): 2026-2027 takvimi varken 2025-2026 geliyordu.
PRIOR_OLD_YEAR = -1.0
# Tarih soran soruda birim belli değilse üniversitenin genel (ön lisans-lisans) akademik takvimi öne çıksın:
# Hukuk/Diş/Tıp takvimleri ve geçmiş yılların takvimleri genel takvimin önüne geçiyordu (18.10).
PRIOR_GENERAL_CALENDAR = 2.0
GENERAL_CALENDAR_RE = re.compile(r"(on ?lisans|onlisans)[- ]?lisans.*akademik takvim|universite akademik takvim")
YEAR_RANGE_RE = re.compile(r"(20\d\d)\s*[-–/ _]\s*(20\d\d|\d\d)(?!\d)")
QUESTION_YEAR_RE = re.compile(r"20\d\d")
TIME_ASK_RE = re.compile(r"takvim|ne zaman|tarih|son gun|basvuru|kayit yenile|sinav program|final|butunleme")
_TODAY = dt.date.today()
CURRENT_YEAR_START = f"{_TODAY.year if _TODAY.month >= 8 else _TODAY.year - 1}-08-01"  # eğitim-öğretim yılı başı
PREVIOUS_YEAR_START = f"{int(CURRENT_YEAR_START[:4]) - 1}-08-01"
UPLOAD_DATE_RE = re.compile(r"webf_(20\d\d)(\d\d)(\d\d)_")  # web dosyalarının yüklenme tarihi (dosya adından)
# Amaç / kapsam / dayanak / tanımlar maddeleri bilgi vermiyor ama sorudaki sözcükleri içerdiği için öne çıkıyordu
# ("KYK kredisi kimlere verilir?" -> Kredi Yönetmeliği "Kapsam" maddesi). Soru bunları sormuyorsa ceza.
PRIOR_BOILERPLATE = -1.0
BOILERPLATE_RE = re.compile(r"^(amac|kapsam|dayanak|tanimlar|tanim)\b")
BOILERPLATE_ASK_RE = re.compile(r"amac|kapsam|dayanak|tanim")
PRIOR_OWN_UNIT = 1.5  # bilinen birimin kendi belgesi
PRIOR_UNKNOWN_UNIT = -1.0  # birim bilinmiyorken fakülteye özel belge (genel kural öne çıksın)
PRIOR_SCOPE = 1.5  # satın alma kapsamı netleştiyse doğru belge +, diğeri -
PRIOR_AUDIENCE = 0.75  # personel ifadesi varsa personel belgesi +, yoksa -

PROCUREMENT_3F = "Yükseköğretim Kurumlarının 4734/3(f) Kapsamındaki Alımları Usul ve Esasları"
PROCUREMENT_LAW = "4734 sayılı Kamu İhale Kanunu"
SCOPE_HINT_RE = re.compile(r"\bbap\b|\bproje|\btubitak|\barastirma|3 ?\(?f\)?|\b4734\b|kamu ihale kanun|\bkik\b")
# Üniversite genelinde merkezi yürütülen konular: bölüm/fakülte sayfaları da geçse birim sorulmaz
# (Erasmus'ta bölüm koordinatör sayfaları birime özgü kural gibi görünüp gereksiz netleştirme açıyordu).
CENTRAL_TOPIC_RE = re.compile(r"erasmus|farabi|mevlana|degisim program|uluslararasi ogrenci|yabanci uyruk|\byos\b|"
                              r"harc|katki pay|ogrenim ucret|\bburs|\bkyk\b|\byurt\b|maas|\baylik|ek ders ucret|"
                              r"yolluk|izin|emekli|sgk|akademik takvim|yatay gecis|cift anadal|\bcap\b|yandal")

# Eşanlamlılar: soruda (normalize) desen geçerse arama metnine karşılığı eklenir.
SYNONYMS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"kayd\w* dondur|kayit dondur|dondurma"), "izinli sayılma kayıt dondurma öğrenime ara verme"),
    (re.compile(r"\bbut\b|\bbute\b|\bbutten\b"), "bütünleme sınavı"),
    (re.compile(r"\bime\b"), "işletmede mesleki eğitim"),
    (re.compile(r"\bfinal"), "yarıyıl sonu sınavları final"),
    (re.compile(r"\bvize|\bara sinav"), "ara sınav yarıyıl içi sınav"),
    (re.compile(r"\byolluk"), "harcırah yolluk gündelik"),
    (re.compile(r"22 ?/ ?d|\bdogrudan temin"), "doğrudan temin 22 nci madde (d) bendi"),
    (re.compile(r"\bek odeme"), "döner sermaye ek ödeme"),
    (re.compile(r"\bbap\b"), "bilimsel araştırma projeleri BAP"),
    (re.compile(r"\bhoca"), "öğretim elemanı öğretim üyesi"),
    (re.compile(r"\bagno\b|\bortalamam"), "ağırlıklı genel not ortalaması"),
    (re.compile(r"\bcap\b"), "çift anadal programı"),
    (re.compile(r"\bdgs\b"), "dikey geçiş sınavı"),
    (re.compile(r"\berasmus\b"), "Erasmus öğrenim hareketliliği değişim programı"),
    (re.compile(r"\bharc\b|\bharci\b|\bharcimi\b"), "katkı payı öğrenim ücreti"),
    (re.compile(r"\bsatin ?alma|\bsatinalma"), "satın alma alım ihale"),
    (re.compile(r"\bmaas"), "aylık ödeme ücret"),
    (re.compile(r"\bintihal"), "benzerlik oranı intihal"),
    (re.compile(r"\bsartli gec|\bsartli"), "koşullu geçme şartlı"),
    (re.compile(r"staj defter"), "staj raporu staj dosyası staj bittikten sonra teslim"),
]

STOPWORDS = set("ve ile bir bu da de mi mı mu mü ne icin için gibi olan olarak veya ya ki daha en cok çok her "
                "nasil nasıl hangi neler nedir mi var yok ben benim bana sen biz".split())


def expand(question: str) -> str:
    normalized = normalize_for_matching(question)
    extra = [text for pattern, text in SYNONYMS if pattern.search(normalized)]
    return question + (" (" + "; ".join(extra) + ")" if extra else "")


def tokenize(text: str) -> list[str]:
    """BM25 için: küçük harf, aksansız; Türkçe ekler için kaba kök (ilk 6 harf); "22/d", "3f" korunur."""
    words = re.findall(r"[a-z0-9]+(?:/[a-z0-9]+)?", normalize_for_matching(text))
    return [w if not w.isalpha() or len(w) <= 6 else w[:6] for w in words if w not in STOPWORDS and len(w) > 1]


class BM25:
    """Ters dizinli BM25: yalnızca sorgu kelimelerini içeren parçalar puanlanır (on binlerce parçada hızlı)."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.lengths = []
        for index, document in enumerate(documents):
            counts = Counter(tokenize(document))
            self.lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                self.postings.setdefault(term, []).append((index, tf))
        self.avg = sum(self.lengths) / max(1, len(self.lengths))
        n = len(documents)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def top(self, query: str, k: int) -> list[int]:
        scores: dict[int, float] = {}
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, tf in self.postings[term]:
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[index] / self.avg)
                scores[index] = scores.get(index, 0.0) + idf * tf * (self.k1 + 1) / norm
        return [index for index, _ in sorted(scores.items(), key=lambda item: -item[1])[:k]]


@functools.lru_cache(maxsize=4096)
def program_words(belge_adi: str) -> tuple[str, ...]:
    """'AKTS Ders Bilgileri – Fen - Edebiyat Fakültesi – Yapay Zeka Ve Makine Öğrenmesi (Lisans)' -> program
    adının anlamlı sözcükleri (normalize)."""
    name = re.sub(r"\([^)]*\)", " ", belge_adi.split("–")[-1])
    return tuple(w for w in normalize_for_matching(name).split() if len(w) >= 3 and w not in PROGRAM_STOP)


@functools.lru_cache(maxsize=50000)
def title_departments_of(belge_adi: str) -> tuple[str, ...]:
    return tuple(departments_in(belge_adi))


@functools.lru_cache(maxsize=50000)
def heading_units_of(madde_basligi: str) -> tuple[str, ...]:
    return tuple(units_in(madde_basligi))


@functools.lru_cache(maxsize=50000)
def effective_unit(belge_id: str, belge_adi: str, birim: str) -> str:
    """AKTS program bilgi paketleri veride "tum" birimli; asıl birimleri başlıktaki fakülte/enstitüdür
    ("AKTS Bilgi Paketi – Mühendislik Fakültesi – Makine Mühendisliği"). 24.10 taraması: "genel kural" seçilince
    enstitü/fakülte programlarının devam kuralı geliyor, "kaç AKTS gerekir?" sorusunda fakülte sorulmuyordu."""
    if belge_id.startswith("akts_") or belge_adi.startswith("AKTS Bilgi Paketi"):
        if "enstitu" in normalize_for_matching(belge_adi):
            return "enstitu"
        found = units_in(belge_adi)
        if found:
            return found[0]
    return birim


FACULTY_WORDS_RE = re.compile(r"\b(fakultesi|fakulte|fak|yuksekokulu|yuksekokul|meslek|myo|bolumu|bolum|enstitusu)\b")


def faculty_from_text(text: str) -> str | None:
    """Kullanıcının yazdığı kısa birim adı: "mühendislik", "tıp", "hukuk", "iletişim" units_in'e takılmıyor, sessizce
    genel kurala düşüyordu (24.10 taraması). Birim adının ilk sözcükleriyle birebir eşleşme; tek aday şart."""
    found = units_in(text)
    if found:
        return found[0]
    words = FACULTY_WORDS_RE.sub(" ", normalize_for_matching(text)).split()
    if not words:
        return None
    matches = [key for key, name in BIRIMLER.items()
               if key != "genel" and FACULTY_WORDS_RE.sub(" ", normalize_for_matching(name)).split()[:len(words)] == words]
    return matches[0] if len(matches) == 1 else None


def unit_of(fields: dict[str, Any]) -> str:
    return effective_unit(fields["belge_id"], fields["belge_adi"], fields["birim"])


# Hazırlık (Yabancı Diller) belgeleri merkezi birimde: soru hazırlıkla ilgili değilse geriye ("Devamsızlık sınırı?"
# genel kural seçilince hazırlık modülü %20 kuralı geliyordu, 24.10)
HAZIRLIK_DOC_RE = re.compile(r"yabanci diller|hazirlik")
HAZIRLIK_ASK_RE = re.compile(r"hazirlik|ingilizce|yabanci dil|modul|ydyo|yeterlik|kur\b")
PRIOR_HAZIRLIK = -1.5


def program_in(question_words: frozenset[str], belge_adi: str) -> bool:
    """Programın bütün sözcükleri soruda geçiyor mu (yazım hatasına toleranslı: "örenmesi" ~ "öğrenmesi")."""
    words = program_words(belge_adi)
    return bool(words) and all(any(w == q or difflib.SequenceMatcher(None, w, q).ratio() >= 0.8
                                   for q in question_words) for w in words)


def old_academic_year(belge_adi: str) -> int:
    """Belge adı geçmiş bir eğitim-öğretim yılına ait mi ("2025-2026 Akademik Takvim", bugün 2026-2027 iken).
    Dönüş: kaç yıl eski (0 = güncel ya da yıl yok)."""
    ends = []
    for start, end in YEAR_RANGE_RE.findall(belge_adi):
        end_year = int(end) if len(end) == 4 else 2000 + int(end)
        if 0 < end_year - int(start) <= 1:
            ends.append(end_year)
    today = dt.date.today()
    current_start = today.year if today.month >= 8 else today.year - 1
    return max(0, current_start - max(ends) + 1) if ends else 0


def deprecated(fields: dict[str, Any]) -> bool:
    end = fields.get("yururluk_bitis")
    return bool(end) and dt.date.fromisoformat(end) <= dt.date.today()


def authority(article: dict[str, Any]) -> int:
    """Kopyalar arasında tercih: 0 = mevzuat arşivi / mevzuat.gov.tr / AKTS, 1 = KOÜ web dosyası ya da sayfası."""
    return 1 if article["ozellikler"]["belge_id"].startswith(("webf_", "webt_", "webd_", "web_")) else 0


class Retriever3:
    def __init__(self) -> None:
        self._word_cache: dict[str, frozenset[str]] = {}
        self.collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(COLLECTION_NAME)
        with ARTICLES_JSONL.open(encoding="utf-8") as input_file:
            self.articles = {row["id"]: row for row in map(json.loads, input_file)}
        with CHUNKS_JSONL.open(encoding="utf-8") as input_file:
            self.chunks = [json.loads(line) for line in input_file]
        self.chunk_index = {c["id"]: i for i, c in enumerate(self.chunks)}
        self.bm25 = BM25([c["metin"] for c in self.chunks])
        self.reranker = Reranker()
        # AKTS dersleri ayrı koleksiyonda (dersler_isle.py): ders sorusu sezilince orada da aranır.
        self.course_collection = None
        self.course_chunks: list[dict[str, Any]] = []
        units_path, chunks_path = ARTICLES_JSONL.parent / "dersler.jsonl", ARTICLES_JSONL.parent / "ders_parcalari.jsonl"
        if units_path.exists() and chunks_path.exists():
            try:
                self.course_collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(COURSE_COLLECTION)
            except Exception:
                self.course_collection = None
        if self.course_collection is not None:
            with units_path.open(encoding="utf-8") as input_file:
                self.articles.update({row["id"]: row for row in map(json.loads, input_file)})
            with chunks_path.open(encoding="utf-8") as input_file:
                self.course_chunks = [json.loads(line) for line in input_file]
            self.course_index = {c["id"]: i for i, c in enumerate(self.course_chunks)}
            self.course_bm25 = BM25([c["metin"] for c in self.course_chunks])

    def candidates(self, question: str, birim: str | None = None) -> list[tuple[float, dict[str, Any]]]:
        """Aday parçalar (anlam + kelime), çapraz kodlayıcı puanıyla: [(puan, parça)].
        birim verilirse arama o birimin ve genel birimlerin belgeleriyle SINIRLI yapılır ("genel": yalnız genel).
        Önceden filtre sıralamadan sonra uygulanıyordu: 80 adayın hepsi başka fakültelerden gelince bağlam boş
        kalıyor, model kaynaksız cevap uyduruyordu (23.10, Fen Edebiyat staj defteri)."""
        query = expand(question)
        vector = ollama.embed(model=EMBEDDING_MODEL, input=query)["embeddings"][0]
        allowed = None
        if birim:
            # "genel" = ön lisans/lisans genel kuralı: lisansüstü (enstitü) belgeleri girmez
            allowed = set(GENERAL_UNITS) | {birim} if birim != "genel" else {"tum"}
        where = {"birim": {"$in": sorted(allowed)}} if allowed else None
        dense = self.collection.query(query_embeddings=[vector], n_results=DENSE_CANDIDATES, include=[],
                                      where=where)["ids"][0]
        sparse = self.bm25.top(query, SPARSE_CANDIDATES * (6 if allowed else 1))
        if allowed:
            sparse = [i for i in sparse if self.chunks[i]["ozellikler"]["birim"] in allowed][:SPARSE_CANDIDATES]
        indexes = list(dict.fromkeys([self.chunk_index[i] for i in dense if i in self.chunk_index] + sparse))
        chunks = [self.chunks[i] for i in indexes]
        if self.course_collection is not None and COURSE_RE.search(normalize_for_matching(question)):
            course_dense = self.course_collection.query(query_embeddings=[vector], n_results=COURSE_CANDIDATES,
                                                        include=[])["ids"][0]
            course_ids = list(dict.fromkeys([self.course_index[i] for i in course_dense if i in self.course_index]
                                            + self.course_bm25.top(query, COURSE_CANDIDATES)))
            chunks += [self.course_chunks[i] for i in course_ids]
        scores = self.reranker.scores(query, [c["metin"] for c in chunks]) if chunks else []
        return list(zip(scores, chunks))

    def rank_units(self, scored: list[tuple[float, dict[str, Any]]], question: str, birim: str | None,
                   kapsam: str | None, priors: bool = True, onceki_belgeler: set[str] | None = None,
                   bolum: str | None = None) -> list[tuple[float, dict[str, Any]]]:
        """Parça puanı + öncelikler -> bağlam birimi puanı (en iyi parçası). Sıralı [(puan, madde)].
        onceki_belgeler: devam sorusunda önceki cevabın belgeleri (aynı belgenin başka maddesi öne çıkar)."""
        normalized = normalize_for_matching(question)
        staff = bool(STAFF_RE.search(normalized))
        question_words = frozenset(re.findall(r"\w{3,}", normalized))
        asks_year = bool(QUESTION_YEAR_RE.search(question))
        asks_time = bool(TIME_ASK_RE.search(normalized))
        asks_boilerplate = bool(BOILERPLATE_ASK_RE.search(normalized))
        asks_hazirlik = bool(HAZIRLIK_ASK_RE.search(normalized))
        bolum = bolum or next(iter(departments_in(question)), None)
        best: dict[str, float] = {}
        for score, chunk in scored:
            fields = chunk["ozellikler"]
            if fields["belge_id"].startswith(("aktsders_", "akts_")) and program_in(question_words, fields["belge_adi"]):
                score += PRIOR_PROGRAM
            age = 0 if asks_year else old_academic_year(f"{fields['belge_adi']} {fields['belge_id']}")
            if age:
                score += PRIOR_OLD_YEAR * min(age, 2)  # geçen yıl -1, daha eskisi -2
            if (asks_time and not birim and not age
                    and GENERAL_CALENDAR_RE.search(normalize_for_matching(fields["belge_adi"]))):
                score += PRIOR_GENERAL_CALENDAR
            elif (asks_time and fields["belge_id"].startswith("webd_") and fields.get("yururluk_tarihi", "")
                  and fields["yururluk_tarihi"] < CURRENT_YEAR_START):
                score += PRIOR_OLD_YEAR  # tarih soran soruda geçen yılın duyurusu (takvim, son başvuru)
            elif asks_time and (upload := UPLOAD_DATE_RE.match(fields["belge_id"])) and \
                    "-".join(upload.groups()) < PREVIOUS_YEAR_START:
                # Yılı başlıkta değil yalnız dosya adında olan eski belge ("Bütünleme ne zaman?" -> 2021 Tıp
                # bütünleme duyurusu, 24.10). Eşik bir önceki eğitim yılı: yeni takvimler yaz başında yayımlanıyor.
                score += 2 * PRIOR_OLD_YEAR
            article = self.articles.get(f"{fields['madde_id']}#{fields['madde_parca_no']}")
            if (not asks_boilerplate and article is not None
                    and BOILERPLATE_RE.search(normalize_for_matching(article["ozellikler"].get("madde_basligi") or ""))):
                score += PRIOR_BOILERPLATE
            if fields["madde_no"].startswith("geçici"):
                score += PRIOR_TRANSITIONAL
            if deprecated(fields):
                score += PRIOR_DEPRECATED
            if fields["belge_id"].startswith("web"):
                score += PRIOR_WEB
                if REPORT_RE.search(normalize_for_matching(fields["belge_adi"] + " " + fields["belge_id"])):
                    score += PRIOR_REPORT
            if not asks_hazirlik and HAZIRLIK_DOC_RE.search(normalize_for_matching(fields["belge_adi"])):
                score += PRIOR_HAZIRLIK
            if priors:
                unit = unit_of(fields)
                if birim and birim != "genel":
                    # Birim bilindiğinde (soruda geçti ya da kullanıcı seçti) başka fakültenin belgesi bağlama hiç
                    # girmez: yalnızca puan cezası verilince model Mühendislik seçilmişken MYO yönergesini kaynak
                    # gösterdi (web denemesi, 17.10).
                    if unit == birim or (unit in GENERAL_UNITS
                                         and birim in heading_units_of(fields.get("madde_basligi") or "")):
                        # Merkezi belgenin o birime ait bölümü ("Kurum içi yatay geçiş kontenjanları – Hereke Asım
                        # Kocabıyık MYO") birimin kendi belgesi sayılır; yoksa birimin genel yönetmeliği onu
                        # geçiyordu (23.10, sohbet_03).
                        score += PRIOR_OWN_UNIT
                    elif unit not in GENERAL_UNITS:
                        continue
                    elif birim in OWN_REGULATION and fields["belge_adi"] in LU_ONLY:
                        continue
                elif birim == "genel" and unit != "tum":
                    # Kullanıcı "birimden bağımsız genel kural" dedi: fakülteye özel belge (bölüm duyurusu dahil)
                    # bağlama girmez (23.10: "Diğer" seçilince İnşaat Mühendisliği duyurusundan cevap geliyordu).
                    continue
                elif unit not in GENERAL_UNITS:
                    score += PRIOR_UNKNOWN_UNIT
                if kapsam == "3f":
                    score += PRIOR_SCOPE if fields["belge_adi"] == PROCUREMENT_3F else \
                        (-PRIOR_SCOPE if fields["belge_adi"] == PROCUREMENT_LAW else 0)
                elif kapsam == "4734":
                    score += PRIOR_SCOPE if fields["belge_adi"] == PROCUREMENT_LAW else \
                        (-PRIOR_SCOPE if fields["belge_adi"] == PROCUREMENT_3F else 0)
                if fields["hedef_kitle"] == "personel":
                    score += PRIOR_AUDIENCE if staff else -PRIOR_AUDIENCE
                if onceki_belgeler and fields["belge_id"] in onceki_belgeler:
                    score += PRIOR_SAME_DOC
                if unit not in GENERAL_UNITS and (bolum or (birim and birim != "genel")):
                    # Bölüm söylenmediyse fakülte geneli belge, tek bir bölümün esaslarından önce gelir
                    title_departments = title_departments_of(fields["belge_adi"])
                    if bolum and bolum in title_departments:
                        score += PRIOR_OWN_DEPARTMENT
                    elif title_departments:
                        score += (2 * PRIOR_OTHER_DEPARTMENT if bolum == BOLUM_GENEL else
                                  PRIOR_OTHER_DEPARTMENT if bolum else PRIOR_OTHER_DEPARTMENT / 2)
            parent = f"{fields['madde_id']}#{fields['madde_parca_no']}"
            if parent in self.articles and score > best.get(parent, -1e9):
                best[parent] = score
        ranked = sorted(best.items(), key=lambda item: -item[1])
        return [(score, self.articles[parent]) for parent, score in ranked]

    def _words(self, article: dict[str, Any]) -> frozenset[str]:
        words = self._word_cache.get(article["id"])
        if words is None:
            words = frozenset(re.findall(r"\w{4,}", normalize_for_matching(article["metin"])))
            self._word_cache[article["id"]] = words
        return words

    def _distinct(self, ranked: list[tuple[float, dict[str, Any]]], size: int) -> list[dict[str, Any]]:
        """Sıralı listeden `size` madde; başka belgedeki neredeyse aynı madde (aynı yönergenin kopyası, aynı dersin
        başka programdaki kaydı) bağlama ikinci kez girmez, yeri bir sonraki maddeye kalır. Kopyalardan yetkili
        olan (web dışı: yönetmelik/kanun arşivi) tutulur. Tam veri denetimi (19.10): 167 test sorusunun 58'inde
        bağlamda kopya vardı; model bazen yanlış kopyayı kaynak gösteriyordu."""
        chosen: list[dict[str, Any]] = []
        for _, article in ranked[:CONTEXT_SCAN]:
            words = self._words(article)
            duplicate = None
            for index, other in enumerate(chosen):
                if other["ozellikler"]["belge_id"] == article["ozellikler"]["belge_id"]:
                    continue
                other_words = self._words(other)
                if len(words & other_words) >= CONTEXT_DUP * max(1, min(len(words), len(other_words))):
                    duplicate = index
                    break
            if duplicate is not None:
                if authority(article) < authority(chosen[duplicate]):
                    chosen[duplicate] = article
                continue
            chosen.append(article)
            if len(chosen) == size:
                break
        return chosen

    def context(self, ranked: list[tuple[float, dict[str, Any]]], size: int = CONTEXT_SIZE) -> list[dict[str, Any]]:
        """İlk `size` farklı madde + en iyi maddenin atıf yaptığı (aynı belgedeki) madde, yer varsa (en fazla size+1)."""
        chosen = self._distinct(ranked, size)
        if chosen:
            # Uzun madde parçalara bölünmüşse en iyi parçanın komşusu da gelsin (koşulların devamı çoğu zaman
            # sonraki parçada; model yalnız bir parçayı görüp eksik cevap veriyordu).
            top_id, _, part = chosen[0]["id"].rpartition("#")
            ids = {a["id"] for a in chosen}
            for neighbor in (f"{top_id}#{int(part) + 1}", f"{top_id}#{int(part) - 1}") if part.isdigit() else ():
                if neighbor in self.articles and neighbor not in ids:
                    chosen.append(self.articles[neighbor])
                    return chosen
            have = {a["ozellikler"]["madde_id"] for a in chosen}
            for target in chosen[0]["ozellikler"].get("atiflar", []):
                referenced = self.articles.get(f"{target}#1")
                if referenced and target not in have:
                    chosen.append(referenced)
                    break
        return chosen


class Asistan3:
    def __init__(self, model: str = CHAT_MODEL, system: str | None = None) -> None:
        self.model = model
        # KOU_PROMPT=v2 eski (kısa cevap) talimatı; varsayılan ayrıntılı v3 talimatı
        self.system = system or (SYSTEM_PROMPT_V2 if os.environ.get("KOU_PROMPT") == "v2" else SYSTEM_PROMPT_V3)
        self.retriever = Retriever3()

    def generate(self, question: str, context: list[dict[str, Any]]) -> str:
        response = ollama.chat(model=self.model, messages=build_messages(question, context, system=self.system),
                               think=False, options={"temperature": TEMPERATURE, "num_ctx": 8192})
        answer = response["message"]["content"].strip()
        personel = sum(a["ozellikler"]["hedef_kitle"] == "personel" for a in context)
        if answer.startswith(REFUSAL_ANSWER.split(".")[0]) and (
                STAFF_RE.search(normalize_for_matching(question)) or personel > len(context) / 2):
            answer = STAFF_REFUSAL  # eğitimdeki red cümlesi Öğrenci İşleri'ne yönlendiriyor
        answer = verify(answer, context)[0]
        # Bağlamda geçmeyen tutar/tarih/gün sayısı varsa uyarı (19.10: uydurma sayıya karşı güvenlik ağı)
        missing = unsupported(answer, context, question)
        body = re.sub(r"Not: Bu bilgi[^\n]*", "", answer.split("\nKaynak:", 1)[0])
        written = {w for _, w in answer_numbers(body)}
        if missing and set(missing) >= written:
            # Cevabın dayandığı sayıların HİÇBİRİ kaynakta yok: uydurma ("Gebze'den 402 numaralı otobüs", 22.10)
            staff = STAFF_RE.search(normalize_for_matching(question)) or personel > len(context) / 2
            return STAFF_REFUSAL if staff else REFUSAL_ANSWER
        return with_warning(answer, missing)


# ---- Harç / katkı payı yönlendirmesi: "harç ödeyecek miyim?" sorusunun cevabı öğrencinin durumuna bağlı
# (birinci öğretimde normal süredeki öğrencinin katkı payını devlet öder; süreyi aşan, ikinci/uzaktan öğretim,
# lisansüstü ve uluslararası öğrenci öder, tutar fakülteye göre değişir). Asistan önce bunları sorar (18.10).
HARC_RE = re.compile(r"\bharc(?!irah|ama|an)|katki pay|ogrenim ucret|okul ucret")  # "harcırah" harç değil (19.10)
# Özel durumlu ücret soruları ağaçla değil doğrudan aramayla cevaplanır (denklik/lisans tamamlama, yaz okulu, iade...)
HARC_SKIP_RE = re.compile(r"denklik|klinik pratik|lisans tamamla|yaz okulu|tek ders|iade|depremzede|gazze|sehit|"
                          r"gazi|engelli|bilimsel hazirlik|ozel ogrenci|sybs")
HARC_ASK_RE = re.compile(r"\bode|\boduy|\bodu|ne kadar|kac (tl|lira|para)|var mi|muaf|gerek|yatir|tutar|miktar")
HARC_TUR = [("birinci", "Birinci öğretim (normal / gündüz)"), ("ikinci", "İkinci öğretim (gece, İÖ)"),
            ("uzaktan", "Uzaktan öğretim"), ("lisansustu", "Lisansüstü (yüksek lisans / doktora)"),
            ("yabanci", "Uluslararası (yabancı uyruklu) öğrenci")]
HARC_TUR_RE = [("ikinci", r"ikinci ogretim|\bio\b|gece|aksam"), ("uzaktan", r"uzaktan"),
               ("lisansustu", r"lisansustu|yuksek lisans|doktora|tezli|tezsiz"),
               ("yabanci", r"yabanci|uluslararasi|yos\b|international"), ("birinci", r"birinci ogretim|normal ogretim|gunduz")]
# Yabancı uyruklu: iki ayrı tablo (YKS ile yerleşen / uluslararası öğrenci kabul) ve uluslararası kabulde kayıt yılı
HARC_KABUL = [("uluslararasi", "Uluslararası öğrenci kabul kontenjanı (YÖS / uluslararası başvuru)"),
              ("yks", "YKS ile yerleştim (yabancı uyruklu)")]
HARC_KABUL_RE = [("yks", r"\byks\b|osym"), ("uluslararasi", r"yos|uluslararasi|kontenjan")]
HARC_YIL = [("2019", "2019-2020 ve öncesinde kayıt oldum"),
            ("2020", "2020-2021 ile 2024-2025 arasında kayıt oldum (numaram 20-24 ile başlıyor)"),
            ("2025", "2025-2026 ve sonrasında kayıt oldum (numaram 25 ya da 26 ile başlıyor)")]
HARC_YIL_RE = [("2025", r"\b2[56]\b|202[56]"), ("2020", r"\b2[0-4]\b|202[0-4]"), ("2019", r"201\d|oncesi")]
# Her dalın arama metni: tablodaki sütun başlığıyla aynı sözcükler (tutarın doğru satır/sütundan gelmesi için)
HARC_ARAMA = {
    "normal": "birinci öğretim normal öğrenim süresi içinde katkı payı Devlet tarafından karşılanmaktadır",
    "asti": "Normal Öğrenim Süresini Aşan Öğrencilerin Katkı Payı Ücretleri Güz YY Taksiti Bahar YY Taksiti Toplam",
    "ikinci": "İkinci Öğretim ve Uzaktan Öğretim Öğrencilerinin Öğrenim Ücretleri Güz YY Taksiti Bahar Toplam",
    "uzaktan": "İkinci Öğretim ve Uzaktan Öğretim Öğrencilerinin Öğrenim Ücretleri Güz YY Taksiti Bahar Toplam",
    "yks": "YKS İle Yerleşen Yabancı Uyruklu Öğrencilerin Öğrenim Ücretleri Güz YY Taksiti Bahar Toplam",
    "uluslararasi": "Uluslararası Öğrenci Kabul Kontenjanları kapsamında kayıtlı öğrencilerin yıllık öğrenim ücretleri",
    "lisansustu": "lisansüstü katkı payı öğrenim ücreti enstitü",
}
HARC_YIL_ARAMA = {"2019": "2019-2020 Eğitim Öğretim Yılı ile Önceki Yıllarda Kayıtlı",
                  "2020": "2020-2021 Eğitim Öğretim Yılı ve Sonrasında Kayıt Olan (Öğrenci Numarası 20,21,22,23,24)",
                  "2025": "2025-2026 Eğitim Öğretim Yılı ve Sonrasında Kayıt Olan (Öğrenci numarası 25 ve 26)"}
# Karar ağacının vardığı hücre tablodan aynen okunur (model tutarı bazen yazmıyor/karıştırıyordu):
# tablo_metin.py'nin ürettiği satırlar "Fakülte — Sütun grubu (tablo adı): Güz ...; Bahar ...; Toplam ...".
FEE_TABLE_GLOB = "*harc-ucretleri*.pdf.txt"
# Karar ağacının yaprakları: kural cümlesi KOÜ ÖİDB "Katkı Payı/Öğrenim Ücreti Genel Bilgiler 2026" (Ek-1) ve ücret
# tablosundan; model burada cevap üretmez (eski duyurulara dayanıp tabloyla çelişebiliyordu).
HARC_KURAL = {
    "normal": "Birinci öğretimde programının normal öğrenim süresi içindeysen katkı payını Devlet karşılar; harç "
              "(katkı payı) ödemezsin. Normal süre sonunda mezun olamazsan katkı payı ödemeye başlarsın.",
    "asti": "Birinci öğretimde programının normal öğrenim süresini aştığın için katkı payı ödemen gerekiyor. Ücret "
            "güz ve bahar yarıyılında iki eşit taksitte ödenir.",
    "ikinci": "İkinci öğretim öğrencileri program süresi boyunca öğrenim ücreti öder. Ücret güz ve bahar yarıyılında "
              "iki eşit taksitte ödenir.",
    "uzaktan": "Uzaktan öğretim öğrencileri program süresi boyunca öğrenim ücreti öder (ek olarak materyal ücreti "
               "alınır). Ücret güz ve bahar yarıyılında iki eşit taksitte ödenir.",
    "yks": "YKS ile yerleşen yabancı uyruklu öğrenciler öğrenim ücreti öder. Ücret güz ve bahar yarıyılında iki eşit "
           "taksitte ödenir.",
    "uluslararasi": "Uluslararası öğrenci kabul kontenjanıyla kayıtlı öğrenciler, kayıt yılına göre belirlenen yıllık "
                    "öğrenim ücretini öder.",
    "lisansustu": "Lisansüstü öğrenciler normal öğrenim süresini aşarsa katkı payı öder; ikinci öğretim (tezsiz) "
                  "lisansüstü ücretleri enstitü web sayfasında ilan edilir.",
}
HARC_NOT = ("Not: Engelli öğrencilere engellilik oranında indirim yapılır; zorunlu stajı kalan öğrenciler katkı "
            "payından muaftır. Ödeme Ziraat Bankası kanallarından, akademik takvimdeki tarihlerde yapılır.")
FEE_TABLE_SOURCE = "Öğrenci İşleri Daire Başkanlığı – 2026-2027 KOÜ Öğrenci Katkı Payı/Öğrenim Ücretleri Tablosu"
HARC_SUTUN = {"asti": "Normal Öğrenim Süresini Aşan", "ikinci": "İkinci Öğretim ve Uzaktan",
              "uzaktan": "İkinci Öğretim ve Uzaktan", "yks": "YKS İle Yerleşen",
              "2019": "2019-2020 Eğitim Öğretim Yılı ile Önceki", "2020": "2020-2021 Eğitim Öğretim Yılı",
              "2025": "2025-2026 Eğitim Öğretim Yılı"}


@functools.lru_cache(maxsize=1)
def fee_rows() -> tuple[str, ...]:
    files = sorted((ARTICLES_JSONL.parents[1] / "web_tam" / "metin").glob(FEE_TABLE_GLOB))
    if not files:
        return ()
    text = files[-1].read_text(encoding="utf-8", errors="ignore")
    section = text.split("__TABLO_SON__", 1)[0] if text.startswith("__TABLO__") else ""
    return tuple(line for line in section.split("\n") if " — " in line)


def fee_units() -> list[tuple[str, str]]:
    """Ücret tablosundaki fakülte/yüksekokul adları (fakülte sorusunun seçenekleri)."""
    names, seen = [], set()
    for row in fee_rows():
        label = re.sub(r"[¹²³]|\s*\(.*$", "", row.partition(" — ")[0]).strip()
        key = re.sub(r"[^a-z]", "", normalize_for_matching(label))
        if label and key not in seen and not re.search(r"Yabancı Diller|Lisansüstü", label):
            seen.add(key)
            names.append(label)
    return [(name, name) for name in names]


def fee_row(unit_name: str, column: str) -> str:
    """Fakülte adı + sütun grubuna uyan tablo satırları (Fen-Edebiyat'ta Fen/Sosyal bölümler iki satır)."""
    if "Meslek Yüksekokulu" in unit_name:  # tabloda tüm MYO'lar tek satır
        unit_name = "Meslek Yüksekokulları"
    key = re.sub(r"[^a-z]", "", normalize_for_matching(unit_name.replace("Fakültesi", "").replace("Yüksekokulu", "")))
    if not key:
        return ""
    found = []
    for row in fee_rows():
        label, _, rest = row.partition(" — ")
        if key in re.sub(r"[^a-z]", "", normalize_for_matching(label)) and rest.startswith(column):
            value = re.sub(r"\s*\([^)]*(TABLOSU|ÖĞRENCİLER)\)", "", rest)
            found.append(f"{label}: {value}")
    return "\n".join(found)


HARC_SURE = [("normal", "Evet, normal öğrenim süresi içindeyim"), ("asti", "Hayır, normal süreyi aştım (uzattım)")]
HARC_SURE_RE = [("asti", r"uzat|sureyi as|sure(m)?i? doldu|ek sure|alttan|mezun olamadim|hayir"),
                ("normal", r"normal sure|sure(m)? icinde|evet")]


def harc_secimi(metin: str, desenler: list[tuple[str, str]]) -> str | None:
    normalized = normalize_for_matching(metin)
    return next((anahtar for anahtar, desen in desenler if re.search(desen, normalized)), None)


# ---- Genel karar ağaçları: cevabı kullanıcının durumuna bağlı sorularda (kayıt dondurma, devamsızlık, yatay
# geçiş, KYK, yıllık izin, görevlendirme) önce durum sorulur; sonra soru durumla ve o dalın arama sözcükleriyle
# zenginleştirilip aranır ve cevaplanır (kullanıcı istedi: "detay isteyen sorularda karar ağacı gibi yönlensin").
# Adım: alan, soru, seçenekler [(anahtar, ad, serbest metin deseni)], arama {anahtar: ek arama sözcükleri},
# isteğe bağlı "kosul" {alan: [anahtarlar]} (adım yalnız o cevaplarda sorulur).
# Ağaç yalnız kişinin KENDİ durumunu sorduğu sorularda açılır ("dondurabilir miyim", "gideceğim", "hakkım").
# "Koordinatör kim tarafından görevlendirilir?" gibi kural soruları da ağaca giriyordu (19.10 testi: hoca %83 -> %72).
AGAC_KISISEL_RE = re.compile(r"\b(ben|benim|bana)\b|m[iu]y[iu]m\b|ecegim|acagim|istiyorum|isterim|hakkim|"
                             r"ne yapmal|nasil (alabil|yapabil|basvur|al[ia]r|yapar)|alabilir miyim|edebilir miyim|"
                             r"gidecek|gitmek ist|katilacag|katilmak ist|"
                             r"ucretim|dersim|yukum|bursum|stajim|harcim|iznim|kaydim|sinavim")
AGAC_MAX_WORDS = 12
AGAC_ATLA_RE = re.compile(r"\bhangi\b|\bkim(ler)?\b|ne zaman|tarih|son gun|hangi form|dilekce|nereye basvur|kime basvur|e-?posta|telefon")
AGACLAR: list[dict[str, Any]] = [
    {"ad": "kayit_dondurma", "tetik": r"kayit(imi|imi)? dondur|kaydimi dondur|kayit dondurma|izinli sayil|dondurabilir",
     "adimlar": [
         {"alan": "duzey", "soru": "Hangi düzeyde okuyorsun?",
          "secenekler": [("lisans", "Ön lisans / lisans", r"\blisans(?!ustu)|on ?lisans|myo|fakulte"),
                         ("lisansustu", "Lisansüstü (yüksek lisans / doktora)", r"lisansustu|yuksek lisans|doktora")],
          "arama": {"lisans": "KOÜ Ön Lisans ve Lisans Eğitim-Öğretim Yönetmeliği İzinli sayılma madde 39",
                    "lisansustu": "Lisansüstü Eğitim ve Öğretim Yönetmeliği kayıt dondurma izinli sayılma"}},
         {"alan": "neden", "soru": "Kaydını hangi nedenle dondurmak istiyorsun?",
          "secenekler": [("saglik", "Sağlık sorunu (raporlu)", r"saglik|hasta|rapor|ameliyat"),
                         ("askerlik", "Askerlik", r"asker"),
                         ("ekonomik", "Ekonomik / ailevi neden", r"ekonomik|maddi|aile|para|calis"),
                         ("afet", "Doğal afet", r"afet|deprem|sel|yangin"),
                         ("yurtdisi", "Yurt dışında eğitim, staj veya burs", r"yurt ?disi|burs|erasmus|staj"),
                         ("tutukluluk", "Tutukluluk", r"tutuk"),
                         ("diger", "Diğer bir neden", r"diger|baska")],
          "arama": {"saglik": "sağlık raporu belgelenmesi", "askerlik": "askere alınma",
                    "ekonomik": "ekonomik nedenler belgelenmesi", "afet": "doğal afet", "yurtdisi":
                    "yurt dışında burslu staj eğitim", "tutukluluk": "tutukluluk hali", "diger": "haklı ve geçerli nedenler"}},
     ]},
    {"ad": "devamsizlik", "tetik": r"devamsiz|devam zorun|devam sart|devam kosul|yoklama|kac (ders|hafta|saat)a? (gelme|girme|kacir)|derse gelmezsem",
     "adimlar": [
         {"alan": "ders", "soru": "Devam durumunu hangi ders türü için soruyorsun?",
          "secenekler": [("teorik", "Teorik ders (ilk kez alıyorum)", r"teori"),
                         ("uygulama", "Uygulama / laboratuvar / atölye dersi", r"uygulama|laboratuvar|lab\b|atolye|klinik"),
                         ("tekrar", "Daha önce devam şartını sağladığım (tekrar aldığım) ders", r"tekrar|alttan|ikinci kez")],
          "arama": {"teorik": "Devam koşulu teorik derslerin en az %70'ine devam",
                    "uygulama": "Devam koşulu diğer öğretim türlerinin en az %80'ine devam uygulama laboratuvar",
                    "tekrar": "Devam koşulu daha önce devam şartını yerine getirdiği derslere devam zorunluluğu tekrar"}},
     ]},
    {"ad": "yatay_gecis", "tetik": r"yatay gecis",
     "adimlar": [
         {"alan": "tur", "soru": "Hangi yatay geçişi yapmak istiyorsun?",
          "secenekler": [("kurum_ici", "Kurum içi (KOÜ'de başka bir bölüme)", r"kurum ?ici|ayni universite|bolum degistir"),
                         ("kurumlar_arasi", "Kurumlar arası (başka üniversiteden KOÜ'ye, not ortalamasıyla)", r"kurumlar ?arasi|baska universite|not ortalama|basariya dayali"),
                         ("ek_madde", "Merkezi yerleştirme puanıyla (Ek Madde-1)", r"ek ?madde|merkezi yerlestirme|yks puan|puanla"),
                         ("yurtdisi", "Yurt dışındaki bir üniversiteden", r"yurt ?disi")],
          "arama": {"kurum_ici": "kurum içi yatay geçiş başarıya dayalı kılavuz şartlar",
                    "kurumlar_arasi": "kurumlar arası yatay geçiş başarıya dayalı genel not ortalaması şartlar kılavuz",
                    "ek_madde": "Ek Madde-1 merkezi yerleştirme puanı ile yatay geçiş şartlar taban puan",
                    "yurtdisi": "yurt dışındaki yükseköğretim kurumlarından yatay geçiş"}},
     ]},
    {"ad": "kyk", "tetik": r"\bkyk\b|ogrenim kredisi|ogrenim kredi|kredi yurtlar|devlet bursu|gsb burs|yurt basvur",
     "adimlar": [
         {"alan": "tur", "soru": "KYK'nın hangi hizmetini soruyorsun?",
          "secenekler": [("kredi", "Öğrenim kredisi", r"kredi"), ("burs", "Burs", r"burs"),
                         ("yurt", "Yurt", r"yurt")],
          "arama": {"kredi": "Gençlik ve Spor Bakanlığı Kredi Yönetmeliği kredi tahsisi kredi başvurularının "
                             "değerlendirilmesi belge ve durum araştırması",
                    "burs": "Gençlik ve Spor Bakanlığı Burs Yönetmeliği burs verilmesi şartları",
                    "yurt": "Gençlik ve Spor Bakanlığı Yurt Hizmetleri Yönetmeliği yurda kabul şartları"}},
     ]},
    {"ad": "staj", "tetik": r"\bstaj",
     "adimlar": [
         {"alan": "tur", "soru": "Hangi uygulamayı soruyorsun?",
          "secenekler": [("zorunlu", "Zorunlu staj (programımın şartı)", r"zorunlu"),
                         ("isteğe", "İsteğe bağlı (gönüllü) staj", r"istege|gonullu|yaz staji"),
                         ("ime", "İşletmede mesleki eğitim (İME)", r"\bime\b|isletmede mesleki")],
          "arama": {"zorunlu": "zorunlu staj yönergesi", "isteğe": "isteğe bağlı gönüllü staj",
                    "ime": "işletmede mesleki eğitim İME yönergesi"}},
         {"alan": "konu", "soru": "Stajla ilgili neyi öğrenmek istiyorsun?",
          "secenekler": [("basvuru", "Başvuru ve kabul (staj yeri, belgeler)", r"basvur|kabul|yer|belge|form"),
                         ("sure", "Süre ve zamanlama", r"sure|kac gun|ne zaman|donem|yaz"),
                         ("sigorta", "Sigorta (SGK) işlemleri", r"sigorta|sgk"),
                         ("degerlendirme", "Staj defteri, değerlendirme ve kabul", r"defter|rapor|degerlendir|sicil|not")],
          "arama": {"basvuru": "staj başvurusu staj yeri kabul belgesi", "sure": "staj süresi iş günü",
                    "sigorta": "staj sigorta SGK 5510 işlemleri", "degerlendirme": "staj defteri staj değerlendirme komisyonu"}},
     ]},
    {"ad": "mazeret", "tetik": r"mazeret",
     "adimlar": [
         {"alan": "sinav", "soru": "Hangi sınavı kaçırdın / kaçıracaksın?",
          "secenekler": [("ara", "Ara sınav (vize)", r"\bara\b|vize|yariyil ici"),
                         ("final", "Final (yarıyıl sonu) ya da bütünleme", r"final|yariyil sonu|butunleme|\bbut\b")],
          "arama": {"ara": "Mazeret sınavı ara sınavı için geçerlidir yönetim kurulu",
                    "final": "Mazeret sınavı ara sınav için geçerlidir yarıyıl sonu sınavı bütünleme mazeret"}},
         {"alan": "neden", "soru": "Mazeretin ne?",
          "secenekler": [("saglik", "Sağlık (rapor)", r"saglik|hasta|rapor"),
                         ("resmi", "Resmî görev / spor müsabakası / yarışma", r"resmi|gorev|spor|musabaka|yarisma"),
                         ("ailevi", "Ailevi neden (yakın kaybı vb.)", r"aile|vefat|olum|cenaze"),
                         ("diger", "Diğer", r"diger|baska")],
          "arama": {"saglik": "sağlık raporu mazeret belgesi", "resmi": "resmi görevlendirme spor müsabakası mazeret",
                    "ailevi": "birinci derece yakının ölümü mazeret", "diger": "haklı ve geçerli mazeret"}},
     ]},
    {"ad": "ek_ders", "tetik": r"ek ders|ders yuk",
     "adimlar": [
         {"alan": "unvan", "soru": "Kadro unvanın nedir?",
          "secenekler": [("prof", "Profesör", r"\bprof"), ("doc", "Doçent", r"\bdoc"),
                         ("dr", "Doktor öğretim üyesi", r"dr\.? ?ogr|doktor ogretim"),
                         ("ogrgor", "Öğretim görevlisi", r"ogretim gorevlisi|ogr\.? ?gor"),
                         ("arsgor", "Araştırma görevlisi", r"arastirma gorevlisi|ars\.? ?gor")],
          "arama": {"prof": "profesör haftalık ders yükü ek ders", "doc": "doçent haftalık ders yükü ek ders",
                    "dr": "doktor öğretim üyesi haftalık ders yükü ek ders",
                    "ogrgor": "öğretim görevlisi haftalık ders yükü ek ders",
                    "arsgor": "araştırma görevlisi ders görevlendirme ek ders"}},
         {"alan": "konu", "soru": "Ek dersle ilgili neyi öğrenmek istiyorsun?",
          "secenekler": [("yuk", "Zorunlu ders yükü ve ek ders üst sınırı", r"yuk|sinir|kac saat|ust"),
                         ("ucret", "Ek ders ücretinin hesaplanması / ödenmesi", r"ucret|odeme|hesap|ne zaman"),
                         ("ikinci", "İkinci öğretim / başka birimde ders verme", r"ikinci ogretim|baska (birim|universite)|gorevlendirme")],
          "arama": {"yuk": "2914 sayılı Kanun madde 11 haftalık ders yükü ek ders üst sınırı",
                    "ucret": "ek ders ücreti ödenmesi hesaplanması gösterge",
                    "ikinci": "ikinci öğretim ek ders başka yükseköğretim kurumunda görevlendirme ek ders ücreti"}},
     ]},
    {"ad": "burs", "tetik": r"\bburs",
     "adimlar": [
         {"alan": "tur", "soru": "Hangi bursu soruyorsun?",
          "secenekler": [("kyk", "KYK (devlet) bursu veya kredisi", r"\bkyk\b|devlet|gsb|kredi"),
                         ("yemek", "KOÜ yemek bursu", r"yemek"),
                         ("kismi", "Kısmi zamanlı çalışma (öğrenci asistanlığı)", r"kismi zaman|calisma|asistan"),
                         ("lisansustu", "Lisansüstü burs (YÖK 100/2000, TÜBİTAK)", r"100/2000|yok bursu|tubitak|lisansustu|doktora"),
                         ("uluslararasi", "Uluslararası öğrenci / Erasmus hibesi", r"erasmus|uluslararasi|yabanci|hibe")],
          "arama": {"kyk": "Gençlik ve Spor Bakanlığı Burs Yönetmeliği burs verilmesi şartları",
                    "yemek": "KOÜ yemek bursu başvuru şartları Sağlık Kültür ve Spor Daire Başkanlığı",
                    "kismi": "kısmi zamanlı öğrenci çalıştırma başvuru şartları ücret",
                    "lisansustu": "YÖK 100/2000 doktora bursu lisansüstü burs şartları",
                    "uluslararasi": "uluslararası öğrenci burs Erasmus hibe miktarları"}},
     ]},
    {"ad": "yillik_izin", "tetik": r"yillik izin|kac gun izin|izin hakk|izin sure",
     "adimlar": [
         {"alan": "hizmet", "soru": "Toplam hizmet süren ne kadar?",
          "secenekler": [("az", "1 yıldan 10 yıla kadar (10 yıl dahil)", r"^(?!.*fazla).*(\b[1-9] ?yil|10 yil|on yil|yeni)"),
                         ("cok", "10 yıldan fazla", r"1[1-9] ?yil|[2-4]\d ?yil|fazla")],
          "arama": {"az": "657 sayılı Devlet Memurları Kanunu yıllık izin hizmeti bir yıldan on yıla kadar yirmi gün",
                    "cok": "657 sayılı Devlet Memurları Kanunu yıllık izin hizmeti on yıldan fazla otuz gün"}},
     ]},
    {"ad": "gorevlendirme", "tetik": r"gorevlendirme|kongre|sempozyum|konferans(a|ya)? katil|yurt ?disina git",
     "adimlar": [
         {"alan": "yer", "soru": "Görevlendirme yurt içi mi, yurt dışı mı?",
          "secenekler": [("yurtici", "Yurt içi", r"yurt ?ici"), ("yurtdisi", "Yurt dışı", r"yurt ?disi")],
          "arama": {"yurtici": "yurt içinde görevlendirme", "yurtdisi": "yurt dışında görevlendirme"}},
         {"alan": "sure", "soru": "Görevlendirme ne kadar sürecek?",
          "secenekler": [("kisa", "7 güne kadar", r"\b[1-7] ?gun|bir hafta"),
                         ("orta", "7 günden fazla, 3 aya kadar", r"\b([89]|[1-9]\d) ?gun|\b[1-3] ?ay"),
                         ("uzun", "3 aydan uzun", r"\b([4-9]|1[0-2]) ?ay|yil|uzun")],
          "arama": {"kisa": "2547 sayılı Yükseköğretim Kanunu madde 39 yurt içinde ve yurt dışında görevlendirme "
                            "yedi güne kadar ilgili birim yöneticisi",
                    "orta": "2547 sayılı Yükseköğretim Kanunu madde 39 yurt içinde ve yurt dışında görevlendirme "
                            "üç aya kadar rektör",
                    "uzun": "2547 sayılı Yükseköğretim Kanunu madde 39 yurt içinde ve yurt dışında görevlendirme "
                            "üç aydan fazla üniversite yönetim kurulu"}},
         {"alan": "yolluk", "soru": "Yolluk ve gündelik (yevmiye) ödenmesini istiyor musun?",
          "secenekler": [("yolluklu", "Evet, yolluklu-yevmiyeli", r"evet|yolluklu|yevmiyeli|odensin"),
                         ("yolluksuz", "Hayır, yolluksuz (masrafı kurum ödemiyor)", r"hayir|yolluksuz|masrafsiz")],
          "arama": {"yolluklu": "yolluk ve gündelik ödenmesi Harcırah Kanunu", "yolluksuz": "kurumlarından yolluk almaksızın"}},
     ]},
]


REWRITE_MODEL = "qwen3:8b"  # 8B yardımcı model (talimat izlemede ince ayarlı/ham Trendyol'dan iyi çıktı, 22.10)
REWRITE_SCHEMA = {"type": "object", "properties": {"soru": {"type": "string"}}, "required": ["soru"]}
REWRITE_EXAMPLES = (
    "Örnekler:\n"
    "Önceki: \"Kayıt dondurma şartları neler?\" Yeni: \"Peki süresi ne kadar?\" -> \"Kayıt dondurma süresi ne kadar?\"\n"
    "Önceki: \"Mühendislik Fakültesi staj süresi kaç gün?\" Yeni: \"Ya Teknoloji Fakültesinde?\" -> "
    "\"Teknoloji Fakültesi staj süresi kaç gün?\"\n"
    "Önceki: \"Erasmus başvurusu nasıl yapılır?\" Yeni: \"Yaz okulu ücreti ne kadar?\" -> \"Yaz okulu ücreti ne kadar?\" "
    "(yeni konu, aynen kalır)\n")


def rewrite_follow_up(previous: str, message: str) -> str | None:
    """Devam mesajını önceki soruya bakmadan anlaşılır tek bir tam soruya çevirir; şüpheliyse None (çağıran eski
    yönteme -birleşik arama- döner). Denetim: model zamirli sorularda iki soruyu birleştiriyor ya da önceki soruyu
    aynen döndürüyordu."""
    prompt = ("Bir üniversite asistanıyla sohbette kullanıcının yeni mesajını, önceki soruyu görmeyen biri için TEK ve "
              "TAM bir soru olarak yeniden yaz. Eksik konuyu önceki sorudan tamamla; yeni mesaj zaten tam ve farklı "
              "konudaysa aynen bırak. İki soruyu birleştirme, cevap verme.\n"
              f"{REWRITE_EXAMPLES}\nÖnceki: \"{previous}\" Yeni: \"{message}\" -> ? JSON ver.")
    try:
        reply = ollama.chat(model=REWRITE_MODEL, messages=[{"role": "user", "content": prompt}], format=REWRITE_SCHEMA,
                            think=False, options={"temperature": 0, "num_ctx": 2048, "num_predict": 120})
        text = json.loads(reply["message"]["content"])["soru"].strip()
    except Exception:
        return None
    norm = lambda value: " ".join(re.findall(r"\w+", normalize_for_matching(value)))
    out, prev, msg = norm(text), norm(previous), norm(message)
    msg_words = {w for w in msg.split() if len(w) >= 4} - {"peki", "nasil", "nedir", "neler", "kadar"}
    if (not 5 <= len(text) <= 300 or out in (prev, msg) or prev in out or msg in out
            or (msg_words and not msg_words & set(out.split()))):
        return None
    return text


def agac_secimi(metin: str, secenekler: list[tuple[str, str, str]]) -> str | None:
    normalized = normalize_for_matching(metin)
    if normalized.isdigit() and 1 <= int(normalized) <= len(secenekler):
        return secenekler[int(normalized) - 1][0]
    for anahtar, ad, desen in secenekler:
        if re.search(desen, normalized) or normalize_for_matching(ad) in normalized:
            return anahtar
    return None


class Sohbet3:
    """Tek kullanıcının açık sohbeti: geçmiş, öğrenilen birim ve satın alma kapsamı, bekleyen netleştirme.
    Rol sorulmaz; nesne silinince (sohbet kapanınca) hafıza da gider."""

    def __init__(self, asistan: Asistan3) -> None:
        self.asistan = asistan
        self.birim: str | None = None
        self.birim_netlesti = False
        self.bolum: str | None = None  # kullanıcının söylediği bölüm/program (birim içinde öncelik)
        self.bolum_soruldu = False  # bu birim için bölüm bir kez soruldu (cevap "fark etmez" olabilir)
        self.kapsam: str | None = None
        self.kapsam_netlesti = False
        self.gecmis: list[dict[str, Any]] = []
        self.bekleyen: dict[str, Any] | None = None
        self.harc: dict[str, Any] | None = None  # harç yönlendirmesi sürerken toplanan durum
        self.agac: dict[str, Any] | None = None  # genel karar ağacı durumu

    def sor(self, metin: str) -> dict[str, Any]:
        """Dönüş: {"tur": "cevap"|"netlestirme", "metin", "baglam", "secenekler": [(anahtar, ad)]}."""
        metin = metin.strip()
        if self.bekleyen is not None and self.bekleyen["tip"].startswith("harc_"):
            bekleyen, self.bekleyen = self.bekleyen, None
            cevap = self._harc_cevabi(bekleyen, metin)
            if cevap is not None:
                return cevap
            self.harc = None  # seçenek değil, yeni soru
        if self.bekleyen is not None and self.bekleyen["tip"] == "agac":
            bekleyen, self.bekleyen = self.bekleyen, None
            adim = bekleyen["adim"]
            secim = agac_secimi(metin, adim["secenekler"])
            if secim is not None or SKIP_WORDS_RE.match(normalize_for_matching(metin)):
                self.agac["cevaplar"][adim["alan"]] = secim or ""
                return self._agac_devam()
            self.agac = None  # seçenek değil, yeni soru
        if self.bekleyen is not None and self.bekleyen["tip"] == "bolum":
            bekleyen, self.bekleyen = self.bekleyen, None
            secim = self._bolum_secimi(metin, bekleyen["secenekler"])
            if secim is not None:
                self.bolum_soruldu = True
                self.bolum = BOLUM_GENEL if secim == "hepsi" else secim
                return self._isle(bekleyen["soru"], bekleyen["arama"])
            # Seçenek değil, yeni soru: bekleyen bırakılır.
        if self.bekleyen is not None:
            bekleyen, self.bekleyen = self.bekleyen, None
            secim = self._secimi_coz(metin, bekleyen["secenekler"])
            skip = SKIP_WORDS_RE.match(normalize_for_matching(metin))
            if bekleyen["tip"] == "birim_yaz" and secim is None and not skip:
                secim = faculty_from_text(metin)
                if secim is None and len(metin.split()) <= 6 and not bekleyen.get("tekrar"):
                    # Tanınmayan ad sessizce "genel kural"a düşmesin: bir kez daha sorulur
                    self._sor_secenek("birim_yaz", bekleyen["soru"], bekleyen["arama"], [], "")
                    self.bekleyen["tekrar"] = True
                    return {"tur": "netlestirme", "baglam": [], "secenekler": [],
                            "metin": f"\"{metin}\" adını tanıyamadım. Fakültenin ya da yüksekokulun adını yazar mısın? "
                                     "(ör. Mühendislik Fakültesi, Tıp Fakültesi, Kocaeli MYO)"}
            # "Hangi fakülte?" sorusuna verilen kısa cevap her zaman cevaptır (tanınmazsa genel kural)
            if secim is not None or skip or (bekleyen["tip"] == "birim_yaz" and len(metin.split()) <= 6):
                if bekleyen["tip"] == "birim" and secim == "baska":
                    return self._sor_secenek("birim_yaz", bekleyen["soru"], bekleyen["arama"], [],
                                             "Hangi fakülte / yüksekokul ya da bölümdesin? (ör. Fen Edebiyat, İnşaat "
                                             "Mühendisliği, Kocaeli MYO)")
                if bekleyen["tip"] in ("birim", "birim_yaz"):
                    self.birim_netlesti = True
                    self.birim = "genel" if (skip or secim in ("genel", None)) else secim
                    self.bolum = next(iter(departments_in(metin)), None) if self.birim != "genel" else None
                    self.bolum_soruldu = False
                else:
                    self.kapsam, self.kapsam_netlesti = (None if skip else secim), True
                return self._isle(bekleyen["soru"], bekleyen["arama"])
            # Seçenek değil, yeni soru: bekleyen bırakılır.
        self.sorulan = 0  # yeni soru: bu soru için sorulan netleştirme sayısı
        mentioned = units_in(metin)
        if mentioned:
            if mentioned[0] != self.birim:
                self.bolum_soruldu = False
            self.birim, self.birim_netlesti = mentioned[0], True
            self.bolum = next(iter(departments_in(metin)), None)
        normalized = normalize_for_matching(metin)
        if HARC_RE.search(normalized) and HARC_ASK_RE.search(normalized) and not HARC_SKIP_RE.search(normalized):
            self.harc = {"soru": metin, "tur": harc_secimi(metin, HARC_TUR_RE), "sure": harc_secimi(metin, HARC_SURE_RE),
                         "kabul": harc_secimi(metin, HARC_KABUL_RE), "yil": None}
            return self._harc_devam()
        # Ağaç kısa/genel sorular içindir; ayrıntılı soru ("…gemide staj yapan öğrenci defterini…") doğrudan aranır
        if (not AGAC_ATLA_RE.search(normalized) and AGAC_KISISEL_RE.search(normalized)
                and len(metin.split()) <= AGAC_MAX_WORDS):
            for agac in AGACLAR:
                if re.search(agac["tetik"], normalized):
                    # Soruda zaten yazan durumlar baştan doldurulur ("yurt dışı kongre 5 gün" -> iki adım atlanır)
                    cevaplar = {a["alan"]: c for a in agac["adimlar"]
                                if (c := next((k for k, _, d in a["secenekler"] if re.search(d, normalized)), None))}
                    self.agac = {"tanim": agac, "soru": metin, "cevaplar": cevaplar}
                    return self._agac_devam()
        arama, scored = self._arama_metni(metin)
        return self._isle(metin, arama, scored)

    def _harc_cevabi(self, bekleyen: dict[str, Any], metin: str) -> dict[str, Any] | None:
        """Harç sorusuna verilen cevabı işler; seçenek değilse None (yeni soru)."""
        if self.harc is None:
            return None
        skip = SKIP_WORDS_RE.match(normalize_for_matching(metin))
        if bekleyen["tip"] == "harc_tur":
            secim = self._secimi_coz(metin, bekleyen["secenekler"]) or harc_secimi(metin, HARC_TUR_RE)
            if secim is None and not skip:
                return None
            self.harc["tur"] = secim or "birinci"
        elif bekleyen["tip"] in ("harc_kabul", "harc_yil"):
            alan, desenler = ("kabul", HARC_KABUL_RE) if bekleyen["tip"] == "harc_kabul" else ("yil", HARC_YIL_RE)
            secim = self._secimi_coz(metin, bekleyen["secenekler"]) or harc_secimi(metin, desenler)
            if secim is None and not skip:
                return None
            self.harc[alan] = secim or bekleyen["secenekler"][-1][0]
        elif bekleyen["tip"] == "harc_sure":
            secim = harc_secimi(metin, HARC_SURE_RE) or self._secimi_coz(metin, bekleyen["secenekler"])
            if secim is None and not skip:
                return None
            self.harc["sure"] = secim or "normal"
        else:  # harc_birim: seçenek (tablodaki fakülte adı) ya da serbest metin ("İşletme", "Kocaeli MYO")
            secim = self._secimi_coz(metin, bekleyen["secenekler"]) if bekleyen["secenekler"] else None
            if re.search(r"\bmyo\b|meslek yuksekokul", normalize_for_matching(metin)):
                secim = "Meslek Yüksekokulları"  # tabloda tüm MYO'lar tek satır
            found = units_in(metin)
            if secim in dict(bekleyen["secenekler"]):
                self.harc["fakulte"] = secim
            elif found:
                self.birim, self.birim_netlesti = found[0], True
                self.harc["fakulte"] = BIRIMLER.get(found[0], found[0])
            elif not skip and len(metin.split()) > 4:
                return None
            self.harc["birim_soruldu"] = True
        return self._harc_devam()

    def _agac_devam(self) -> dict[str, Any]:
        """Sıradaki cevaplanmamış adımı sorar; hepsi tamamsa durumla zenginleştirilmiş soruyu cevaplar."""
        agac = self.agac
        for adim in agac["tanim"]["adimlar"]:
            kosul = adim.get("kosul", {})
            if adim["alan"] in agac["cevaplar"] or any(agac["cevaplar"].get(k) not in v for k, v in kosul.items()):
                continue
            secenekler = [(k, ad) for k, ad, _ in adim["secenekler"]]
            self.bekleyen = {"tip": "agac", "soru": agac["soru"], "arama": agac["soru"], "secenekler": secenekler,
                             "adim": adim}
            self.sorulan = getattr(self, "sorulan", 0) + 1
            adlar = "\n".join(f"{i}) {ad}" for i, (_, ad) in enumerate(secenekler, start=1))
            return {"tur": "netlestirme", "baglam": [], "secenekler": secenekler,
                    "metin": f"{adim['soru']}\n{adlar}\n(Numarasını ya da adını yazabilirsin.)"}
        durum, ekler = [], []
        for adim in agac["tanim"]["adimlar"]:
            secim = agac["cevaplar"].get(adim["alan"])
            if secim:
                durum.append(next(ad for k, ad, _ in adim["secenekler"] if k == secim))
                ekler.append(adim["arama"].get(secim, ""))
        self.agac = None
        soru = f"{agac['soru']} (Durumum: {'; '.join(durum)}.)" if durum else agac["soru"]
        return self._isle(soru, " ".join([soru] + ekler))

    def _harc_yaprak(self, soru: str, kural: str, row: str) -> dict[str, Any]:
        """Karar ağacının sonucu: kural + tablodaki tutar (model kullanılmaz)."""
        parts = [kural]
        if row:
            parts.append(f"📌 2026-2027 KOÜ ücret tablosundaki tutar:\n{row}")
        parts.append(HARC_NOT)
        metin = "\n\n".join(parts) + f"\n\nKaynak: {FEE_TABLE_SOURCE}"
        self.gecmis.append({"soru": soru, "cevap": metin, "kaynaklar": []})
        return {"tur": "cevap", "metin": metin, "baglam": [], "secenekler": []}

    def _harc_devam(self) -> dict[str, Any]:
        """Eksik bilgiyi sorar; hepsi tamamsa durumla zenginleştirilmiş soruyu cevaplar."""
        harc = self.harc
        if harc["tur"] is None:
            return self._sor_secenek("harc_tur", harc["soru"], harc["soru"], HARC_TUR,
                                     "Harç durumu öğretim türüne göre değişiyor. Hangi öğretim türünde okuyorsun?")
        if harc["tur"] == "birinci" and harc["sure"] is None:
            return self._sor_secenek("harc_sure", harc["soru"], harc["soru"], HARC_SURE,
                                     "Programının normal öğrenim süresi içinde misin? (Ör. 4 yıllık bölümde 1-4. "
                                     "yıllar normal süredir; 5. yıl ve sonrası süreyi aşmak sayılır.)")
        if harc["tur"] == "yabanci" and harc["kabul"] is None:
            return self._sor_secenek("harc_kabul", harc["soru"], harc["soru"], HARC_KABUL,
                                     "Yabancı uyruklu öğrencilerde iki ayrı ücret tablosu var. Üniversiteye nasıl "
                                     "kabul edildin?")
        if harc["tur"] == "yabanci" and harc["kabul"] == "uluslararasi" and harc["yil"] is None:
            return self._sor_secenek("harc_yil", harc["soru"], harc["soru"], HARC_YIL,
                                     "Uluslararası öğrencilerde ücret kayıt yılına göre değişiyor. Hangi yıl kayıt oldun?")
        needs_unit = harc["tur"] in ("ikinci", "uzaktan", "yabanci") or harc["sure"] == "asti"
        if needs_unit and self.birim and self.birim != "genel" and not harc.get("fakulte"):
            harc["fakulte"] = BIRIMLER.get(self.birim, self.birim)
        if needs_unit and not harc.get("fakulte") and not harc.get("birim_soruldu"):
            return self._sor_secenek("harc_birim", harc["soru"], harc["soru"], fee_units(),
                                     "Tutar fakülteye göre değişiyor. Hangi fakülte / yüksekokulda okuyorsun?")
        tur = dict(HARC_TUR)[harc["tur"]].split(" (")[0]
        parts = [f"{tur} öğrencisiyim"]
        if harc["tur"] == "birinci":
            parts.append("normal öğrenim süresi içindeyim" if harc["sure"] == "normal" else "normal öğrenim süresini aştım")
        if harc["tur"] == "yabanci":
            parts.append(dict(HARC_KABUL)[harc["kabul"]])
            if harc["yil"]:
                parts.append(dict(HARC_YIL)[harc["yil"]].split(" (")[0])
        unit_name = harc.get("fakulte") or (BIRIMLER.get(self.birim, self.birim)
                                            if self.birim and self.birim != "genel" else "")
        if unit_name:
            parts.append(unit_name)
        soru = f"{harc['soru']} (Durumum: {', '.join(parts)}.)"
        dal = harc["sure"] if harc["tur"] == "birinci" else (harc["kabul"] if harc["tur"] == "yabanci" else harc["tur"])
        arama = " ".join(x for x in (soru, unit_name, HARC_ARAMA.get(dal, ""), HARC_YIL_ARAMA.get(harc["yil"] or "", ""),
                                     "2026-2027") if x)
        self.harc = None
        if dal == "normal":  # tutar yok, kural kesin
            return self._harc_yaprak(soru, HARC_KURAL["normal"], "")
        reply = None
        if dal not in HARC_KURAL or (dal != "lisansustu" and not unit_name):
            reply = self._isle(soru, arama)  # yaprak belirlenemedi (fakülte bilinmiyor): arama + model
        column = HARC_SUTUN.get(harc["yil"] or "") if harc["tur"] == "yabanci" and harc["kabul"] == "uluslararasi" \
            else HARC_SUTUN.get(dal)
        if harc["tur"] == "lisansustu":
            row = "\n".join(r for r in (fee_row("Lisansüstü", "Normal Öğrenim"), fee_row("Lisansüstü", "İkinci Öğretim")) if r)
        else:
            row = fee_row(unit_name, column) if column and unit_name else ""
        if row and reply is None:
            return self._harc_yaprak(soru, HARC_KURAL[dal], row)
        if reply is None:
            reply = self._isle(soru, arama)
        if row and reply["tur"] == "cevap":
            body = reply["metin"].partition("\n\nKaynak:")[0]
            # Model aynı cümleyi tekrarlayabiliyor (uluslararası öğrenci dalında görüldü): tekrarlar atılır
            sentences = list(dict.fromkeys(x.strip() for x in re.split(r"(?<=[.!?])\s+", body.strip()) if x.strip()))
            note = f"📌 2026-2027 KOÜ ücret tablosundaki tutar:\n{row}"
            reply["metin"] = (f"{' '.join(sentences)}\n\n{note}\n\nKaynak: {FEE_TABLE_SOURCE}")
        return reply

    def _isle(self, soru: str, arama: str, scored: list[tuple[float, dict[str, Any]]] | None = None) -> dict[str, Any]:
        retriever = self.asistan.retriever
        if scored is None:
            scored = retriever.candidates(arama, self.birim if self.birim_netlesti else None)
        if max((s for s, _ in scored), default=-99.0) < RELEVANCE_MIN:
            if self.birim == "genel" and (offer := self._genel_cikmaz(soru, arama)):
                return offer
            # Hiçbir parça soruyla ilgili değil: modele zayıf bağlam verilince uyduruyordu ("maaşlara %8,9 zam").
            personel = STAFF_RE.search(normalize_for_matching(soru)) or re.search(r"maas|\baylik|ek ders|izin", normalize_for_matching(soru))
            cevap = STAFF_REFUSAL if personel else REFUSAL_ANSWER
            self.gecmis.append({"soru": soru, "cevap": cevap, "kaynaklar": []})
            return {"tur": "cevap", "metin": cevap, "baglam": [], "secenekler": []}
        if self.bekleyen is None:
            question = self._netlestirme(scored, soru, arama)
            if question:
                return question
        onceki = None
        # Devam sorusu: arama metni önceki soruyla birleştirilmiş. (Harç yönlendirmesi gibi yalnızca sona ek
        # yapılan arama metni devam sorusu değildir.)
        devam = arama != soru and not arama.startswith(soru) and bool(self.gecmis)
        if devam:  # önceki cevabın ilk iki kaynağının belgeleri
            onceki = {retriever.articles[i]["ozellikler"]["belge_id"] for i in self.gecmis[-1]["kaynaklar"][:2]
                      if i in retriever.articles}
        ranked = retriever.rank_units(scored, arama, self.birim, self.kapsam, onceki_belgeler=onceki,
                                      bolum=self.bolum)
        context = retriever.context(ranked)
        if not context and self.birim == "genel" and (offer := self._genel_cikmaz(soru, arama)):
            return offer
        if not context:  # filtrelerden sonra madde kalmadı: model kaynaksız cevap uydurmasın
            ad = BIRIMLER.get(self.birim, "") if self.birim and self.birim != "genel" else ""
            cevap = (f"{ad} için bu konuda elimdeki belgelerde bilgi bulamadım. " if ad else "") + REFUSAL_ANSWER
            self.gecmis.append({"soru": soru, "cevap": cevap, "kaynaklar": []})
            return {"tur": "cevap", "metin": cevap, "baglam": [], "secenekler": []}
        if devam and getattr(self, "tam_soru", None):
            model_question = self.tam_soru
        else:
            model_question = f"{soru} (Önceki soru: {self.gecmis[-1]['soru']})" if devam else soru
        cevap = self.asistan.generate(model_question, context)
        cevap = self._bolum_notu(cevap, context)
        self.gecmis.append({"soru": soru, "cevap": cevap, "kaynaklar": [a["id"] for a in context]})
        return {"tur": "cevap", "metin": cevap, "baglam": context, "secenekler": []}

    def _bolum_notu(self, cevap: str, context: list[dict[str, Any]]) -> str:
        """Bölüm söylenmemişken cevap tek bir bölümün belgesine dayanıyorsa bunu açıkça yazar
        (23.10: "Mühendislik Fakültesi" seçilince İnşaat duyurusundaki tarih fakülte kuralı gibi verildi)."""
        body, sep, kaynak = cevap.partition("\n\nKaynak:")
        if (self.bolum and self.bolum != BOLUM_GENEL) or not sep:
            return cevap
        for article in context:
            title = article["ozellikler"]["belge_adi"]
            if title in kaynak and unit_of(article["ozellikler"]) not in GENERAL_UNITS and title_departments_of(title):
                # Not kaynak satırından ÖNCE: web arayüzü "Kaynak:" satırından sonrasını kaynak sayıyor
                # 24.10: FEF'te staj belgesi yalnız Yapay Zeka bölümünde var; kullanıcı bunu genel kural sandı.
                # "Fakülte geneli için belge yok" yalnız bağlamda birimin bölümsüz (fakülte geneli) belgesi yoksa.
                faculty_wide = any(unit_of(a["ozellikler"]) == self.birim
                                   and not title_departments_of(a["ozellikler"]["belge_adi"]) for a in context)
                scope = ("Fakültenin genel kuralı farklı olabilir" if faculty_wide else
                         f"{BIRIMLER.get(self.birim, 'Birimin')} geneli ya da diğer bölümler için elimde bu konuda "
                         "belge bulamadım")
                # AKTS paketinde bölüm/program adı son parçada ("AKTS Bilgi Paketi – Devlet Konservatuvarı – Trombon")
                part = title.split(" – ")[-1] if title.startswith("AKTS") else title.split(" – ")[0]
                return (f"{body}\n\nℹ️ Bu bilgi yalnızca şu bölüme ait bir belgeden: {part}. "
                        f"{scope}; başka bir bölümdeysen bölümünün adını yazarsan ona göre bakarım.{sep}{kaynak}")
        return cevap

    def _netlestirme(self, scored: list[tuple[float, dict[str, Any]]], soru: str, arama: str) -> dict[str, Any] | None:
        retriever = self.asistan.retriever
        if not self.birim_netlesti and not CENTRAL_TOPIC_RE.search(normalize_for_matching(arama)):
            # Belge başına bir madde: tek bir fakülte belgesinin parçaları listeyi doldurunca ("Stajın amacı ne?"
            # -> Tıp kalp sağlığı stajı x6) farklı birimler görünmüyor, soru sorulmuyordu.
            top, seen_docs = [], set()
            for _, a in retriever.rank_units(scored, arama, None, None, priors=False):
                fields = a["ozellikler"]
                # Duyuru ve rapor/form belgeleri birime özgü KURAL değildir (fakültelerin kalite komisyonu
                # duyuruları genel yönerge sorusunda "hangi fakülte?" sordurtuyordu).
                if fields["belge_id"].startswith("webd_") or (
                        fields["belge_id"].startswith("web")
                        and REPORT_RE.search(normalize_for_matching(fields["belge_adi"] + " " + fields["belge_id"]))):
                    continue
                if a["ozellikler"]["belge_id"] not in seen_docs:
                    seen_docs.add(a["ozellikler"]["belge_id"])
                    top.append(a)
                if len(top) == AMBIGUITY_PROBE:
                    break
            birimler = [unit_of(a["ozellikler"]) for a in top]
            ozel = [b for b in birimler if b not in GENERAL_UNITS]
            sirali = list(dict.fromkeys(ozel))
            # En iyi sonuç bir ders kaydıysa soru bir derse ait: fakülte sormak anlamsız ("Yapay Sinir Ağları
            # dersinin içeriği nedir?" -> "hangi fakülte?" diye soruyordu). Program bilgi paketi (akts_program)
            # ders değildir: "Mezuniyet için kaç AKTS?" programa/fakülteye göre değişir (24.10).
            course = bool(top) and top[0]["ozellikler"]["belge_id"].startswith(("aktsders_", "dersler_"))
            if not course and len(sirali) >= 2 and not (birimler[0] in GENERAL_UNITS and len(ozel) < AMBIGUITY_PROBE // 2):
                secenekler = [(b, BIRIMLER.get(b, b)) for b in sirali] + [
                    ("baska", "Başka bir fakülte / yüksekokul (adını yazacağım)"), ("genel", BIRIMLER["genel"])]
                return self._sor_secenek("birim", soru, arama, secenekler,
                                         "Bu konuda birimlere göre farklı kurallar var. Hangi fakülte / yüksekokul "
                                         "için soruyorsun?")
        if (self.birim_netlesti and self.birim and self.birim != "genel" and not self.bolum
                and not self.bolum_soruldu and getattr(self, "sorulan", 0) < 2):
            # En fazla 2 soru: karar ağacından sonra (mazeret: tür -> neden -> fakülte) bölüm de sorulmaz.
            # Fakülte belli, bölüm değil ve fakültede birden çok bölümün bu konuda kendi belgesi var: bölüm sorulur
            # (24.10: Mühendislik seçilince staj defteri cevabı yalnız İnşaat duyurusundan geliyordu).
            secenekler = self._bolum_secenekleri(arama)
            if len(secenekler) >= 2:
                self.bolum_soruldu = True
                return self._sor_secenek(
                    "bolum", soru, arama, secenekler + [("hepsi", "Bölüm fark etmez (fakülte geneli)")],
                    f"{BIRIMLER.get(self.birim, 'Bu birim')} içinde bölümlere göre farklı belgeler var. "
                    "Hangi bölüm için soruyorsun?")
        if not self.kapsam_netlesti and not SCOPE_HINT_RE.search(normalize_for_matching(arama)):
            top = [a for _, a in retriever.rank_units(scored, arama, self.birim, None)[:CONTEXT_SIZE]]
            docs = {a["ozellikler"]["belge_adi"] for a in top}
            if PROCUREMENT_3F in docs and PROCUREMENT_LAW in docs:
                return self._sor_secenek(
                    "kapsam", soru, arama,
                    [("3f", "Evet, araştırma projesi (BAP, TÜBİTAK vb.) kapsamında"), ("4734", "Hayır, genel alım")],
                    "Bu alım bir araştırma projesi kapsamında mı? Proje alımlarında 4734/3(f) Usul ve Esasları, "
                    "diğer alımlarda 4734 sayılı Kamu İhale Kanunu uygulanır.")
        return None

    def _genel_cikmaz(self, soru: str, arama: str) -> dict[str, Any] | None:
        """"Genel kural" seçildi ama birimden bağımsız kural yok (bitirme projesi, proje dersi notu: her fakülte kendi
        yönergesinde): "bilgi yok" çıkmazı yerine konuda belgesi olan fakülteler sorulur (24.10 taraması)."""
        secenekler = self._birim_secenekleri(arama)
        if not secenekler:
            return None
        self.birim, self.birim_netlesti, self.bolum = None, False, None
        return self._sor_secenek(
            "birim", soru, arama, secenekler + [("baska", "Başka bir fakülte / yüksekokul (adını yazacağım)")],
            "Bu konuda birimden bağımsız genel bir kural bulamadım; fakülteler kendi yönergelerinde düzenliyor. "
            "Hangi fakülte / yüksekokul için bakayım?")

    def _birim_secenekleri(self, arama: str) -> list[tuple[str, str]]:
        """Filtresiz aramada bu konuda belgesi olan birimler (ilgi sırasıyla, en fazla BOLUM_SECENEK)."""
        retriever = self.asistan.retriever
        found: list[str] = []
        for _, article in retriever.rank_units(retriever.candidates(arama), arama, None, None, priors=False)[:30]:
            unit = unit_of(article["ozellikler"])
            if unit not in GENERAL_UNITS and unit in BIRIMLER and unit not in found:
                found.append(unit)
        return [(unit, BIRIMLER[unit]) for unit in found[:BOLUM_SECENEK]]

    def _bolum_secenekleri(self, arama: str) -> list[tuple[str, str]]:
        """Birim içindeki en ilgili belgelerin bölümleri (belge adından), ilgi sırasıyla, en fazla BOLUM_SECENEK."""
        retriever = self.asistan.retriever
        ranked = retriever.rank_units(retriever.candidates(arama, self.birim), arama, self.birim, None)
        found: list[str] = []
        shown: dict[str, str] = {}
        program_question = bool(PROGRAM_ASK_RE.search(normalize_for_matching(arama)))
        for _, article in ranked[:BOLUM_TARAMA]:
            fields = article["ozellikler"]
            if unit_of(fields) != self.birim:
                continue
            # Program bilgi paketleri yalnız programa bağlı soruda bölüm sordurur ("final şartı?" -> Konservatuvar
            # "Trombon mu?" diye soruyordu, 24.10)
            if fields["belge_id"].startswith("akts_") and not program_question:
                continue
            for name in title_departments_of(fields["belge_adi"]):
                if name not in found:
                    found.append(name)
                    # Ad listede yoksa belge başlığındaki yazım ("Metalurji Mühendisliği – …"; "Muhendisligi" değil)
                    head = fields["belge_adi"].split(" – ")[0].replace(" duyurusu", "")
                    shown[name] = head if departments_in(head) == [name] else name.title()
        return [(name, DEPARTMENT_NAMES.get(name, shown[name])) for name in found[:BOLUM_SECENEK]]

    def _bolum_secimi(self, metin: str, secenekler: list[tuple[str, str]]) -> str | None:
        """Bölüm sorusunun cevabı: numara, seçenek adı, yazılan bölüm adı; "fark etmez" ya da fakülte adı -> "hepsi"."""
        normalized = normalize_for_matching(metin)
        if normalized.isdigit() and 1 <= int(normalized) <= len(secenekler):
            return secenekler[int(normalized) - 1][0]
        for anahtar, ad in secenekler:
            ad_norm = normalize_for_matching(ad)
            if ad_norm in normalized or (len(normalized) >= 4 and normalized in ad_norm):
                return anahtar
        found = departments_in(metin)
        if found:
            return found[0]
        if SKIP_WORDS_RE.match(normalized) or self.birim in units_in(metin):
            return "hepsi"
        return None

    def _sor_secenek(self, tip: str, soru: str, arama: str, secenekler: list[tuple[str, str]],
                     giris: str) -> dict[str, Any]:
        self.bekleyen = {"tip": tip, "soru": soru, "arama": arama, "secenekler": secenekler}
        self.sorulan = getattr(self, "sorulan", 0) + 1
        adlar = "\n".join(f"{i}) {ad}" for i, (_, ad) in enumerate(secenekler, start=1))
        return {"tur": "netlestirme", "baglam": [], "secenekler": secenekler,
                "metin": f"{giris}\n{adlar}\n(Numarasını ya da adını yazabilirsin.)"}

    def _arama_metni(self, metin: str) -> tuple[str, list[tuple[float, dict[str, Any]]]]:
        """Devam sorusu (bağlantı sözcüğü ya da 1-2 kelime) önceki soruyla birleştirilerek aranır.
        Not: puana dayalı karar (tek başına ve birleşik aramayı karşılaştırma) v3 testinde devam sorularında
        aramayı 13/16'dan 10/16'ya düşürdü; tam bir yeni soruyu da önceki konuya bağlıyordu. Kural geri alındı."""
        retriever = self.asistan.retriever
        self.tam_soru = None
        if self.gecmis and (len(metin.split()) <= 3 or FOLLOW_UP_V3_RE.search(normalize_for_matching(metin))):
            # 8B model devam sorusunu bağımsız tam soruya çevirir ("Peki süresi?" -> "Kayıt dondurmanın süresi ne
            # kadar?"); arama hem tam soruyla hem birleşik metinle yapılır, model tam soruyu görür (22.10).
            self.tam_soru = rewrite_follow_up(self.gecmis[-1]["soru"], metin)
            merged = f"{self.gecmis[-1]['soru']} {metin}"
            if self.tam_soru:
                merged = f"{self.tam_soru} {merged}"
            return merged, retriever.candidates(merged, self.birim if self.birim_netlesti else None)
        return metin, retriever.candidates(metin, self.birim if self.birim_netlesti else None)

    @staticmethod
    def _secimi_coz(metin: str, secenekler: list[tuple[str, str]]) -> str | None:
        """Kullanıcının seçimi: numara, tam ad, evet/hayır (3f), birim adı (bölüm dahil), son çare ilk sözcük.
        İlk sözcük en sona alındı: "Hereke Asım ..." ve "Hereke Ömer ..." ilk sözcükte karışıyordu (23.10)."""
        normalized = normalize_for_matching(metin)
        if normalized.isdigit() and 1 <= int(normalized) <= len(secenekler):
            return secenekler[int(normalized) - 1][0]
        keys = [k for k, _ in secenekler]
        if "3f" in keys:  # evet/hayır cevabı
            if re.match(r"^(evet|proje|bap|tubitak|arastirma)", normalized):
                return "3f"
            if re.match(r"^(hayir|genel|degil)", normalized):
                return "4734"
        for anahtar, ad in secenekler:
            if normalize_for_matching(ad) in normalized or normalized in normalize_for_matching(ad):
                return anahtar
        mentioned = units_in(metin)
        if mentioned:
            return mentioned[0]
        for anahtar, ad in secenekler:
            ilk = normalize_for_matching(ad).split()[0]
            if ilk in normalized.split():
                return anahtar
        return None


def describe(article: dict[str, Any]) -> str:
    fields = article["ozellikler"]
    return f"{fields['belge_adi']} | {fields['madde_etiketi']} {fields['madde_basligi']}"


def main() -> int:
    args = sys.argv[1:]
    model = CHAT_MODEL
    if "--model" in args:
        position = args.index("--model")
        model = args[position + 1]
        del args[position:position + 2]
    if "--arama" in args:
        args.remove("--arama")
        question = " ".join(args)
        retriever = Retriever3()
        for score, article in retriever.rank_units(retriever.candidates(question), question, None, None)[:8]:
            print(f" {score:6.2f}  {describe(article)}")
        return 0
    sohbet = Sohbet3(Asistan3(model))
    print(f"KOÜ asistanı v3 ({model}) — çıkmak için boş satır.\n")
    while True:
        try:
            question = input("Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        reply = sohbet.sor(question)
        print(f"\nAsistan: {reply['metin']}")
        if reply["baglam"]:
            print(f"[Bağlam: {' || '.join(describe(a) for a in reply['baglam'])}]")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
