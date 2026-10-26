#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Çok belgeli RAG: kapsam yönlendirmeli arama (birim + hedef_kitle) + reranker + cevap.

Kapsam yönlendirmesi: Ön Lisans ve Lisans Yönetmeliği Tıp, Diş Hekimliği ve Konservatuvar'a
uygulanmaz; lisansüstü kendi yönetmeliklerine tabidir. Soruda bu birimler geçmiyorsa yalnızca
ön lisans/lisans ve tüm öğrencilere uygulanan belgeler aranır. Aksi halde bir lisans öğrencisine
Konservatuvar'ın kuralı söylenebiliyordu (ör. "kaydımı dondurabilir miyim").

Kullanım:
    python -X utf8 rag_v2.py "kopya çekersem ne olur"
    python -X utf8 rag_v2.py --sohbet --kitle ogrenci   # sohbet: önceki soruyu hatırlar, belirsizse birimi sorar
    python -X utf8 rag_v2.py --arama "tıp fakültesinde devamsızlık"   # sadece getirilen maddeler
    python -X utf8 rag_v2.py --kitle ogrenci --birim tip "devamsızlık sınırı"   # birim: genel/tip/dis/konservatuvar/enstitu
    python -X utf8 rag_v2.py --kitle personel "devamsızlık sınırı"   # personel: tüm birimlerde arar
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

import chromadb
import ollama

from belge_isle import ARTICLES_JSONL, CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL
from pdf_to_chromadb import normalize_for_matching
from kaynak_dogrula import verify
from raft_format import REFUSAL_ANSWER, SYSTEM_PROMPT_V2, build_messages

CHAT_MODEL = "kou-asistan-v2"  # Trendyol-LLM-8B-T1 + KOÜ RAFT eğitimi
CONTEXT_SIZE = 4
CHUNK_CANDIDATES = 40
# Embedding araması "yatay geçiş" geçen onlarca benzer parçayı ayırt edemiyor (takvim sorusunda takvim
# bölümü 9. sıradaydı). Çapraz kodlayıcı (cross-encoder) her adayı soruyla birlikte okuyup yeniden sıralar.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Soruda geçerse ilgili birimin belgeleri aranır (normalize: küçük harf, aksansız).
UNIT_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\btip (fakultesi|ogrencisi|egitimi)|\btipta\b|\bintorn|\bdonem [1-6]\b|\bkomite"), "tip"),
    (re.compile(r"\bdis hekimligi|\bdis fakultesi|\bdiscilik"), "dis"),
    (re.compile(r"\bkonservatuvar|\bkonservatuar"), "konservatuvar"),
    (re.compile(r"\byuksek lisans|\bdoktora|\blisansustu|\btez\b|\btezi\b|\benstitu|\bsanatta yeterlik|\btik\b"
                r"|\bbilimsel hazirlik|\bjuri"), "enstitu"),
    # Fakülte/MYO'ya özel belgeler (staj, İME, bitirme projesi). Bölüm adı tek başına yetmez
    # ("mekatronik" hem Mühendislik hem Teknoloji'de var); belirsizse asistan hangi birim olduğunu sorar.
    (re.compile(r"\bmuhendislik fakultes|\bmuhfak\b"), "muhendislik"),
    (re.compile(r"\bteknoloji fakultes"), "teknoloji"),
    (re.compile(r"\bturizm fakultes|\bgastronomi"), "turizm"),
    (re.compile(r"\bziraat"), "ziraat"),
    (re.compile(r"\bmimarlik|\bic mimar|\btasarim fakultes"), "mimarlik"),
    (re.compile(r"\bdenizcilik"), "denizcilik"),
    (re.compile(r"\biletisim fakultes|\bradyo,? televizyon|\brts\b"), "iletisim"),
    (re.compile(r"\bfen[ -]edebiyat|\byapay zeka (ve makine ogrenmesi )?bolum"), "fen_edebiyat"),
    (re.compile(r"\bkocaeli (myo|meslek yuksekokul)|\bkmyo\b"), "kocaeli_myo"),
    (re.compile(r"\bkartepe"), "kartepe_myo"),
    (re.compile(r"\bford otosan|\bihsaniye"), "ford_otosan_myo"),
    (re.compile(r"\bdegirmendere|\bali ozbay|\bel sanatlari"), "degirmendere_myo"),
]
OWN_REGULATION = {"tip", "dis", "konservatuvar"}  # Ön Lisans ve Lisans Yönetmeliği uygulanmayan birimler
# Akademik personel soruları (ek ders, BAP, harcırah...). Personel belgeleri öğrenci sorularına karışmasın diye
# yalnızca bu ifadeler geçince (ya da arayüzde "akademisyenim" seçilince) aranır.
# ("proje" tek başına yok: öğrencinin bitirme projesi olabilir.)
STAFF_RE = re.compile(
    r"\bek ders|\bders yuku|\bbap\b|\barastirma projesi|\bproje (butce|yurutucu|suresi|basvurusu|ara rapor)"
    r"|\bharcirah|\byolluk|\bgundelik|\bgorevlendir|\bogretim (uyesi|elemani)|\barastirma gorevlisi"
    r"|\bdocent|\bprofesor|\bkadro|\batama\b|\bihale|\bdogrudan temin|\bsatin alma|\bfatura|\bdoner sermaye"
    r"|\bkdv\b|\btevkifat"
    r"|\bkalite komisyonu|\begitim komisyonu|\bakademik personel|\bpersonel|\bbursiyer|\bsinav ucreti"
)
STUDENT_AUDIENCE = {"hedef_kitle": {"$in": ["ogrenci", "herkes"]}}
# Yalnızca Ön Lisans ve Lisans Yönetmeliği'ne tabi öğrencilere uygulanan belgeler (Tıp/Diş/Konservatuvar'ın kendi kuralları var).
LU_ONLY = ["KOÜ Ön Lisans ve Lisans Eğitim-Öğretim Yönetmeliği", "KOÜ Başarı Notu Değerlendirme Usul ve Esasları"]
# Arayüzdeki "birimim" seçimi ve netleştirme sorusundaki seçenek adları.
# "genel": aşağıda adı olmayan fakülte ve MYO'lar (kendine özel belgesi yok, genel kurallar geçerli).
BIRIMLER = {"genel": "Genel kural (birimden bağımsız, tüm ön lisans/lisans)",
            "muhendislik": "Mühendislik Fakültesi", "teknoloji": "Teknoloji Fakültesi",
            "fen_edebiyat": "Fen Edebiyat Fakültesi", "mimarlik": "Mimarlık ve Tasarım Fakültesi",
            "denizcilik": "Denizcilik Fakültesi", "iletisim": "İletişim Fakültesi", "turizm": "Turizm Fakültesi",
            "ziraat": "Ziraat Fakültesi", "kocaeli_myo": "Kocaeli Meslek Yüksekokulu",
            "kartepe_myo": "Kartepe Turizm Meslek Yüksekokulu",
            "ford_otosan_myo": "Ford Otosan İhsaniye Otomotiv Meslek Yüksekokulu",
            "degirmendere_myo": "Değirmendere Ali Özbay Meslek Yüksekokulu",
            "tip": "Tıp Fakültesi", "dis": "Diş Hekimliği Fakültesi", "konservatuvar": "Devlet Konservatuvarı",
            "enstitu": "Enstitü (yüksek lisans, doktora)"}
def _extend_units_from_web() -> None:
    """web_belgeleri.py'nin bulduğu yeni fakülte/MYO'lar (Hukuk, Gölcük MYO ...): ad + soruda geçme kuralı."""
    path = ARTICLES_JSONL.parent.parent / "web" / "birimler_ek.json"
    if not path.exists():
        return
    for key, name in json.loads(path.read_text(encoding="utf-8")).items():
        if key in BIRIMLER:
            continue
        BIRIMLER[key] = name
        norm = normalize_for_matching(name)
        if "fakultesi" in norm:
            pattern = re.escape(norm.replace("fakultesi", "fakultes"))
        elif "meslek yuksekokulu" in norm:
            pattern = re.escape(norm.split(" meslek yuksekokulu")[0]) + r" (myo|meslek)"
        else:
            pattern = re.escape(norm[:-1])
        UNIT_RULES.append((re.compile(r"\b" + pattern), key))


_extend_units_from_web()


def _extend_units_from_departments() -> None:
    """Bölüm/program adları ve kısaltmalar -> birim (birim_haritasi.py; "İnşaat Mühendisliği" -> muhendislik)."""
    path = ARTICLES_JSONL.parent.parent / "web" / "bolum_birim.json"
    if not path.exists():
        return
    for name, key in json.loads(path.read_text(encoding="utf-8")).items():
        if key in BIRIMLER:
            pattern = re.compile(r"\b" + re.escape(name).replace(r"\ ", r"\s+") + r"\w*")
            UNIT_RULES.append((pattern, key))
            if not NOT_DEPARTMENT_RE.search(name):
                DEPARTMENT_RULES.append((pattern, name, key))


# Kısaltma / fakülte adı olan girdiler bölüm sayılmaz (bölüm önceliği yalnız gerçek bölüm/program adları için)
NOT_DEPARTMENT_RE = re.compile(r"\bfak\b|myo|^(fef|mtf|sbf|gsf|iibf)$|meslek yuksek|konservatuvari|merkezi")
DEPARTMENT_RULES: list[tuple[re.Pattern[str], str, str]] = []  # (desen, bölüm adı, birim)
_extend_units_from_departments()


_NAMES_PATH = ARTICLES_JSONL.parent.parent / "web" / "bolum_adlari.json"
# normalize bölüm adı -> Türkçe görünen ad (bölüm seçme düğmeleri; birim_haritasi.py üretir)
DEPARTMENT_NAMES: dict[str, str] = json.loads(_NAMES_PATH.read_text(encoding="utf-8")) if _NAMES_PATH.exists() else {}


def departments_in(text: str) -> list[str]:
    """Metinde geçen bölüm/program adları (normalize, uzun eşleşme kısayı yutar; "makina" = "makine")."""
    normalized = normalize_for_matching(text).replace("makina", "makine")
    found = [(m.start(), m.end(), name.replace("makina", "makine"))
             for pattern, name, _ in DEPARTMENT_RULES for m in pattern.finditer(normalized)]
    kept = [f for f in found if not any(o[0] <= f[0] and f[1] <= o[1] and (o[1] - o[0]) > (f[1] - f[0])
                                        for o in found)]
    return list(dict.fromkeys(name for _, _, name in sorted(kept)))
STAFF_REFUSAL = (
    "Bu sorunun cevabı elimdeki mevzuat maddelerinde yer almıyor. Doğru bilgi için ilgili birime "
    "(Personel Daire Başkanlığı, BAP Koordinatörlüğü, Strateji Geliştirme veya Satın Alma birimi) danışmanızı öneririm."
)


def units_in(text: str) -> list[str]:
    """Metinde geçen birimler (geçiş sırasıyla). Uzun eşleşme kısayı yutar: "elektrik mühendisliği" -> Mühendislik,
    içindeki "elektrik" (MYO programı) sayılmaz."""
    normalized = normalize_for_matching(text)
    found = [(m.start(), m.end(), unit) for pattern, unit in UNIT_RULES for m in pattern.finditer(normalized)]
    kept = [f for f in found if not any(o[0] <= f[0] and f[1] <= o[1] and (o[1] - o[0]) > (f[1] - f[0])
                                        for o in found)]
    return list(dict.fromkeys(unit for _, _, unit in sorted(kept)))


def is_staff(question: str, kitle: str | None = None) -> bool:
    if kitle is not None:
        return kitle == "personel"
    return bool(STAFF_RE.search(normalize_for_matching(question)))


def scope_filter(question: str, kitle: str | None = None, birim: str | None = None) -> dict[str, Any] | None:
    """Soruya göre ChromaDB 'where' filtresi (birim + hedef_kitle). None: filtre yok (tüm belgeler).

    - Soruda Tıp/Diş/Konservatuvar/lisansüstü geçerse ya da öğrenci arayüzde birimini seçtiyse: o birimin
      belgeleri + öğrenciye dönük ulusal mevzuat (YÖK, 2547 öğrenci maddeleri). Ön lisans/lisans yönetmeliği
      bu birimlere uygulanmaz. Personel sorusunda personel belgeleri de eklenir.
    - Personel (hoca/idari) ve birim belirtilmemiş: TÜM birimlerin belgeleri. Hoca "devamsızlık sınırı"
      sorduğunda Lisans, Tıp ve Konservatuvar kuralları birlikte gelir; model farkı belirtir.
    - Öğrenci ve birim belirtilmemiş: genel birimlerin öğrenci belgeleri (en kalabalık öğrenci grubu).
    """
    units = units_in(question)
    staff = is_staff(question, kitle)
    if not units and birim and birim != "genel":
        units = [birim]
    if units:
        national = {"$and": [{"kurum": {"$in": ["YÖK", "TBMM"]}}, {"birim": "tum"}, STUDENT_AUDIENCE]}
        clauses: list[dict[str, Any]] = [{"birim": {"$in": units}}, national]
        if set(units) - OWN_REGULATION - {"enstitu"}:
            # Mühendislik, Teknoloji, MYO'lar...: kendi staj/İME belgeleri + Ön Lisans-Lisans dahil tüm genel belgeler.
            clauses.append({"$and": [{"birim": "tum"}, STUDENT_AUDIENCE]})
        elif set(units) & OWN_REGULATION:
            # Tıp/Diş/Konservatuvar öğrencisine diploma, yatay geçiş, hazırlık gibi genel yönergeler de uygulanır;
            # yalnızca kendi yönetmeliklerinin yerini tutan Ön Lisans ve Lisans kuralları çıkarılır.
            clauses.append({"$and": [{"birim": "tum"}, STUDENT_AUDIENCE, {"belge_adi": {"$nin": LU_ONLY}}]})
        else:
            clauses.append({"hedef_kitle": "herkes"})  # danışmanlık, uzaktan öğretim
        if staff:
            clauses.append({"hedef_kitle": "personel"})
        return {"$or": clauses}
    if staff:
        return None
    return {"$and": [{"birim": "tum"}, STUDENT_AUDIENCE]}


class Reranker:
    def __init__(self, name: str = RERANKER_MODEL) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, dtype=dtype).to(self.device).eval()

    def scores(self, question: str, passages: list[str]) -> list[float]:
        with self.torch.no_grad():
            batch = self.tokenizer([question] * len(passages), passages, padding=True, truncation=True,
                                   max_length=512, return_tensors="pt").to(self.device)
            return self.model(**batch).logits.squeeze(-1).float().tolist()


class Retriever:
    def __init__(self, use_reranker: bool = True) -> None:
        self.collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(COLLECTION_NAME)
        with ARTICLES_JSONL.open(encoding="utf-8") as input_file:
            self.articles = {row["id"]: row for row in map(json.loads, input_file)}
        self.reranker = Reranker() if use_reranker else None

    def retrieve(self, question: str, size: int = CONTEXT_SIZE, kitle: str | None = None,
                 birim: str | None = None, where: dict[str, Any] | None | str = "kapsam") -> list[dict[str, Any]]:
        """where="kapsam": scope_filter ile; başka bir değer verilirse doğrudan o filtre (None: filtresiz)."""
        vector = ollama.embed(model=EMBEDDING_MODEL, input=question)["embeddings"][0]
        result = self.collection.query(
            query_embeddings=[vector], n_results=CHUNK_CANDIDATES, include=["metadatas", "documents"],
            where=scope_filter(question, kitle, birim) if where == "kapsam" else where,
        )
        metadatas, documents = result["metadatas"][0], result["documents"][0]
        if self.reranker is not None and documents:
            scores = self.reranker.scores(question, documents)
            order = sorted(range(len(documents)), key=lambda i: -scores[i])
            metadatas = [metadatas[i] for i in order]  # madde puanı = en iyi parçasının puanı
        article_ids: list[str] = []
        for metadata in metadatas:
            # Parça -> bağlam birimi (madde ya da uzun maddenin bir parçası): "<madde_id>#<madde_parca_no>"
            parent = f"{metadata['madde_id']}#{metadata['madde_parca_no']}"
            if parent in self.articles and parent not in article_ids:
                article_ids.append(parent)
            if len(article_ids) == size:
                break
        return [self.articles[article_id] for article_id in article_ids]


class Assistant:
    def __init__(self, model: str = CHAT_MODEL) -> None:
        self.model = model
        self.retriever = Retriever()

    def answer(self, question: str, kitle: str | None = None, birim: str | None = None,
               dogrula: bool = True, arama: str | None = None) -> tuple[str, list[dict[str, Any]]]:
        """kitle: arayüzdeki "öğrenciyim / personelim" seçimi ("ogrenci" / "personel"); None: sorudan tahmin.
        birim: kullanıcının birimi (BIRIMLER anahtarı); öğrenci seçer ya da sohbette netleştirilir.
        dogrula: kaynak satırını bağlamla doğrula + yürürlük uyarısı (kaynak_dogrula.py).
        arama: aramada kullanılacak metin (sohbette önceki soruyla birleştirilmiş devam sorusu)."""
        context = self.retriever.retrieve(arama or question, kitle=kitle, birim=birim)
        response = ollama.chat(
            model=self.model,
            messages=build_messages(question, context, system=SYSTEM_PROMPT_V2),
            think=False,  # Qwen3 tabanlı modellerde zorunlu; diğerlerinde etkisiz.
            options={"temperature": 0.2, "num_ctx": 8192},
        )
        answer = response["message"]["content"].strip()
        if answer.startswith(REFUSAL_ANSWER.split(".")[0]) and is_staff(question, kitle):
            answer = STAFF_REFUSAL  # Eğitimdeki red cümlesi öğrenciyi Öğrenci İşleri'ne yönlendiriyor.
        if dogrula:
            answer = verify(answer, context)[0]
        return answer, context


# Önceki soruya dayanan kısa devam sorusu: "peki süresi ne kadar", "ya bütünlemede?", "bunun için ne lazım"
FOLLOW_UP_RE = re.compile(r"^(peki|ya |ve |bunun|bunu|buna|onun|onu|o zaman|o halde|ayrica|bir de|peki ya)"
                          r"|\b(bunun|bunda|bunlar|onun|orada|o durumda)\b")
AMBIGUITY_PROBE = 6  # netleştirme kararı için bakılan en iyi bağlam birimi sayısı
SKIP_WORDS_RE = re.compile(r"^(fark etmez|farketmez|hepsi|genel|bilmiyorum|vazgec|iptal)\b")


class Sohbet:
    """Tek kullanıcının sohbeti: önceki soruları ve öğrenilen birimi hatırlar, belirsiz soruda netleştirir.

    Netleştirme kodla yapılır (model eğitilmedi): birim bilinmiyorken soru birden fazla fakülte/MYO'nun kendi
    belgesine (staj, İME, bitirme projesi) gidiyorsa cevap yerine "hangi birim?" sorulur. Kullanıcı birimini
    söyleyince bekleyen soru o birimle cevaplanır ve birim sohbet boyunca hatırlanır.
    """

    def __init__(self, assistant: Assistant, kitle: str | None = None, birim: str | None = None) -> None:
        self.assistant = assistant
        self.kitle = kitle
        self.birim = birim if birim != "genel" else None
        self.birim_secildi = birim is not None  # arayüzde seçildiyse ("genel" dahil) bir daha sorulmaz
        self.gecmis: list[dict[str, str]] = []  # {"soru", "cevap"}
        self.bekleyen: dict[str, Any] | None = None  # netleştirme bekleyen soru ve seçenekler

    def sor(self, metin: str) -> dict[str, Any]:
        """Dönüş: {"tur": "cevap"|"netlestirme", "metin", "baglam": [...], "secenekler": [(anahtar, ad)]}."""
        metin = metin.strip()
        if self.bekleyen is not None:
            secim = self._secimi_coz(metin, self.bekleyen["secenekler"])
            bekleyen, self.bekleyen = self.bekleyen, None
            if secim is not None:  # netleştirme cevabı: bekleyen soruyu bu birimle cevapla
                self.birim, self.birim_secildi = (None if secim == "genel" else secim), True
                return self._cevapla(bekleyen["soru"], bekleyen["arama"])
            if SKIP_WORDS_RE.match(normalize_for_matching(metin)):
                self.birim_secildi = True  # birim vermek istemedi: genel kurallarla cevapla
                return self._cevapla(bekleyen["soru"], bekleyen["arama"])
            # Seçenek değil, yeni bir soru: bekleyen soru bırakılır.

        mentioned = units_in(metin)
        if mentioned:  # soruda birim geçti ("ziraat fakültesinde staj"): sonraki sorular için de hatırla
            self.birim, self.birim_secildi = mentioned[0], True
        arama = self._arama_metni(metin)
        if not self.birim_secildi:
            secenekler = self._belirsiz_birimler(arama)
            if len(secenekler) >= 2:
                self.bekleyen = {"soru": metin, "arama": arama, "secenekler": secenekler}
                adlar = "\n".join(f"{i}) {ad}" for i, (_, ad) in enumerate(secenekler, start=1))
                return {"tur": "netlestirme", "baglam": [], "secenekler": secenekler,
                        "metin": "Bu konuda birimlere göre farklı kurallar var. Hangi fakülte / yüksekokul için "
                                 f"soruyorsun?\n{adlar}\n(Numarasını ya da adını yazabilirsin.)"}
        return self._cevapla(metin, arama)

    def _cevapla(self, soru: str, arama: str) -> dict[str, Any]:
        cevap, baglam = self.assistant.answer(soru, kitle=self.kitle, birim=self.birim, arama=arama)
        self.gecmis.append({"soru": soru, "cevap": cevap})
        return {"tur": "cevap", "metin": cevap, "baglam": baglam, "secenekler": []}

    def _arama_metni(self, metin: str) -> str:
        """Devam sorusu ("peki süresi ne kadar?") tek başına aranamaz: önceki soruyla birleştirilir."""
        if not self.gecmis:
            return metin
        # Yalnızca bağlantı sözcüğüyle ("peki", "bunun") ya da 1-2 kelimelik ("bütünlemede?") sorular devamdır;
        # "kaydımı dondurabilir miyim" kısa ama kendi başına yeni bir sorudur.
        kisa = len(metin.split()) <= 2
        if kisa or FOLLOW_UP_RE.search(normalize_for_matching(metin)):
            return f"{self.gecmis[-1]['soru']} {metin}"
        return metin

    def _belirsiz_birimler(self, arama: str) -> list[tuple[str, str]]:
        """Birim filtresi olmadan aranır; en iyi sonuçlar ≥2 farklı fakülte/MYO'nun kendi belgesindeyse
        (genel belgeler öne çıkmıyorsa) bu birimler netleştirme seçeneği olur."""
        where = None if is_staff(arama, self.kitle) else STUDENT_AUDIENCE
        top = self.assistant.retriever.retrieve(arama, size=AMBIGUITY_PROBE, where=where)
        birimler = [a["ozellikler"]["birim"] for a in top]
        ozel = [b for b in birimler if b not in ("tum", "enstitu")]
        if not ozel or birimler[0] == "tum" and len(ozel) < AMBIGUITY_PROBE // 2:
            return []
        sirali = list(dict.fromkeys(ozel))  # en iyi eşleşen birim önce
        if len(sirali) < 2:
            return []
        secenekler = [(b, BIRIMLER.get(b, b)) for b in sirali]
        return secenekler + [("genel", BIRIMLER["genel"])]

    @staticmethod
    def _secimi_coz(metin: str, secenekler: list[tuple[str, str]]) -> str | None:
        normalized = normalize_for_matching(metin)
        if normalized.isdigit() and 1 <= int(normalized) <= len(secenekler):
            return secenekler[int(normalized) - 1][0]
        for anahtar, ad in secenekler:  # "mühendislik", "ziraat fakültesi", "kartepe"
            ilk = normalize_for_matching(ad).split()[0]
            if ilk in normalized.split() or normalize_for_matching(ad) in normalized:
                return anahtar
        mentioned = units_in(metin)
        return mentioned[0] if mentioned else None


def describe(article: dict[str, Any]) -> str:
    fields = article["ozellikler"]
    return f"{fields['belge_adi']} | {fields['madde_etiketi']} {fields['madde_basligi']}"


def main() -> int:
    args = sys.argv[1:]
    options: dict[str, str | None] = {"--model": CHAT_MODEL, "--kitle": None, "--birim": None}
    for flag in options:
        if flag in args:
            position = args.index(flag)
            options[flag] = args[position + 1]
            del args[position:position + 2]
    model, kitle, birim = options["--model"], options["--kitle"], options["--birim"]

    if "--arama" in args:
        args.remove("--arama")
        question = " ".join(args)
        print(f"filtre: {scope_filter(question, kitle, birim)}")
        for article in Retriever().retrieve(question, kitle=kitle, birim=birim):
            print(" -", describe(article))
        return 0

    assistant = Assistant(model)
    if "--sohbet" in args:
        print(f"KOÜ asistanı v2 ({model}) — çıkmak için boş satır.\n")
        sohbet = Sohbet(assistant, kitle=kitle, birim=birim)
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

    if not args:
        print(__doc__)
        return 1
    answer, context = assistant.answer(" ".join(args), kitle=kitle, birim=birim)
    print(answer)
    print("\n[Bağlam:\n  " + "\n  ".join(describe(a) for a in context) + "]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
