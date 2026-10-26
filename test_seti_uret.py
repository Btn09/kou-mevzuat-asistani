#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 test seti (PLAN_v3.md Faz 0): qwen3:14b gerçek kullanıcı dilinde soru ve sohbet senaryosu üretir.

Tek soru (~90; ~%60 hoca/personel, ~%40 öğrenci), çok turlu sohbet (20) ve kapsam dışı soru (10).
- Maddeler eğitimde (raft_train.jsonl) hiç görülmemiş olanlardan seçilir; seçilen maddeler test_maddeleri.json'a
  yazılır ve v3 eğitim verisinden çıkarılmalıdır.
- Rol sorulmayan sistem için: soru kimin ağzından olursa olsun aynı akıştan geçer.
- Fakülte/MYO'ya özel maddelerde soruların yarısında birim adı geçmez (gerçekçi: "staj defterini ne zaman
  teslim ederim"); sistem ya doğrudan doğru birimi bulmalı ya da netleştirme sorup cevaplamalı.
- Her soru: alıntı doğrulaması (cevabın dayandığı cümle maddede birebir geçmeli) + hakem kontrolü.

Kullanım:  python -X utf8 test_seti_uret.py      Çıktı: veri/test_v3/test_seti.jsonl
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import ollama

from rag_v2 import BIRIMLER as BIRIM_ADLARI

ROOT = Path(__file__).resolve().parent.parent
ARTICLES = ROOT / "veri" / "islenmis" / "maddeler.jsonl"
TRAIN = ROOT / "veri" / "egitim" / "raft_train.jsonl"
OUT_DIR = ROOT / "veri" / "test_v3"
MODEL = "qwen3:14b"
SEED = 7
N_STAFF, N_STUDENT = 60, 40  # aday sayısı (filtreden sonra ~%90'ı kalır)
N_FOLLOW_UP, N_CLARIFY, N_SWITCH = 8, 6, 6
MAX_PER_DOC = 4
GENERAL_UNITS = {"tum", "enstitu"}

PERSONA = {
    "personel": "Kocaeli Üniversitesi'nde çalışan bir öğretim üyesi veya idari personel",
    "ogrenci": "Kocaeli Üniversitesi öğrencisi",
}
QA_SCHEMA = {"type": "object", "properties": {"soru": {"type": "string"}, "cevap": {"type": "string"},
                                              "alinti": {"type": "string"}},
             "required": ["soru", "cevap", "alinti"]}
PAIR_SCHEMA = {"type": "object", "properties": {
    "soru1": {"type": "string"}, "cevap1": {"type": "string"}, "alinti1": {"type": "string"},
    "soru2": {"type": "string"}, "cevap2": {"type": "string"}, "alinti2": {"type": "string"}},
    "required": ["soru1", "cevap1", "alinti1", "soru2", "cevap2", "alinti2"]}
JUDGE_SCHEMA = {"type": "object", "properties": {"cevaplanabilir": {"type": "boolean"}, "dogal": {"type": "boolean"},
                                                 "neden": {"type": "string"}},
                "required": ["cevaplanabilir", "dogal", "neden"]}

# Sistemin kapsamı dışında, eğitim verisindeki listelerde olmayan sorular (elle yazıldı).
OUT_OF_SCOPE = [
    "Kampüste otopark ücreti ne kadar, aylık kart alabiliyor muyuz?",
    "Umuttepe yerleşkesine Gebze'den hangi otobüs gidiyor?",
    "Bu dönem maaşlara ne kadar zam yapıldı?",
    "Kütüphanede grup çalışma odası nasıl rezerve edilir?",
    "Rektörlük binasındaki toplantı salonunu nasıl ayırtırım?",
    "Öğrenci kulübü kurmak istiyorum, hangi formu doldurmam gerekiyor?",
    "Personel servisi sabah kaçta kalkıyor?",
    "Yemekhanede bu hafta menüde ne var?",
    "Üniversitenin e-posta şifremi unuttum, nasıl sıfırlarım?",
    "Kampüste spor salonu üyeliği için ücret ödüyor muyuz?",
]


def normalize(text: str) -> str:
    text = text.replace("İ", "i").replace("I", "ı").lower()
    return re.sub(r"[^\w]+", " ", text).strip()


def quote_ok(quote: str, text: str) -> bool:
    """Alıntının en az 5 kelimesi maddede ardışık geçmeli (model noktalama/harf değiştirebiliyor)."""
    words = normalize(quote).split()
    if len(words) < 5:
        return False
    body = " " + normalize(text) + " "
    return any(" " + " ".join(words[i:i + 5]) + " " in body for i in range(len(words) - 4))


def chat_json(prompt: str, schema: dict[str, Any]) -> dict[str, Any] | None:
    try:
        response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}], format=schema,
                               think=False, options={"temperature": 0.7, "num_ctx": 8192})
        return json.loads(response["message"]["content"])
    except Exception as exc:  # bozuk JSON vb.: bu aday atlanır
        print(f"   uyarı: {exc}", flush=True)
        return None


def unit_brief(unit: dict[str, Any]) -> str:
    fields = unit["ozellikler"]
    return f"[{fields['belge_adi']} — {fields['madde_etiketi']} {fields['madde_basligi']}]\n{unit['metin']}"


def gen_single(unit: dict[str, Any], mention_unit: bool | None) -> dict[str, Any] | None:
    kitle = "personel" if unit["ozellikler"]["hedef_kitle"] == "personel" else "ogrenci"
    if mention_unit is None:
        unit_rule = ""
    elif mention_unit:
        # Örnek olarak sabit bir fakülte adı vermek yanlış birim adı yazdırıyordu ("Ziraat'te stajım" -> MYO maddesi).
        name = BIRIM_ADLARI.get(unit["ozellikler"]["birim"], unit["ozellikler"]["birim"])
        unit_rule = f"- Soruda birimin adını ({name}) doğal biçimde geçir.\n"
    else:
        unit_rule = ("- Soruda fakülte, yüksekokul ya da bölüm adı GEÇMESİN; kişi kendi birimini söylemeden "
                     "sorsun (ör. 'staj defterimi ne zaman teslim etmeliyim').\n")
    prompt = (
        f"Aşağıdaki mevzuat maddesini okuyan değil, bu konuyu merak eden {PERSONA[kitle]} gibi düşün.\n\n"
        f"{unit_brief(unit)}\n\n"
        "Bu maddeyle cevaplanabilen TEK bir soru yaz:\n"
        "- Gerçek bir kullanıcının sohbet kutusuna yazacağı gibi doğal, kısa, günlük dilde olsun.\n"
        "- Madde metnindeki ifadeleri birebir kopyalama; madde numarası veya belge adı yazma.\n"
        "- Tek ve net bir bilgi sorsun (süre, şart, kim karar verir, hangi belge gerekir vb.).\n"
        f"{unit_rule}"
        "- cevap: maddeye göre doğru, kısa cevap. alinti: cevabın dayandığı ifade, maddeden BİREBİR kopyala.\n"
        "JSON ver."
    )
    result = chat_json(prompt, QA_SCHEMA)
    if not result or not quote_ok(result["alinti"], unit["metin"]):
        return None
    return {"soru": result["soru"].strip(), "cevap": result["cevap"].strip(), "kitle": kitle}


def judge(question: str, unit: dict[str, Any], unit_named_required: bool) -> bool:
    note = ("Soruda fakülte adı geçmemesi normaldir; bunu sorun sayma." if not unit_named_required else "")
    prompt = (
        f"{unit_brief(unit)}\n\nSORU: {question}\n\n"
        "- cevaplanabilir: Bu soru yukarıdaki maddeyle doğru ve eksiksiz cevaplanabilir mi?\n"
        "- dogal: Gerçek bir kullanıcının soracağı, tek başına anlaşılır bir soru mu? " + note + "\n"
        "Emin değilsen false. JSON ver."
    )
    try:
        response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}], format=JUDGE_SCHEMA,
                               think=False, options={"temperature": 0, "num_ctx": 8192})
        verdict = json.loads(response["message"]["content"])
        return bool(verdict["cevaplanabilir"] and verdict["dogal"])
    except Exception:
        return False


def gen_pair(first: dict[str, Any], second: dict[str, Any], kind: str) -> dict[str, Any] | None:
    kitle = "personel" if first["ozellikler"]["hedef_kitle"] == "personel" else "ogrenci"
    if kind == "devam":
        rule2 = ("- soru2, soru1'in DEVAMI olsun: kısa ve önceki soruya dayansın, konuyu tekrar etmesin "
                 "(ör. 'peki bunun süresi ne kadar?', 'ya raporu kime teslim ediyorum?'). İkinci madde ile cevaplanmalı.")
    else:
        rule2 = ("- soru2 tamamen YENİ bir konu olsun ve tek başına anlaşılır olsun (önceki soruya dayanmasın); "
                 "ikinci madde ile cevaplanmalı.")
    prompt = (
        f"Bir sohbet asistanıyla konuşan {PERSONA[kitle]} gibi iki ardışık mesaj yaz.\n\n"
        f"BİRİNCİ MADDE:\n{unit_brief(first)}\n\nİKİNCİ MADDE:\n{unit_brief(second)}\n\n"
        "- soru1: birinci maddeyle cevaplanan doğal, kısa bir soru (madde numarası/belge adı yazma).\n"
        f"{rule2}\n"
        "- cevap1/cevap2: maddelere göre kısa doğru cevaplar. alinti1/alinti2: maddelerden BİREBİR ifade.\n"
        "JSON ver."
    )
    result = chat_json(prompt, PAIR_SCHEMA)
    if not result or not quote_ok(result["alinti1"], first["metin"]) or not quote_ok(result["alinti2"], second["metin"]):
        return None
    return result


def turn(message: str, unit: dict[str, Any] | None, answer: str = "", expect: str = "cevap") -> dict[str, Any]:
    fields = unit["ozellikler"] if unit else {}
    return {"mesaj": message, "beklenen": {"tip": expect, "unit_id": unit["id"] if unit else "",
                                           "birim": fields.get("birim", ""), "cevap_ozeti": answer}}


def main_web(n_staff: int = 18, n_student: int = 16) -> int:
    """--web: yalnızca KOÜ sitelerinden toplanan belgelerden (web_belgeleri.py) tek soruluk test seti.
    Belge başına 1 soru (çeşitlilik); birim adı yanlış yazılan sorular atılır. Çıktı: test_seti_web.jsonl"""
    from rag_v2 import units_in

    rng = random.Random(SEED + 1)
    pool = [json.loads(line) for line in ARTICLES.open(encoding="utf-8")]
    pool = [u for u in pool if u["ozellikler"]["belge_id"].startswith("web") and len(u["metin"]) >= 250]
    rng.shuffle(pool)
    chosen, docs = [], set()
    for want, is_staff in ((n_staff, True), (n_student, False)):
        count = 0
        for unit in pool:
            doc = unit["ozellikler"]["belge_id"]
            if doc in docs or (unit["ozellikler"]["hedef_kitle"] == "personel") != is_staff:
                continue
            docs.add(doc)
            chosen.append(unit)
            count += 1
            if count == want:
                break
    items = []
    for index, unit in enumerate(chosen, start=1):
        special = unit["ozellikler"]["birim"] not in GENERAL_UNITS
        mention = (index % 2 == 0) if special else None
        qa = gen_single(unit, mention)
        if not qa or not judge(qa["soru"], unit, unit_named_required=bool(mention)):
            continue
        found = units_in(qa["soru"])
        if found and unit["ozellikler"]["birim"] not in found:
            continue  # yanlış birim adı yazılmış
        items.append({"id": f"web_{index:03d}", "tur": "tek", "kitle": qa["kitle"],
                      "birim_soruda": (unit["ozellikler"]["birim"] in found) if special else None,
                      "turlar": [turn(qa["soru"], unit, qa["cevap"])]})
        if index % 10 == 0:
            print(f"  {index}/{len(chosen)} -> {len(items)} kabul", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "test_seti_web.jsonl").open("w", encoding="utf-8") as output:
        output.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
    staff = sum(1 for item in items if item["kitle"] == "personel")
    print(f"Web test seti: {len(items)} soru ({staff} personel) -> {OUT_DIR / 'test_seti_web.jsonl'}")
    return 0


# --yeni: tam veriyle gelen konular (harç, KYK, Erasmus, personel mevzuatı, Tıp, takvim, dersler); hiçbir
# eğitimde (v2, v4) ve önceki test setlerinde kullanılmamış maddelerden. Çıktı: test_seti_yeni.jsonl
NEW_TOPICS = [
    ("harc", 6, r"harc(?!ama|ir|an)|katki pay|ogrenim ucret", ("webf_", "webt_", "webd_")),
    ("kyk", 4, r"kredi|burs|yurt", ("mevzuat_1_5_5102", "mevzuat_7_5_40391", "mevzuat_7_5_40390", "mevzuat_7_5_38510")),
    ("personel", 8, r".", ("mevzuat_1_5_657", "mevzuat_1_5_2914", "mevzuat_7_5_")),
    ("erasmus", 5, r"erasmus|uluslararasi|yabanci uyruk|degisim", ("webt_int", "webf_", "webt_yos", "webd_int")),
    ("tip", 4, r"tip fakultesi|intorn|donem [iv]+|klinik", ("webt_atip", "webt_tip", "webf_", "kou_tip")),
    ("takvim", 3, r"2026.2027.*(takvim|kayit|sinav)|(takvim|kayit yenileme).*2026", ("webf_", "webd_", "webt_")),
    ("ders", 5, r".", ("aktsders_",)),
]


def main_yeni() -> int:
    from pdf_to_chromadb import normalize_for_matching
    from rag_v2 import units_in
    from rag_v3 import REPORT_RE

    rng = random.Random(SEED + 2)
    pool = [json.loads(line) for line in ARTICLES.open(encoding="utf-8")]
    pool += [json.loads(line) for line in ARTICLES.with_name("dersler.jsonl").open(encoding="utf-8")]
    used: set[str] = set()
    egitim = ROOT / "veri"
    for path in (TRAIN, egitim / "egitim" / "raft_test.jsonl", egitim / "egitim_v4" / "raft_train_v4.jsonl",
                 egitim / "egitim_v4" / "raft_eval_v4.jsonl"):
        used |= {json.loads(line)["meta"].get("unit_id") for line in path.open(encoding="utf-8")}
    used |= {json.loads(line)["id"] for line in (egitim / "egitim_v4" / "qa_raw_v4.jsonl").open(encoding="utf-8")}
    for name in ("test_seti.jsonl", "test_seti_web.jsonl"):
        for line in (OUT_DIR / name).open(encoding="utf-8"):
            used |= {t["beklenen"].get("unit_id") for t in json.loads(line)["turlar"]}
    rng.shuffle(pool)
    items, docs = [], set()
    for topic, want, pattern, prefixes in NEW_TOPICS:
        regex = re.compile(pattern)
        count = 0
        for unit in pool:
            fields = unit["ozellikler"]
            text = normalize_for_matching(fields["belge_adi"] + " " + unit["metin"][:800])
            if (unit["id"] in used or fields["belge_id"] in docs or not fields["belge_id"].startswith(prefixes)
                    or len(unit["metin"]) < 250 or not regex.search(text) or "OZET" in unit["id"]
                    or REPORT_RE.search(normalize_for_matching(fields["belge_adi"]))):
                continue
            special = fields["birim"] not in GENERAL_UNITS
            mention = (count % 2 == 0) if special else None
            qa = gen_single(unit, mention)
            if not qa or not judge(qa["soru"], unit, unit_named_required=bool(mention)):
                continue
            found = units_in(qa["soru"])
            if found and fields["birim"] not in found:
                continue
            docs.add(fields["belge_id"])
            items.append({"id": f"yeni_{topic}_{count + 1:02d}", "tur": "tek", "kitle": qa["kitle"],
                          "birim_soruda": (fields["birim"] in found) if special else None,
                          "turlar": [turn(qa["soru"], unit, qa["cevap"])]})
            count += 1
            print(f"  {topic}: {count}/{want} {qa['soru'][:70]}", flush=True)
            if count == want:
                break
    with (OUT_DIR / "test_seti_yeni.jsonl").open("w", encoding="utf-8") as output:
        output.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
    print(f"Yeni test seti: {len(items)} soru -> {OUT_DIR / 'test_seti_yeni.jsonl'}")
    return 0


def main() -> int:
    if "--web" in sys.argv[1:]:
        return main_web()
    if "--yeni" in sys.argv[1:]:
        return main_yeni()
    BIRIMLER = BIRIM_ADLARI  # netleştirme cevabında kullanıcı birim adını yazar

    rng = random.Random(SEED)
    units = [json.loads(line) for line in ARTICLES.open(encoding="utf-8")]
    train_ids = {json.loads(line)["meta"].get("unit_id") for line in TRAIN.open(encoding="utf-8")}
    pool = [u for u in units if u["id"] not in train_ids and u["ozellikler"]["konu"] != "genel_hukumler"
            and len(u["metin"]) >= 200 and not u["ozellikler"]["madde_etiketi"].startswith("GEÇİCİ")]
    rng.shuffle(pool)

    def take(candidates: list[dict[str, Any]], n: int, used: set[str], cap: int = MAX_PER_DOC) -> list[dict[str, Any]]:
        per_doc: dict[str, int] = {}
        chosen = []
        for unit in candidates:
            doc = unit["ozellikler"]["belge_id"]
            if unit["id"] in used or per_doc.get(doc, 0) >= cap:
                continue
            if doc.startswith("kdv") and per_doc.get(doc, 0) >= 2:  # yürürlükten kalkmış tebliğ: en fazla 2
                continue
            per_doc[doc] = per_doc.get(doc, 0) + 1
            chosen.append(unit)
            used.add(unit["id"])
            if len(chosen) == n:
                break
        return chosen

    used: set[str] = set()
    # Personel belgeleri az sayıda ama büyük (4734, 2547, harcırah, BAP): belge başına sınır daha yüksek.
    staff = take([u for u in pool if u["ozellikler"]["hedef_kitle"] == "personel"], N_STAFF, used, cap=7)
    student = take([u for u in pool if u["ozellikler"]["hedef_kitle"] != "personel"], N_STUDENT, used)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []

    print(f"Tek sorular: {len(staff)} personel + {len(student)} öğrenci aday", flush=True)
    for index, unit in enumerate(staff + student, start=1):
        special = unit["ozellikler"]["birim"] not in GENERAL_UNITS
        mention = (index % 2 == 0) if special else None
        qa = gen_single(unit, mention)
        if qa and judge(qa["soru"], unit, unit_named_required=bool(mention)):
            items.append({"id": f"tek_{index:03d}", "tur": "tek", "kitle": qa["kitle"],
                          "birim_soruda": bool(mention) if special else None,
                          "turlar": [turn(qa["soru"], unit, qa["cevap"])]})
        if index % 10 == 0:
            print(f"  {index}/{len(staff) + len(student)} -> {len(items)} kabul", flush=True)

    # Çok turlu: devam sorusu (aynı belge), netleştirme (birimsiz fakülte sorusu), konu değiştirme (farklı belge).
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for unit in pool:
        if unit["id"] not in used:
            by_doc.setdefault(unit["ozellikler"]["belge_id"], []).append(unit)
    multi_docs = [d for d, us in by_doc.items() if len(us) >= 2]
    rng.shuffle(multi_docs)
    special_docs = [d for d in multi_docs if by_doc[d][0]["ozellikler"]["birim"] not in GENERAL_UNITS]
    general_docs = [d for d in multi_docs if d not in special_docs]

    scenario = 0
    def add_scenario(kind: str, turns: list[dict[str, Any]], kitle: str) -> None:
        nonlocal scenario
        scenario += 1
        items.append({"id": f"sohbet_{scenario:02d}_{kind}", "tur": "cok_turlu", "senaryo": kind,
                      "kitle": kitle, "birim_soruda": None, "turlar": turns})

    print("Çok turlu senaryolar...", flush=True)
    made = {"devam": 0, "netlestirme": 0, "konu_degisimi": 0}
    for doc in general_docs + special_docs:
        if made["devam"] >= N_FOLLOW_UP:
            break
        first, second = by_doc[doc][:2]
        pair = gen_pair(first, second, "devam")
        if pair:
            kitle = "personel" if first["ozellikler"]["hedef_kitle"] == "personel" else "ogrenci"
            add_scenario("devam", [turn(pair["soru1"], first, pair["cevap1"]),
                                   turn(pair["soru2"], second, pair["cevap2"])], kitle)
            made["devam"] += 1
            by_doc[doc] = by_doc[doc][2:]
    for doc in special_docs:
        if made["netlestirme"] >= N_CLARIFY or len(by_doc[doc]) < 2:
            continue
        first, second = by_doc[doc][:2]
        qa = gen_single(first, mention_unit=False)
        pair = gen_pair(first, second, "devam") if qa else None
        if qa and pair:
            name = BIRIMLER.get(first["ozellikler"]["birim"], first["ozellikler"]["birim"])
            add_scenario("netlestirme", [turn(qa["soru"], first, qa["cevap"], expect="netlestirme_veya_cevap"),
                                         turn(name, first, qa["cevap"]),
                                         turn(pair["soru2"], second, pair["cevap2"])], "ogrenci")
            made["netlestirme"] += 1
            by_doc[doc] = by_doc[doc][2:]
    docs_left = [d for d in multi_docs if by_doc[d]]
    for i in range(0, len(docs_left) - 1, 2):
        if made["konu_degisimi"] >= N_SWITCH:
            break
        first, second = by_doc[docs_left[i]][0], by_doc[docs_left[i + 1]][0]
        if first["ozellikler"]["birim"] not in GENERAL_UNITS or second["ozellikler"]["birim"] not in GENERAL_UNITS:
            continue  # konu değişiminde birim netleştirmesi karışmasın
        pair = gen_pair(first, second, "konu_degisimi")
        if pair:
            kitle = "personel" if first["ozellikler"]["hedef_kitle"] == "personel" else "ogrenci"
            add_scenario("konu_degisimi", [turn(pair["soru1"], first, pair["cevap1"]),
                                           turn(pair["soru2"], second, pair["cevap2"])], kitle)
            made["konu_degisimi"] += 1
    print(f"  {made}", flush=True)

    for index, question in enumerate(OUT_OF_SCOPE, start=1):
        items.append({"id": f"disi_{index:02d}", "tur": "kapsam_disi", "kitle": "", "birim_soruda": None,
                      "turlar": [turn(question, None, expect="red")]})

    with (OUT_DIR / "test_seti.jsonl").open("w", encoding="utf-8") as output:
        output.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
    test_units = sorted({t["beklenen"]["unit_id"] for item in items for t in item["turlar"] if t["beklenen"]["unit_id"]})
    (OUT_DIR / "test_maddeleri.json").write_text(json.dumps(test_units, ensure_ascii=False, indent=0), encoding="utf-8")
    counts: dict[str, int] = {}
    for item in items:
        key = item["tur"] + ("/" + item["kitle"] if item["tur"] == "tek" else "")
        counts[key] = counts.get(key, 0) + 1
    print(f"\nTest seti: {len(items)} öğe {counts} -> {OUT_DIR / 'test_seti.jsonl'}")
    print(f"Test maddeleri (v3 eğitiminden çıkarılacak): {len(test_units)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
