#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""web_tara.py çıktısını ayıklar: işe yaramayan sayfaları ve elimizdeki PDF'lerle aynı olanları çıkarır.

1. Değersiz sayfa: başlığı tanıtım/kadro/iletişim türü ("Hakkımızda", "Misyon", "Akademik Personel" ...)
   ya da metninin çoğu isim/unvan listesi olan sayfalar.
2. Mevcut veriyle aynı: sayfa metninin kelime 8'lilerinin (shingle) %40'ından fazlası maddeler.jsonl'de geçiyorsa
   (ör. Kalite Yönergesi hem PDF hem web sayfası olarak var).
3. Web sayfaları arasında kopya: aynı metin birden fazla sitede (fakülte siteleri ortak sayfa paylaşıyor).

Çıktı: veri/web/secilen.json (kalan sayfalar, gerekçeli), veri/web/ayiklama_raporu.tsv (her sayfanın kararı).
Kullanım:  python -X utf8 web_ayikla.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "veri" / "web"
ARTICLES = ROOT / "veri" / "islenmis" / "maddeler.jsonl"
SHINGLE = 8
EXISTING_OVERLAP = 0.40
COPY_OVERLAP = 0.70

LOW_VALUE_TITLE = re.compile(
    r"hakkımızda|hakkında|misyon|vizyon|tarihçe|tarihce|iletişim|akademik personel|idari personel|kadro|"
    r"personelimiz|organizasyon|teşkilat şeması|yönetim kurulu üye|fakülte kurulu üye|senato üye|komisyon üye|"
    r"üyeleri|bölüm başkan|dekan|müdür|rektör|mesaj|galeri|basın|haber|duyuru|etkinlik|laboratuvar|"
    r"değerlerimiz|politika|kalite elçi|temsilci|öğretim elemanları|ders programı|sınav programı|ders içerikleri|"
    r"ders planı|müfredat|öğretim planı|kalite el|amaç ve hedef|stratejik plan|faaliyet raporu|toplantı",
    re.IGNORECASE)
TITLE_WORD_RE = re.compile(r"\b(Prof\.|Doç\.|Dr\.|Öğr\.|Arş\.|Gör\.|Uzm\.|Okt\.)")


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.replace("İ", "i").replace("I", "ı").lower())


def shingles(text: str) -> set[tuple[str, ...]]:
    w = words(text)
    return {tuple(w[i:i + SHINGLE]) for i in range(max(0, len(w) - SHINGLE + 1))}


def body(path: Path) -> tuple[str, str, str]:
    raw = path.read_text(encoding="utf-8")
    head, _, text = raw.partition("\n\n")
    url = re.search(r"KAYNAK: (.*)", head).group(1)
    title = re.search(r"BAŞLIK: (.*)", head).group(1)
    return url, title, text


LLM_CACHE = WEB / "konu_kontrol.json"
CLASSIFY_SCHEMA = {"type": "object", "properties": {
    "faydali": {"type": "boolean"}, "kitle": {"type": "string", "enum": ["ogrenci", "personel", "herkes"]},
    "ozet": {"type": "string"}, "neden": {"type": "string"}}, "required": ["faydali", "kitle", "ozet", "neden"]}


def classify(title: str, text: str) -> dict:
    import ollama

    prompt = (
        "Kocaeli Üniversitesi web sitesinden bir sayfa. Öğrencilerin ve personelin (hoca, idari) kurallar, işlemler "
        "ve süreçler hakkındaki sorularını cevaplayan bir asistanın bilgi kaynağına eklenmeli mi?\n\n"
        f"BAŞLIK: {title}\nMETİN (ilk kısım):\n{text[:3500]}\n\n"
        "- faydali: Sayfa kural, süreç, başvuru koşulu, süre/tarih, gerekli belge, ücret, sorumlu birim, iş akışı, "
        "sıkça sorulan soru gibi BİRİNİN SORABİLECEĞİ somut bilgi içeriyorsa true. Tanıtım, tarihçe, program/bölüm "
        "tanıtımı, ders içeriği, kişi/iletişim listesi, haber, faaliyet raporu, kulüp tanıtımı, yalnızca bağlantı "
        "listesi ise false.\n- kitle: bu bilgiyi kim sorar.\n- ozet: tek cümle konu.\n- neden: tek cümle.\nJSON ver."
    )
    try:
        response = ollama.chat(model="qwen3:14b", messages=[{"role": "user", "content": prompt}],
                               format=CLASSIFY_SCHEMA, think=False, options={"temperature": 0, "num_ctx": 6144})
        return json.loads(response["message"]["content"])
    except Exception as exc:
        return {"faydali": False, "kitle": "", "ozet": "", "neden": f"hata: {exc}"}


def main() -> int:
    existing: set[tuple[str, ...]] = set()
    with ARTICLES.open(encoding="utf-8") as input_file:
        for line in input_file:
            existing |= shingles(json.loads(line)["metin"])
    pages = sorted((WEB / "sayfalar").glob("*.txt"))
    decisions, kept, kept_shingles = [], [], []
    for path in pages:
        url, title, text = body(path)
        sh = shingles(text)
        reason = ""
        if LOW_VALUE_TITLE.search(title):
            reason = "değersiz başlık"
        elif len(TITLE_WORD_RE.findall(text)) > len(text) / 300:
            reason = "isim/unvan listesi"
        elif sh and len(sh & existing) / len(sh) > EXISTING_OVERLAP:
            reason = f"elimizde var (%{100 * len(sh & existing) / len(sh):.0f} örtüşme)"
        else:
            for other_path, other in kept_shingles:
                if sh and len(sh & other) / len(sh) > COPY_OVERLAP:
                    reason = f"kopya: {other_path.name}"
                    break
        decisions.append((path.name, url, title, reason or "ALINDI", len(text)))
        if not reason:
            kept.append({"dosya": path.name, "url": url, "baslik": title, "uzunluk": len(text)})
            kept_shingles.append((path, sh))
    # 4. Konu kontrolü (qwen3:14b): tanıtım, program açıklaması, kulüp, faaliyet haberi gibi sayfalar elenir;
    # kural/süreç/başvuru/süre/belge/ücret içerenler kalır. Sonuç önbelleğe yazılır (yeniden çalıştırmada sorulmaz).
    cache = json.loads(LLM_CACHE.read_text(encoding="utf-8")) if LLM_CACHE.exists() else {}
    final = []
    for index, page in enumerate(kept, start=1):
        if page["dosya"] not in cache:
            _, title, text = body(WEB / "sayfalar" / page["dosya"])
            cache[page["dosya"]] = classify(title, text)
            LLM_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
        verdict = cache[page["dosya"]]
        if verdict.get("faydali"):
            final.append({**page, "kitle": verdict.get("kitle", ""), "ozet": verdict.get("ozet", "")})
        else:
            decisions[[d[0] for d in decisions].index(page["dosya"])] = (
                page["dosya"], page["url"], page["baslik"], f"konu dışı: {verdict.get('neden', '')[:80]}", page["uzunluk"])
        if index % 50 == 0:
            print(f"  konu kontrolü: {index}/{len(kept)} ({len(final)} faydalı)", flush=True)
    kept = final

    with (WEB / "ayiklama_raporu.tsv").open("w", encoding="utf-8") as output:
        output.write("dosya\turl\tbaşlık\tkarar\tuzunluk\n")
        output.writelines("\t".join(map(str, d)) + "\n" for d in decisions)
    (WEB / "secilen.json").write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")
    counts: dict[str, int] = {}
    for d in decisions:
        key = d[3].split(" (")[0].split(":")[0]
        counts[key] = counts.get(key, 0) + 1
    print(f"Toplam {len(decisions)} sayfa -> {counts}")
    print(f"Seçilen: {len(kept)} -> {WEB / 'secilen.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
